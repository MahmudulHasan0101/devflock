"""Local per-worker gateway.

Claude Code (via the Agent SDK) talks to ``http://127.0.0.1:<port>``. The
gateway forwards everything to the worker's current upstream (a Kaggle
tunnel URL + key), and:

  * retries idempotent failures (connection errors, 429/502/503/524) with
    backoff *before* any bytes have gone to the client -- once streaming has
    started we can no longer retry transparently, so we only retry the
    connect/prefill phase;
  * lets the orchestrator hot-swap the upstream URL/key (new Kaggle session,
    new tunnel) via an admin endpoint, without restarting the Claude Code
    subprocess that's talking to this port;
  * injects the real API key, so it never sits in the worker subprocess's
    environment inspectable by tools it runs;
  * records simple per-request timing/byte counters for the TUI and ledger.

One Gateway instance == one worker. Run many on different ports.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import Optional

import aiohttp
from aiohttp import web

logger = logging.getLogger("devflock.gateway")

RETRIABLE_STATUS = {429, 502, 503, 504, 524}
RETRY_BACKOFFS = [0.5, 1.5, 4.0, 8.0]  # seconds, before first byte only


@dataclasses.dataclass
class Upstream:
    base_url: str
    api_key: str
    auth_style: str = "x-api-key"  # or "bearer"

    def headers(self, incoming: dict) -> dict:
        h = {k: v for k, v in incoming.items()
             if k.lower() not in ("host", "content-length", "x-api-key", "authorization")}
        if self.auth_style == "bearer":
            h["authorization"] = f"Bearer {self.api_key}"
        else:
            h["x-api-key"] = self.api_key
        return h


@dataclasses.dataclass
class GatewayStats:
    requests: int = 0
    retries: int = 0
    failures: int = 0
    bytes_out: int = 0
    last_ok: Optional[float] = None
    last_error: Optional[str] = None


class Gateway:
    def __init__(self, worker_id: str, upstream: Upstream, port: int = 0,
                 on_upstream_dead: "asyncio.Future | None" = None):
        self.worker_id = worker_id
        self.upstream = upstream
        self.port = port
        self.stats = GatewayStats()
        self._session: aiohttp.ClientSession | None = None
        self._app = web.Application(client_max_size=64 * 1024 * 1024)
        self._app.router.add_route("*", "/_devflock/upstream", self._set_upstream)
        self._app.router.add_route("*", "/_devflock/stats", self._get_stats)
        self._app.router.add_route("*", "/{path:.*}", self._proxy)
        self._runner: web.AppRunner | None = None
        self._on_upstream_dead = on_upstream_dead
        self._consecutive_dead = 0

    async def start(self) -> int:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=None))
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        # discover actual bound port when self.port was 0
        for sock in self._runner.addresses:
            pass
        sockets = self._runner.sites
        if self.port == 0:
            self.port = list(sockets)[0]._server.sockets[0].getsockname()[1]
        logger.info("gateway[%s] listening on 127.0.0.1:%d -> %s",
                    self.worker_id, self.port, self.upstream.base_url)
        return self.port

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
        if self._session:
            await self._session.close()

    async def _set_upstream(self, request: web.Request) -> web.Response:
        """POST {"base_url": "...", "api_key": "...", "auth_style": "x-api-key"}"""
        data = await request.json()
        self.upstream = Upstream(
            base_url=data["base_url"], api_key=data["api_key"],
            auth_style=data.get("auth_style", self.upstream.auth_style))
        self._consecutive_dead = 0
        logger.info("gateway[%s] upstream swapped -> %s", self.worker_id, self.upstream.base_url)
        return web.json_response({"ok": True})

    async def _get_stats(self, request: web.Request) -> web.Response:
        return web.json_response(dataclasses.asdict(self.stats))

    async def _proxy(self, request: web.Request) -> web.StreamResponse:
        path = request.match_info["path"]
        body = await request.read()
        url = self.upstream.base_url.rstrip("/") + "/" + path
        if request.query_string:
            url += "?" + request.query_string
        headers = self.upstream.headers(dict(request.headers))

        self.stats.requests += 1
        last_exc: Exception | None = None

        for attempt, backoff in enumerate([0.0] + RETRY_BACKOFFS):
            if backoff:
                self.stats.retries += 1
                await asyncio.sleep(backoff)
            try:
                upstream_resp = await self._session.request(
                    request.method, url, headers=headers, data=body or None,
                    allow_redirects=False)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                self.stats.last_error = f"{type(e).__name__}: {e}"
                continue

            if upstream_resp.status in RETRIABLE_STATUS and attempt < len(RETRY_BACKOFFS):
                self.stats.last_error = f"HTTP {upstream_resp.status} (retrying)"
                upstream_resp.release()
                continue

            # success (or a non-retriable/final status) -- stream it back verbatim
            self.stats.last_ok = time.time()
            self._consecutive_dead = 0
            resp = web.StreamResponse(status=upstream_resp.status)
            for k, v in upstream_resp.headers.items():
                if k.lower() not in ("content-length", "content-encoding", "transfer-encoding"):
                    resp.headers[k] = v
            await resp.prepare(request)
            async for chunk in upstream_resp.content.iter_any():
                self.stats.bytes_out += len(chunk)
                await resp.write(chunk)
            await resp.write_eof()
            return resp

        # exhausted retries without any upstream response at all
        self.stats.failures += 1
        self._consecutive_dead += 1
        self.stats.last_error = f"upstream unreachable: {last_exc}"
        logger.warning("gateway[%s] upstream dead after retries: %s", self.worker_id, last_exc)
        if self._consecutive_dead >= 2 and self._on_upstream_dead and not self._on_upstream_dead.done():
            self._on_upstream_dead.set_result(self.worker_id)
        return web.json_response(
            {"error": {"type": "api_error", "message": f"devflock gateway: upstream unreachable ({last_exc})"}},
            status=502)


async def run_standalone(worker_id: str, upstream: Upstream, port: int):
    gw = Gateway(worker_id, upstream, port)
    await gw.start()
    try:
        await asyncio.Event().wait()
    finally:
        await gw.stop()
