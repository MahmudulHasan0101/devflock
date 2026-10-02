"""Quick health check of one endpoint before we trust it with work."""
from __future__ import annotations

import dataclasses
import time
from typing import Optional

import aiohttp

from .config import Endpoint


@dataclasses.dataclass
class ProbeResult:
    ok: bool
    auth_style: str = "x-api-key"
    latency_s: float = 0.0
    n_ctx: Optional[int] = None
    slots: Optional[int] = None
    tools_ok: Optional[bool] = None
    warnings: list[str] = dataclasses.field(default_factory=list)
    error: Optional[str] = None


def _headers(ep_key: str, style: str) -> dict:
    h = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if style == "bearer":
        h["authorization"] = f"Bearer {ep_key}"
    else:
        h["x-api-key"] = ep_key
    return h


async def probe(ep: Endpoint, timeout_s: float = 120.0) -> ProbeResult:
    base = ep.base_url.rstrip("/")
    t = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=t) as s:
        try:
            async with s.get(base + "/health") as r:
                if r.status != 200:
                    return ProbeResult(False, error=f"/health returned HTTP {r.status} (model still loading, or tunnel down?)")
        except Exception as e:
            return ProbeResult(False, error=f"cannot reach {base}: {type(e).__name__}: {e}")

        body = {"model": ep.model_name, "max_tokens": 16,
                "messages": [{"role": "user", "content": "Reply with the single word OK."}]}
        style, status, t0 = ep.auth_style, 0, time.time()
        for candidate in dict.fromkeys([ep.auth_style, "x-api-key", "bearer"]):
            t0 = time.time()
            try:
                async with s.post(base + "/v1/messages", json=body, headers=_headers(ep.api_key, candidate)) as r:
                    status = r.status
                    if r.status == 200:
                        style = candidate
                        await r.read()
                        break
            except Exception as e:
                return ProbeResult(False, error=f"/v1/messages failed: {type(e).__name__}: {e}")
        if status != 200:
            return ProbeResult(False, error=f"/v1/messages returned HTTP {status} (wrong API key, or the server lacks --jinja / Anthropic API support)")
        res = ProbeResult(True, auth_style=style, latency_s=time.time() - t0)

        try:
            async with s.get(base + "/props", headers=_headers(ep.api_key, style)) as r:
                if r.status == 200:
                    p = await r.json()
                    res.n_ctx = (p.get("default_generation_settings") or {}).get("n_ctx")
                    res.slots = p.get("total_slots")
                    if res.slots and res.slots > 1:
                        res.warnings.append(f"server runs {res.slots} slots, so each request may get only "
                                            f"1/{res.slots} of the context; consider --parallel 1")
        except Exception:
            pass

        tool_body = {"model": ep.model_name, "max_tokens": 200,
                     "tools": [{"name": "get_time", "description": "Returns the current time.",
                                "input_schema": {"type": "object", "properties": {}}}],
                     "messages": [{"role": "user", "content": "Use the get_time tool now."}]}
        try:
            async with s.post(base + "/v1/messages", json=tool_body, headers=_headers(ep.api_key, style)) as r:
                data = await r.json()
                res.tools_ok = any(b.get("type") == "tool_use" for b in data.get("content", []))
                if not res.tools_ok:
                    res.warnings.append("model did not make a tool call in the smoke test; "
                                        "coding sessions may be unreliable")
        except Exception:
            res.tools_ok = None
        if res.latency_s > 30:
            res.warnings.append(f"a trivial request took {res.latency_s:.0f}s; the tunnel or GPU is slow")
        return res
