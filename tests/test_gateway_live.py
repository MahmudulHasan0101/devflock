"""Live (not mocked) test: real aiohttp mock upstream + real Gateway + real
HTTP client, all in-process. Run: python tests/test_gateway_live.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import aiohttp
from aiohttp import web

from devflock.gateway import Gateway, Upstream
from mock_llama_server import make_app


async def run_mock(port, key, flaky=False):
    app = make_app(key, flaky=flaky, tool_capable=True)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


async def main():
    ok = True

    # --- 1. normal round trip through the gateway ---
    mock = await run_mock(18090, "sk-test")
    gw = Gateway("w1", Upstream("http://127.0.0.1:18090", "sk-test"), port=0)
    gw_port = await gw.start()

    async with aiohttp.ClientSession() as s:
        async with s.get(f"http://127.0.0.1:{gw_port}/health") as r:
            assert r.status == 200, r.status
            print("[PASS] health passthrough")

        payload = {"model": "qwen", "max_tokens": 32,
                   "messages": [{"role": "user", "content": "hello via gateway"}]}
        async with s.post(f"http://127.0.0.1:{gw_port}/v1/messages",
                           json=payload, headers={"x-api-key": "whatever-worker-thinks"}) as r:
            data = await r.json()
            text = data["content"][0]["text"]
            assert "hello via gateway" in text, data
            print(f"[PASS] non-streaming round trip: {text!r}")

        # streaming
        events = 0
        async with s.post(f"http://127.0.0.1:{gw_port}/v1/messages",
                           json={**payload, "stream": True}) as r:
            async for raw in r.content:
                line = raw.decode().strip()
                if line.startswith("data:"):
                    events += 1
        assert events > 3, events
        print(f"[PASS] streaming passthrough: {events} SSE events")

    await gw.stop()
    await mock.cleanup()

    # --- 2. retry path: upstream fails every 3rd request ---
    mock2 = await run_mock(18091, "sk-test", flaky=True)
    gw2 = Gateway("w2", Upstream("http://127.0.0.1:18091", "sk-test"), port=0)
    # shrink backoffs for the test so it doesn't take 14s
    import devflock.gateway as gwmod
    gwmod.RETRY_BACKOFFS = [0.05, 0.1, 0.1, 0.1]
    gw2_port = await gw2.start()

    async with aiohttp.ClientSession() as s:
        successes = 0
        for i in range(6):
            payload = {"model": "qwen", "max_tokens": 8,
                       "messages": [{"role": "user", "content": f"req {i}"}]}
            async with s.post(f"http://127.0.0.1:{gw2_port}/v1/messages", json=payload) as r:
                if r.status == 200:
                    successes += 1
        print(f"[{'PASS' if successes == 6 else 'FAIL'}] retry absorbed flaky 502s: "
              f"{successes}/6 succeeded, gateway retried {gw2.stats.retries} times")
        ok &= successes == 6

    await gw2.stop()
    await mock2.cleanup()

    # --- 3. hot-swap upstream without restarting the gateway ---
    mockA = await run_mock(18092, "sk-A")
    mockB = await run_mock(18093, "sk-B")
    gw3 = Gateway("w3", Upstream("http://127.0.0.1:18092", "sk-A"), port=0)
    gw3_port = await gw3.start()

    async with aiohttp.ClientSession() as s:
        payload = {"model": "qwen", "max_tokens": 8, "messages": [{"role": "user", "content": "pre-swap"}]}
        async with s.post(f"http://127.0.0.1:{gw3_port}/v1/messages", json=payload) as r:
            assert r.status == 200

        await s.post(f"http://127.0.0.1:{gw3_port}/_devflock/upstream",
                      json={"base_url": "http://127.0.0.1:18093", "api_key": "sk-B"})

        async with s.post(f"http://127.0.0.1:{gw3_port}/v1/messages",
                           json={**payload, "messages": [{"role": "user", "content": "post-swap"}]}) as r:
            data = await r.json()
            swapped = "post-swap" in data["content"][0]["text"]
            print(f"[{'PASS' if swapped else 'FAIL'}] hot-swap upstream mid-run: {data['content'][0]['text']!r}")
            ok &= swapped

    await gw3.stop()
    await mockA.cleanup()
    await mockB.cleanup()

    print("\nALL GATEWAY TESTS " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
