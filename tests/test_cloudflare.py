"""Cloudflare Workers AI backend: translation units, live gateway, and a REAL Claude Code
session driven through the translator against a mock Workers AI server.
Run: python tests/test_cloudflare.py
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

PORT = 18790
os.environ["CLOUDFLARE_API_BASE"] = f"http://127.0.0.1:{PORT}/client/v4"   # before importing devflock
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import aiohttp
from aiohttp import web

from devflock import cloudflare as cf
from devflock.config import Endpoint, PROVIDER_CLOUDFLARE, CF_CLIENT_MODEL_ALIAS
from devflock.gateway import Gateway, Upstream
from devflock.probe import probe
from devflock.worker import Worker
from mock_cloudflare_server import make_app, ACCT, TOKEN

ok = True
KIMI = "@cf/moonshotai/kimi-k2.7-code"


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra and not cond else ""))


def parse_sse(raw: bytes):
    events = []
    for block in raw.decode().split("\n\n"):
        if not block.strip():
            continue
        ev = dict(line.split(": ", 1) for line in block.split("\n") if ": " in line)
        events.append((ev.get("event"), json.loads(ev["data"])))
    return events


def rebuild(events):
    """Fold Anthropic SSE events back into content blocks, like an SDK would."""
    blocks, stop = {}, None
    for name, d in events:
        if name == "content_block_start":
            blocks[d["index"]] = dict(d["content_block"], _json="")
        elif name == "content_block_delta":
            b = blocks[d["index"]]
            if d["delta"]["type"] == "text_delta":
                b["text"] += d["delta"]["text"]
            elif d["delta"]["type"] == "input_json_delta":
                b["_json"] += d["delta"]["partial_json"]
        elif name == "message_delta":
            stop = d["delta"]["stop_reason"]
    out = []
    for i in sorted(blocks):
        b = blocks[i]
        if b["type"] == "tool_use":
            b["input"] = json.loads(b.pop("_json") or "{}")
        else:
            b.pop("_json")
        out.append(b)
    return out, stop


# ---------------------------------------------------------------- 1. pure translation
def unit_tests():
    req = {
        "system": [{"type": "text", "text": "You are X.", "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": "Be brief."}],
        "max_tokens": 100, "stream": True, "stop_sequences": ["END"],
        "tools": [{"name": "Write", "description": "w", "input_schema": {"type": "object", "properties": {"p": {"type": "string"}}}}],
        "tool_choice": {"type": "any"},
        "messages": [
            {"role": "user", "content": "make a file"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hm", "signature": "s"},
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "toolu_1", "name": "Write", "input": {"p": "a"}}]},
            {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>x</system-reminder>"},
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "written"}]},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "boom", "is_error": True}]},
        ]}
    o = cf.to_openai_request(req, KIMI)
    m = o["messages"]
    check("system blocks joined, cache_control dropped", m[0] == {"role": "system", "content": "You are X.\nBe brief."})
    check("model overridden with the @cf id", o["model"] == KIMI)
    check("assistant: thinking dropped, text + tool_calls kept",
          m[2]["content"] == "ok" and m[2]["tool_calls"][0]["function"]["arguments"] == '{"p": "a"}'
          and m[2]["tool_calls"][0]["id"] == "toolu_1")
    check("tool results become role=tool messages BEFORE the user text",
          [x["role"] for x in m[3:]] == ["tool", "tool", "user"] and m[3]["content"] == "written"
          and m[4]["content"] == "ERROR: boom")
    check("tools -> function schema, tool_choice any -> required, stop mapped",
          o["tools"][0]["function"]["parameters"]["properties"]["p"] and o["tool_choice"] == "required"
          and o["stop"] == ["END"] and o["stream"] is True)
    o2 = cf.to_openai_request({"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
        {"type": "text", "text": "what is this"}]}], "tools": [{"name": "t", "input_schema": {}}],
        "tool_choice": {"type": "tool", "name": "t"}}, KIMI)
    check("image -> image_url data URI; forced tool choice mapped",
          o2["messages"][0]["content"][0]["image_url"]["url"] == "data:image/png;base64,AAA"
          and o2["tool_choice"] == {"type": "function", "function": {"name": "t"}})

    r = cf.to_anthropic_response({"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c1", "function": {"name": "Write", "arguments": {"p": "z"}}}]}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3}}, KIMI)
    check("response: dict-typed args accepted, stop_reason forced to tool_use even if finish=stop",
          r["content"][0]["input"] == {"p": "z"} and r["stop_reason"] == "tool_use"
          and r["usage"] == {"input_tokens": 5, "output_tokens": 3})
    r = cf.to_anthropic_response({"result": {"choices": [{"message": {"content": "hi"}, "finish_reason": "length"}]}}, KIMI)
    check("response: native {result:{...}} envelope unwrapped; length -> max_tokens",
          r["content"][0]["text"] == "hi" and r["stop_reason"] == "max_tokens")

    # streaming edge cases
    def run(chunks):
        t = cf.StreamTranslator(KIMI)
        ev = []
        for c in chunks:
            ev += t.feed(c if isinstance(c, str) else json.dumps(c))
        ev += t.finish()
        return rebuild(parse_sse(b"".join(ev)))
    blocks, stop = run([
        {"choices": [{"delta": {"content": "Let me "}}]}, {"choices": [{"delta": {"content": "write."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "Write", "arguments": '{"p":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"1"}'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "Read", "arguments": '{"q":2}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}, "[DONE]"])
    check("stream: text then two parallel tool calls assemble correctly",
          [b["type"] for b in blocks] == ["text", "tool_use", "tool_use"] and blocks[0]["text"] == "Let me write."
          and blocks[1]["input"] == {"p": "1"} and blocks[2]["input"] == {"q": 2} and stop == "tool_use")
    blocks, stop = run([{"choices": [{"delta": {"tool_calls": [
        {"id": "x", "function": {"name": "Write", "arguments": '{"p":"1"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]}])
    check("stream: tool call without `index` still works", blocks[0]["name"] == "Write" and stop == "tool_use")
    blocks, stop = run([])
    check("stream: empty model output still yields a well-formed message",
          blocks == [{"type": "text", "text": ""}] and stop == "end_turn")


# ---------------------------------------------------------------- 2. live gateway
async def gateway_tests(app):
    up = Upstream.from_endpoint(Endpoint("w", cf.ai_base_url(ACCT), TOKEN, "bearer", KIMI, PROVIDER_CLOUDFLARE))
    check("Upstream.from_endpoint picks the cloudflare backend", up.backend == "cloudflare" and up.model == KIMI)
    gw = Gateway("cf1", up)
    port = await gw.start()
    u = f"http://127.0.0.1:{port}"
    msg = {"model": CF_CLIENT_MODEL_ALIAS, "max_tokens": 64, "messages": [{"role": "user", "content": "hello cf"}]}
    async with aiohttp.ClientSession() as s:
        async with s.get(u + "/health") as r:
            check("gateway /health (cloudflare)", r.status == 200)
        async with s.post(u + "/v1/messages/count_tokens", json=msg) as r:
            check("count_tokens answered locally", (await r.json())["input_tokens"] >= 1)

        async with s.post(u + "/v1/messages", json=msg, headers={"x-api-key": "devflock-local"}) as r:
            d = await r.json()
            check("non-stream: Anthropic-shaped reply", r.status == 200 and d["type"] == "message"
                  and d["content"][0]["text"] == "Echo: hello cf" and d["usage"]["input_tokens"] == 11, str(d))
        sent = app["requests"][-1]
        check("upstream got the REAL @cf model id, not Claude Code's alias", sent["model"] == KIMI)
        check("upstream got Authorization: Bearer <cf token> (client key not forwarded)",
              app["headers"][-1].get("Authorization") == f"Bearer {TOKEN}")

        async with s.post(u + "/v1/messages", json={**msg, "stream": True}) as r:
            raw = await r.read()
        blocks, stop = rebuild(parse_sse(raw))
        names = [e for e, _ in parse_sse(raw)]
        check("stream text: full event envelope", names[0] == "message_start" and names[-1] == "message_stop"
              and "content_block_start" in names and blocks[0]["text"] == "Echo: hello cf" and stop == "end_turn", str(names))

        tool_msg = {**msg, "stream": True, "tools": [{"name": "get_time", "description": "t",
                                                     "input_schema": {"type": "object", "properties": {}}}],
                    "messages": [{"role": "user", "content": "please call get_time"}]}
        async with s.post(u + "/v1/messages", json=tool_msg) as r:
            blocks, stop = rebuild(parse_sse(await r.read()))
        check("stream tool call: tool_use block + stop_reason tool_use",
              blocks and blocks[-1]["type"] == "tool_use" and blocks[-1]["name"] == "get_time" and stop == "tool_use", str(blocks))

        async with s.post(u + "/v1/foo", json={}) as r:
            check("unknown route -> Anthropic-style 404", r.status == 404 and (await r.json())["type"] == "error")
    await gw.stop()

    # rate limit then success: gateway retries before any bytes reach the client
    app2 = make_app(fail_first=2, fail_status=429)
    run2 = web.AppRunner(app2); await run2.setup(); await web.TCPSite(run2, "127.0.0.1", PORT + 1).start()
    import devflock.gateway as gwm
    saved = gwm.RETRY_BACKOFFS; gwm.RETRY_BACKOFFS = [0.05, 0.05, 0.05]
    up2 = Upstream(f"http://127.0.0.1:{PORT + 1}/client/v4/accounts/{ACCT}/ai/v1", TOKEN, "bearer", "cloudflare", KIMI)
    gw2 = Gateway("cf2", up2); p2 = await gw2.start()
    async with aiohttp.ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{p2}/v1/messages", json=msg) as r:
            check("429 x2 then OK: retried transparently", r.status == 200 and gw2.stats.retries == 2,
                  f"status={r.status} retries={gw2.stats.retries}")
    await gw2.stop(); gwm.RETRY_BACKOFFS = saved; await run2.cleanup()

    # bad token -> 401 surfaced in Anthropic error format with a useful hint
    bad = Gateway("cf3", Upstream(cf.ai_base_url(ACCT), "wrong-token", "bearer", "cloudflare", KIMI))
    pb = await bad.start()
    async with aiohttp.ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{pb}/v1/messages", json=msg) as r:
            d = await r.json()
            check("bad token: 401 + Anthropic error body mentioning Workers AI permission",
                  r.status == 401 and d["error"]["type"] == "authentication_error" and "Workers AI" in d["error"]["message"], str(d))
            check("token never appears in the error text", "wrong-token" not in json.dumps(d))
    await bad.stop()

    # reasoning-style models: reasoning_content must not leak into visible text
    app3 = make_app(reasoning=True)
    run3 = web.AppRunner(app3); await run3.setup(); await web.TCPSite(run3, "127.0.0.1", PORT + 2).start()
    gw4 = Gateway("cf4", Upstream(f"http://127.0.0.1:{PORT + 2}/client/v4/accounts/{ACCT}/ai/v1", TOKEN, "bearer", "cloudflare", KIMI))
    p4 = await gw4.start()
    async with aiohttp.ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{p4}/v1/messages", json={**msg, "stream": True}) as r:
            blocks, _ = rebuild(parse_sse(await r.read()))
        check("reasoning_content is not shown as answer text", blocks[0]["text"] == "Echo: hello cf")
    await gw4.stop(); await run3.cleanup()


# ---------------------------------------------------------------- 3. models + probe
async def listing_and_probe():
    creds = cf.CfCreds(ACCT, TOKEN)
    models = await cf.list_text_models(creds)
    check("model list: all pages fetched, sorted", [m.id for m in models] == sorted(m["name"] for m in __import__("mock_cloudflare_server").MODELS))
    k = next(m for m in models if "kimi" in m.id)
    check("model list: context window + function-calling parsed", k.context_window == 262144 and k.function_calling is True)
    check("model list: absent function_calling -> unknown (None)",
          next(m for m in models if "llama" in m.id).function_calling is None)
    try:
        await cf.list_text_models(cf.CfCreds(ACCT, "nope"))
        check("bad token on listing raises", False)
    except RuntimeError as e:
        check("bad token on listing raises a readable error", "401" in str(e) and "nope" not in str(e), str(e))

    ep = Endpoint("w1", cf.ai_base_url(ACCT), TOKEN, "bearer", KIMI, PROVIDER_CLOUDFLARE)
    r = await probe(ep)
    check("probe: cloudflare model ok, latency measured", r.ok and r.auth_style == "bearer", r.error or "")
    check("probe: tool call smoke test passes", r.tools_ok is True)
    r = await probe(Endpoint("w1", cf.ai_base_url(ACCT), "bad", "bearer", KIMI, PROVIDER_CLOUDFLARE))
    check("probe: bad token reported with permission hint", (not r.ok) and "Workers AI" in (r.error or ""), r.error or "")
    check("probe error never contains the token", "bad" not in (r.error or "").replace("bad token", ""))


# ---------------------------------------------------------------- 4. REAL Claude Code through the translator
async def real_claude_code(app):
    ep = Endpoint("worker-1", cf.ai_base_url(ACCT), TOKEN, "bearer", KIMI, PROVIDER_CLOUDFLARE)
    gw = Gateway("cf-real", Upstream.from_endpoint(ep))
    port = await gw.start()
    region = Path(tempfile.mkdtemp(prefix="devflock-cf-"))
    w = Worker("worker-1", port, ep.client_key, region, model_name=ep.client_model)
    check("worker env gets a placeholder key, not the Cloudflare token",
          w._base_env()["ANTHROPIC_API_KEY"] == "devflock-local" and TOKEN not in json.dumps(w._base_env()))
    target = region / "cf_hello.txt"
    res = await w.run_task(f"Use the Write tool to create a file at the absolute path {target} containing a greeting.",
                           max_turns=4, allowed_tools=["Write"], resume=False)
    check("Claude Code completed the task via Cloudflare mock", res.ok, f"error={res.error} summary={res.summary!r}")
    check("tool call made and file written", "Write" in res.tool_calls and target.exists()
          and "cloudflare mock" in target.read_text(), f"calls={res.tool_calls}")
    check("final answer came back after the tool result", "Done." in res.summary, res.summary)
    reqs = app["requests"]
    check("tool results were sent upstream as role=tool messages",
          any(m["role"] == "tool" for r in reqs for m in r["messages"]))
    check("every upstream request used the @cf model id", all(r["model"] == KIMI for r in reqs))
    await gw.stop()


async def main():
    unit_tests()
    app = make_app()
    runner = web.AppRunner(app); await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", PORT).start()
    await gateway_tests(app)
    await listing_and_probe()
    app["requests"].clear()
    await real_claude_code(app)
    await runner.cleanup()
    print("\nALL CLOUDFLARE TESTS " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


asyncio.run(main())
