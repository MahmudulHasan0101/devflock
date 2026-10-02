"""A tiny stand-in for `llama-server --jinja` (Anthropic Messages API),
used to test DevFlock's gateway and worker runner without a real GPU box.

Not part of the shipped package. Run standalone:
    python tests/mock_llama_server.py --port 8090 --key sk-test [--flaky]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
import uuid

from aiohttp import web

REQUEST_LOG: list[dict] = []


def make_app(api_key: str, flaky: bool, tool_capable: bool) -> web.Application:
    app = web.Application()
    state = {"calls": 0}

    async def health(request):
        return web.Response(text="ok")

    async def props(request):
        return web.json_response({
            "default_generation_settings": {"n_ctx": 262144},
            "total_slots": 1,
        })

    async def count_tokens(request):
        body = await request.json()
        text = json.dumps(body.get("messages", []))
        return web.json_response({"input_tokens": max(1, len(text) // 4)})

    def check_auth(request):
        key = request.headers.get("x-api-key") or (
            request.headers.get("authorization", "").removeprefix("Bearer ").strip() or None
        )
        return key == api_key

    async def messages(request):
        state["calls"] += 1
        REQUEST_LOG.append({"t": time.time(), "n": state["calls"]})

        if flaky and state["calls"] % 3 == 0:
            # simulate a submachine hiccup: 502 with no body
            return web.Response(status=502, text="bad gateway (simulated)")

        if not check_auth(request):
            return web.json_response({"error": {"type": "authentication_error"}}, status=401)

        body = await request.json()
        stream = bool(body.get("stream"))
        tools = body.get("tools") or []
        msgs = body.get("messages", [])

        # The original instruction is the first plain-text user message (not a
        # tool_result). Later requests in the same turn re-send the whole
        # history, so re-deriving from the ORIGINAL keeps a multi-tool-call
        # plan (e.g. "use the ask_region tool, then use the Write tool")
        # working across turns, unlike keying off only the most recent message.
        def extract_text(content):
            """Plain text only -- avoids JSON-escaping artifacts (and the noise
            of a prepended system-reminder block) confusing the regexes below."""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(b.get("text", "") for b in content
                                 if isinstance(b, dict) and b.get("type") == "text")
            return ""

        original = ""
        for m in msgs:
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    original = c
                    break
                if isinstance(c, list) and not any(b.get("type") == "tool_result" for b in c
                                                    if isinstance(b, dict)):
                    original = extract_text(c)
                    break

        # Claude Code may prepend a <system-reminder> block ahead of our actual
        # prompt in that same first user message. Keep the longest segment --
        # our real task instruction is reliably much longer than a reminder
        # note -- so every regex below (region, filename, ...) sees only OUR
        # text, not reminder boilerplate that can itself contain stray
        # "word.ext"-shaped or URL-shaped tokens.
        if "<system-reminder>" in original:
            segments = [seg.strip() for seg in re.split(r"</?system-reminder>", original) if seg.strip()]
            if segments:
                original = max(segments, key=len)

        tool_results_seen = sum(
            1 for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result")
        last_tool_result_text = ""
        for m in reversed(msgs):
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        c = b.get("content")
                        if isinstance(c, list):
                            c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                        last_tool_result_text = c if isinstance(c, str) else json.dumps(c)
                break

        # Our own seed prompt (worker.build_region_seed_prompt) wraps the real
        # instruction under "## Your task". Everything else -- region memory
        # templates, neighbor docs, and whatever context Claude Code itself
        # injects (e.g. a cwd path that, given our own worktree layout,
        # literally contains ".devflock" as a path SEGMENT, not a file
        # extension) -- is noise these regexes should never see. Scope to
        # just the task text when that heading is present; fall back to the
        # whole (already reminder-stripped) text for prompts that don't use
        # the seed-prompt format (e.g. direct calls in test_worker_run.py).
        task_m = re.search(r"## Your task\n(.*?)(?:\nWhen finished|\Z)", original, re.DOTALL)
        scan_text = task_m.group(1).strip() if task_m else original

        named_in_order = [m.group(1).lower() for m in re.finditer(r"the (\w+) tool", scan_text, re.IGNORECASE)]
        # dedupe keeping order, in case the instruction repeats a tool name
        seen_names, plan = set(), []
        for n in named_in_order:
            if n not in seen_names:
                seen_names.add(n)
                plan.append(n)

        if os.environ.get("MOCK_DEBUG"):
            print(f"DEBUG scan_text[:300]={scan_text[:300]!r}  plan={plan}  "
                 f"tool_results_seen={tool_results_seen}", flush=True)
        want_tool = tool_capable and tools and plan and tool_results_seen < len(plan)
        msg_id = "msg_" + uuid.uuid4().hex[:24]

        def dummy_for(prop_schema, prop_name):
            t = prop_schema.get("type")
            if prop_name.lower() in ("content", "text", "body"):
                # if we just got a tool result back, "read" it into this write --
                # this is what actually proves ask_region's answer reaches the file
                return last_tool_result_text if tool_results_seen else "print('hi')\n"
            if prop_name.lower() in ("path", "file_path", "filepath"):
                # prefer an absolute path mentioned verbatim (this is exactly what
                # tests/test_worker_run.py's path-guard-escape case relies on --
                # a real small model asked to write "at the absolute path X" would
                # pass X through, not invent a relative name instead).
                # Lookbehind excludes URL fragments like "https://claude.com" (a
                # system-reminder injected ahead of our prompt can contain these):
                # both slashes of "//" would otherwise look like a rooted path.
                # Only treat something as an absolute path if the instruction
                # SAYS "absolute path" (that's literally how the one test that
                # needs this -- test_worker_run.py's path-guard-escape case --
                # phrases it). Any other absolute path mentioned for CONTEXT
                # (e.g. runner.py's "Module `x` (folder: /abs/path)" prefix,
                # which itself contains ".devflock" as a path segment, not a
                # file extension) must never be picked up here.
                abs_m = re.search(r'absolute path[:\s]+([^\s\'"]+)', scan_text, re.IGNORECASE)
                if abs_m:
                    return abs_m.group(1).rstrip(").,;:")
                # Bare filename: exclude matches that are part of a larger
                # path (preceded by "/"), which is exactly what would
                # otherwise make ".devflock" inside a folder prefix look
                # like a file.
                name_m = re.search(r'(?<!/)\b([\w-]+\.\w{1,5})\b', scan_text)
                return name_m.group(1) if name_m else "hello.py"
            if prop_name.lower() == "region":
                # require a straight quote (our seed prompt's OWN "owning the
                # region `x`" uses backticks, so this won't self-match it)
                rm = re.search(r"region\s*['\"]([\w-]+)['\"]", scan_text, re.IGNORECASE)
                return rm.group(1) if rm else "unknown"
            if prop_name.lower() in ("question", "description"):
                return scan_text[:300]
            if t == "string":
                return "value"
            if t in ("number", "integer"):
                return 1
            if t == "boolean":
                return True
            if t == "array":
                return []
            return {}

        if want_tool:
            wanted = plan[tool_results_seen]

            def matches(tname):
                tname = tname.lower()
                return tname == wanted or tname.endswith("__" + wanted)

            named = [t for t in tools if matches(t.get("name", ""))]
            if not named:
                # The specifically-named tool isn't currently offered -- most
                # likely an MCP handshake that hasn't completed yet on a brand
                # new session. NEVER guess some other tool instead (that
                # previously produced a bogus Bash("value") call cascading
                # into garbage file contents); just end the turn with no tool
                # call. The scheduler's own repair loop then retries on a
                # fresh session, where the handshake completes before turn 1.
                blocks = [{"type": "text",
                          "text": f"(tool {wanted!r} is not available yet; will retry)"}]
                stop_reason = "end_turn"
            else:
                chosen = named[0]
                schema = chosen.get("input_schema", {}) or {}
                props = schema.get("properties", {}) or {}
                required = schema.get("required", list(props.keys()))
                tool_input = {k: dummy_for(props.get(k, {}), k) for k in required}
                if os.environ.get("MOCK_DEBUG"):
                    print(f"DEBUG tool_input for {chosen['name']}: {tool_input}", flush=True)
                blocks = [{"type": "tool_use", "id": "tu_" + uuid.uuid4().hex[:16],
                          "name": chosen["name"], "input": tool_input}]
                stop_reason = "tool_use"
        else:
            reply = f"Echo ({len(original)} chars received): " + original[:120]
            blocks = [{"type": "text", "text": reply}]
            stop_reason = "end_turn"

        usage = {"input_tokens": max(1, len(original) // 4), "output_tokens": 24}

        if not stream:
            return web.json_response({
                "id": msg_id, "type": "message", "role": "assistant",
                "model": body.get("model", "qwen"), "content": blocks,
                "stop_reason": stop_reason, "usage": usage,
            })

        resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await resp.prepare(request)

        async def send(ev):
            await resp.write(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode())

        await send({"type": "message_start", "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "model": body.get("model", "qwen"), "content": [],
            "usage": {"input_tokens": usage["input_tokens"], "output_tokens": 0},
        }})

        for idx, block in enumerate(blocks):
            if block["type"] == "tool_use":
                await send({"type": "content_block_start", "index": idx, "content_block": {
                    "type": "tool_use", "id": block["id"], "name": block["name"], "input": {},
                }})
                partial = json.dumps(block["input"])
                for i in range(0, len(partial), 8):
                    await send({"type": "content_block_delta", "index": idx, "delta": {
                        "type": "input_json_delta", "partial_json": partial[i:i + 8]}})
                    await asyncio.sleep(0.01)
                await send({"type": "content_block_stop", "index": idx})
            else:
                await send({"type": "content_block_start", "index": idx, "content_block": {
                    "type": "text", "text": ""}})
                text = block["text"]
                for i in range(0, len(text), 12):
                    await send({"type": "content_block_delta", "index": idx, "delta": {
                        "type": "text_delta", "text": text[i:i + 12]}})
                    await asyncio.sleep(0.01)
                await send({"type": "content_block_stop", "index": idx})

        await send({"type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": usage["output_tokens"]}})
        await send({"type": "message_stop"})
        await resp.write_eof()
        return resp

    app.router.add_get("/health", health)
    app.router.add_get("/props", props)
    app.router.add_post("/v1/messages/count_tokens", count_tokens)
    app.router.add_post("/v1/messages", messages)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--key", default="sk-test")
    ap.add_argument("--flaky", action="store_true", help="fail every 3rd request with 502")
    ap.add_argument("--no-tools", action="store_true", help="never emit tool_use blocks")
    a = ap.parse_args()
    app = make_app(a.key, a.flaky, tool_capable=not a.no_tools)
    print(f"mock llama-server on http://127.0.0.1:{a.port}  key={a.key}  flaky={a.flaky}")
    web.run_app(app, host="127.0.0.1", port=a.port, print=None)


if __name__ == "__main__":
    main()
