"""A stand-in for Cloudflare Workers AI's OpenAI-compatible endpoint.

Serves, under /client/v4/accounts/<ACCT>/ai/ :
    POST v1/chat/completions     (Bearer auth; streaming and non-streaming; tool calls)
    GET  models/search           (paged model catalogue with context_window/function_calling)

Scripted behaviour (so a REAL Claude Code binary can be driven through the gateway):
  * if the original user prompt says "...absolute path <P> ..." and a `Write` tool is offered,
    the first turn is a Write tool call to <P>; once a tool result is in the history, it answers "Done."
  * or, for DevFlock's real worker prompt, a Write of hello.py into the first writable folder;
  * otherwise it echoes the last user text.
Tool-call arguments are streamed in several fragments to exercise partial-JSON assembly.

Not part of the shipped package.
"""
from __future__ import annotations

import json
import re

from aiohttp import web

ACCT = "acct123"
TOKEN = "cf-test-token"

MODELS = [
    {"name": "@cf/moonshotai/kimi-k2.7-code", "description": "Kimi coding model",
     "properties": [{"property_id": "context_window", "value": "262144"},
                    {"property_id": "function_calling", "value": "true"}]},
    {"name": "@cf/openai/gpt-oss-120b", "description": "gpt-oss 120b",
     "properties": [{"property_id": "context_window", "value": "128000"},
                    {"property_id": "function_calling", "value": "true"}]},
    {"name": "@cf/meta/llama-3.1-8b-instruct", "description": "Llama 3.1 8B",
     "properties": [{"property_id": "context_window", "value": "7968"}]},
    {"name": "@cf/qwen/qwen3.8-27b", "description": "Qwen 27B",
     "properties": [{"property_id": "context_window", "value": "262144"},
                    {"property_id": "function_calling", "value": "true"}]},
]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def make_app(token: str = TOKEN, acct: str = ACCT, fail_first: int = 0, fail_status: int = 429,
             reasoning: bool = False) -> web.Application:
    app = web.Application()
    app["requests"] = []          # every chat/completions body received, for assertions
    app["headers"] = []
    st = {"fails_left": fail_first}

    def auth_ok(request):
        return request.headers.get("Authorization") == f"Bearer {token}"

    def denied():
        return web.json_response({"success": False, "result": None, "messages": [],
                                  "errors": [{"code": 10000, "message": "Authentication error"}]}, status=401)

    async def models_search(request):
        if not auth_ok(request):
            return denied()
        page, per = int(request.query.get("page", 1)), int(request.query.get("per_page", 100))
        per = min(per, 2)  # force pagination so the client's paging loop is exercised
        chunk = MODELS[(page - 1) * per: page * per]
        return web.json_response({"success": True, "errors": [], "result": chunk,
                                  "result_info": {"page": page, "per_page": per, "count": len(chunk),
                                                  "total_count": len(MODELS),
                                                  "total_pages": -(-len(MODELS) // per)}})

    async def chat(request):
        if not auth_ok(request):
            return denied()
        if st["fails_left"] > 0:
            st["fails_left"] -= 1
            return web.json_response({"success": False, "errors": [{"code": 3040, "message": "capacity"}]},
                                     status=fail_status)
        body = await request.json()
        app["requests"].append(body)
        app["headers"].append(dict(request.headers))
        msgs = body.get("messages", [])
        tools = {t["function"]["name"] for t in body.get("tools") or []}

        first_user = next((_text(m["content"]) for m in msgs if m["role"] == "user"), "")
        last = msgs[-1] if msgs else {}
        m = re.search(r"absolute path (\S+)", first_user)
        tool_call = None
        if last.get("role") == "tool":
            reply = "Done."
        elif "Write" in tools and m:
            path = m.group(1).rstrip(".,)`'\"")
            tool_call = ("Write", {"file_path": path, "content": "hi from cloudflare mock\n"})
            reply = None
        elif "Write" in tools and re.search(r"inside these folders[^\n]*\n\s*- (\S+)", first_user):
            # DevFlock's real worker prompt: write hello.py in the region's first writable folder
            folder = re.search(r"inside these folders[^\n]*\n\s*- (\S+)", first_user).group(1)
            tool_call = ("Write", {"file_path": f"{folder}/hello.py", "content": "print('hi from cloudflare')\n"})
            reply = None
        elif "get_time" in tools and "get_time" in first_user:
            tool_call = ("get_time", {})
            reply = None
        else:
            reply = "Echo: " + (_text(last.get("content")) if last.get("role") == "user" else first_user)[:200]

        if not body.get("stream"):
            msg = {"role": "assistant", "content": reply}
            if tool_call:
                msg["content"] = None
                msg["tool_calls"] = [{"id": "call_1", "type": "function",
                                      "function": {"name": tool_call[0], "arguments": json.dumps(tool_call[1])}}]
            if reasoning:
                msg["reasoning_content"] = "thinking..."
            return web.json_response({
                "id": "chatcmpl-1", "object": "chat.completion", "model": body["model"],
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": "tool_calls" if tool_call else "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        async def send(obj):
            await resp.write(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")

        def chunk(delta, finish=None):
            return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": body["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        await send(chunk({"role": "assistant", "content": ""}))
        if reasoning:
            await send(chunk({"reasoning_content": "hmm"}))
        if tool_call:
            args = json.dumps(tool_call[1])
            third = max(1, len(args) // 3)
            await send(chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                              "function": {"name": tool_call[0], "arguments": ""}}]}))
            for i in range(0, len(args), third):
                await send(chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[i:i + third]}}]}))
            await send(chunk({}, "tool_calls"))
        else:
            for i in range(0, len(reply), 6):
                await send(chunk({"content": reply[i:i + 6]}))
            await send(chunk({}, "stop"))
        await send({"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7}})
        await send(b"[DONE]")
        await resp.write_eof()
        return resp

    base = f"/client/v4/accounts/{acct}/ai"
    app.router.add_get(base + "/models/search", models_search)
    app.router.add_post(base + "/v1/chat/completions", chat)
    return app
