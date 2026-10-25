"""Cloudflare Workers AI support.

Claude Code only speaks the Anthropic Messages API. Workers AI (``@cf/...``
models) only speaks OpenAI-style ``/ai/v1/chat/completions`` -- Cloudflare's own
``/ai/v1/messages`` route does not accept ``@cf/`` models. So the local Gateway
translates, per request:

    Claude Code --/v1/messages--> Gateway --/chat/completions--> Cloudflare

This module holds everything Cloudflare-specific:

  * request/response translation (non-streaming and SSE streaming),
  * model listing for the wizard's "pick a model for each worker" step,
  * credential lookup (env vars first, so a token never has to be typed or shown).

Nothing here logs or returns a raw token.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import aiohttp

CF_API_BASE = os.environ.get("CLOUDFLARE_API_BASE", "https://api.cloudflare.com/client/v4").rstrip("/")


def ai_base_url(account_id: str) -> str:
    """OpenAI-compatible root for one account (ends at /ai/v1)."""
    return f"{CF_API_BASE}/accounts/{account_id}/ai/v1"


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
@dataclass
class CfCreds:
    account_id: str
    token: str
    gateway_id: str = ""


def creds_from_env() -> tuple[Optional[str], Optional[str], str]:
    acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID") or os.environ.get("CF_ACCOUNT_ID")
    tok = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CF_API_TOKEN")
    gw = os.environ.get("CLOUDFLARE_AI_GATEWAY_ID", "")
    return acct, tok, gw


# --------------------------------------------------------------------------
# model listing
# --------------------------------------------------------------------------
@dataclass
class CfModel:
    id: str                         # "@cf/moonshotai/kimi-k2.7-code"
    description: str = ""
    context_window: Optional[int] = None
    function_calling: Optional[bool] = None

    @property
    def short(self) -> str:
        return self.id[len("@cf/"):] if self.id.startswith("@cf/") else self.id


def _prop(model: dict, key: str) -> Optional[str]:
    for p in model.get("properties") or []:
        if p.get("property_id") == key:
            return str(p.get("value"))
    return None


def parse_models(payload: dict) -> list[CfModel]:
    out = []
    for m in payload.get("result") or []:
        name = m.get("name")
        if not name:
            continue
        ctx = _prop(m, "context_window") or _prop(m, "max_total_tokens")
        fc = _prop(m, "function_calling")
        out.append(CfModel(
            id=name, description=(m.get("description") or "").strip(),
            context_window=int(ctx) if ctx and ctx.isdigit() else None,
            function_calling=None if fc is None else fc.lower() == "true"))
    return out


async def list_text_models(creds: CfCreds, timeout_s: float = 30.0) -> list[CfModel]:
    """GET /accounts/{id}/ai/models/search?task=Text Generation (needs Workers AI Read)."""
    url = f"{CF_API_BASE}/accounts/{creds.account_id}/ai/models/search"
    models: list[CfModel] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as s:
        page = 1
        while page <= 10:
            async with s.get(url, params={"task": "Text Generation", "per_page": 100, "page": page},
                             headers={"Authorization": f"Bearer {creds.token}"}) as r:
                data = await r.json(content_type=None)
                if r.status != 200 or data.get("success") is False:
                    raise RuntimeError(describe_error(r.status, data))
            batch = parse_models(data)
            models += batch
            info = data.get("result_info") or {}
            total_pages = info.get("total_pages") or 1
            if page >= total_pages or not batch:
                break
            page += 1
    models.sort(key=lambda m: m.id)
    return models


def describe_error(status: int, data: Any) -> str:
    msg = ""
    if isinstance(data, dict):
        errs = data.get("errors") or []
        if errs and isinstance(errs[0], dict):
            msg = errs[0].get("message") or ""
        elif isinstance(data.get("error"), dict):
            msg = data["error"].get("message") or ""
        elif isinstance(data.get("error"), str):
            msg = data["error"]
    hint = {401: "token invalid or missing the Account > Workers AI > Read permission",
            403: "token not allowed to use Workers AI on this account",
            404: "account id or model id not found",
            429: "rate limited (Workers AI quota or concurrency)"}.get(status, "")
    return f"HTTP {status}" + (f" ({hint})" if hint else "") + (f": {msg}" if msg else "")


# --------------------------------------------------------------------------
# Anthropic request  ->  OpenAI chat.completions request
# --------------------------------------------------------------------------
def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def _tool_result_text(block: dict) -> str:
    c = block.get("content")
    if isinstance(c, str):
        text = c
    else:
        bits = []
        for b in c or []:
            if isinstance(b, dict) and b.get("type") == "text":
                bits.append(b.get("text", ""))
            elif isinstance(b, dict) and b.get("type") == "image":
                bits.append("[image omitted]")
        text = "\n".join(bits)
    if block.get("is_error"):
        text = "ERROR: " + text
    return text or "(no output)"


def _image_part(block: dict) -> Optional[dict]:
    src = block.get("source") or {}
    if src.get("type") == "base64":
        return {"type": "image_url",
                "image_url": {"url": f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"}}
    if src.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": src.get("url", "")}}
    return None


def to_openai_request(req: dict, model: str) -> dict:
    msgs: list[dict] = []
    system = req.get("system")
    if system:
        text = _text_of(system)
        if text:
            msgs.append({"role": "system", "content": text})

    for m in req.get("messages", []):
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        content = content or []
        if role == "assistant":
            text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                     for b in content if b.get("type") == "tool_use"]
            am: dict = {"role": "assistant", "content": text}
            if calls:
                am["tool_calls"] = calls
            if text or calls:
                msgs.append(am)
        else:  # user
            # tool results must directly follow the assistant tool_calls, so they go first
            for b in content:
                if b.get("type") == "tool_result":
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""),
                                 "content": _tool_result_text(b)})
            parts: list[dict] = []
            for b in content:
                if b.get("type") == "text" and b.get("text"):
                    parts.append({"type": "text", "text": b["text"]})
                elif b.get("type") == "image":
                    ip = _image_part(b)
                    if ip:
                        parts.append(ip)
            if parts:
                if all(p["type"] == "text" for p in parts):
                    msgs.append({"role": "user", "content": "\n".join(p["text"] for p in parts)})
                else:
                    msgs.append({"role": "user", "content": parts})

    out: dict = {"model": model, "messages": msgs}
    if req.get("max_tokens"):
        out["max_tokens"] = req["max_tokens"]
    for k in ("temperature", "top_p"):
        if req.get(k) is not None:
            out[k] = req[k]
    if req.get("stop_sequences"):
        out["stop"] = req["stop_sequences"]
    if req.get("stream"):
        out["stream"] = True

    tools = [t for t in req.get("tools") or [] if t.get("name")]
    if tools:
        out["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}} for t in tools]
        tc = req.get("tool_choice")
        if isinstance(tc, dict):
            kind = tc.get("type")
            if kind == "any":
                out["tool_choice"] = "required"
            elif kind == "tool" and tc.get("name"):
                out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
            elif kind == "none":
                out["tool_choice"] = "none"
            else:
                out["tool_choice"] = "auto"
    return out


# --------------------------------------------------------------------------
# OpenAI response  ->  Anthropic message
# --------------------------------------------------------------------------
STOP_MAP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
            "function_call": "tool_use", "content_filter": "end_turn"}


def new_msg_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def _args_to_obj(args: Any) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.strip():
        try:
            v = json.loads(args)
            return v if isinstance(v, dict) else {"value": v}
        except json.JSONDecodeError:
            return {}
    return {}


def _args_to_str(args: Any) -> str:
    if isinstance(args, str):
        return args
    if args is None:
        return ""
    return json.dumps(args)


def unwrap(data: dict) -> dict:
    """Tolerate the native-API envelope {"result": {...}, "success": true}."""
    if "choices" not in data and isinstance(data.get("result"), dict):
        return data["result"]
    return data


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def to_anthropic_response(data: dict, model: str, est_input_chars: int = 0) -> dict:
    data = unwrap(data)
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    blocks: list[dict] = []
    text = msg.get("content")
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    if text:
        blocks.append({"type": "text", "text": text})
    calls = msg.get("tool_calls") or []
    for c in calls:
        fn = c.get("function") or {}
        blocks.append({"type": "tool_use", "id": c.get("id") or "toolu_" + uuid.uuid4().hex[:20],
                       "name": fn.get("name", ""), "input": _args_to_obj(fn.get("arguments"))})
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    stop = "tool_use" if calls else STOP_MAP.get(choice.get("finish_reason") or "stop", "end_turn")
    u = data.get("usage") or {}
    out_text = (text or "") + "".join(_args_to_str((c.get("function") or {}).get("arguments")) for c in calls)
    return {"id": new_msg_id(), "type": "message", "role": "assistant", "model": model,
            "content": blocks, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": u.get("prompt_tokens") or est_input_chars // 4,
                      "output_tokens": u.get("completion_tokens") or estimate_tokens(out_text)}}


def anthropic_error(status: int, message: str) -> dict:
    kind = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
            404: "not_found_error", 429: "rate_limit_error"}.get(status, "api_error")
    return {"type": "error", "error": {"type": kind, "message": message}}


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


# --------------------------------------------------------------------------
# OpenAI SSE stream  ->  Anthropic SSE stream
# --------------------------------------------------------------------------
async def iter_sse_data(content: aiohttp.StreamReader) -> AsyncIterator[str]:
    """Yield the payload of each ``data:`` line (handles chunks split mid-line)."""
    buf = b""
    async for chunk in content.iter_any():
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if line.startswith(b"data:"):
                yield line[5:].strip().decode("utf-8", "replace")
    tail = buf.strip()
    if tail.startswith(b"data:"):
        yield tail[5:].strip().decode("utf-8", "replace")


class StreamTranslator:
    """Stateful OpenAI-chunk -> Anthropic-event converter.

    Feed it decoded ``data:`` payloads with ``feed()``; call ``finish()`` at the end.
    Each returns a list of encoded SSE events ready to write to the client.
    """

    def __init__(self, model: str, est_input_chars: int = 0):
        self.model = model
        self.est_in = est_input_chars // 4
        self.started = False
        self.block_idx = -1
        self.open_kind: Optional[str] = None       # "text" | "tool"
        self.tool_key: Any = None                   # which upstream tool call is open
        self.finish_reason: Optional[str] = None
        self.saw_tool = False
        self.in_tokens: Optional[int] = None
        self.out_tokens: Optional[int] = None
        self.out_chars = 0
        self.done = False

    def _start(self) -> list[bytes]:
        if self.started:
            return []
        self.started = True
        return [sse("message_start", {"type": "message_start", "message": {
            "id": new_msg_id(), "type": "message", "role": "assistant", "model": self.model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": self.est_in, "output_tokens": 0}}})]

    def _close(self) -> list[bytes]:
        if self.open_kind is None:
            return []
        self.open_kind, self.tool_key = None, None
        return [sse("content_block_stop", {"type": "content_block_stop", "index": self.block_idx})]

    def _open(self, kind: str, block: dict) -> list[bytes]:
        ev = self._close()
        self.block_idx += 1
        self.open_kind = kind
        ev.append(sse("content_block_start", {"type": "content_block_start",
                                              "index": self.block_idx, "content_block": block}))
        return ev

    def feed(self, payload: str) -> list[bytes]:
        if payload == "[DONE]":
            self.done = True
            return []
        try:
            obj = unwrap(json.loads(payload))
        except json.JSONDecodeError:
            return []
        ev = self._start()
        u = obj.get("usage")
        if isinstance(u, dict):
            self.in_tokens = u.get("prompt_tokens", self.in_tokens)
            self.out_tokens = u.get("completion_tokens", self.out_tokens)
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                if self.open_kind != "text":
                    ev += self._open("text", {"type": "text", "text": ""})
                self.out_chars += len(text)
                ev.append(sse("content_block_delta", {"type": "content_block_delta", "index": self.block_idx,
                                                      "delta": {"type": "text_delta", "text": text}}))
            for tc in delta.get("tool_calls") or []:
                self.saw_tool = True
                fn = tc.get("function") or {}
                key = tc.get("index", tc.get("id"))
                new_call = (self.open_kind != "tool") or (
                    key is not None and key != self.tool_key) or (
                    tc.get("id") and key is None and tc.get("id") != self.tool_key)
                if new_call:
                    ev += self._open("tool", {"type": "tool_use",
                                              "id": tc.get("id") or "toolu_" + uuid.uuid4().hex[:20],
                                              "name": fn.get("name", ""), "input": {}})
                    self.tool_key = key if key is not None else tc.get("id")
                args = _args_to_str(fn.get("arguments"))
                if args:
                    self.out_chars += len(args)
                    ev.append(sse("content_block_delta", {"type": "content_block_delta", "index": self.block_idx,
                                                          "delta": {"type": "input_json_delta", "partial_json": args}}))
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return ev

    def finish(self) -> list[bytes]:
        ev = self._start()
        if self.block_idx < 0:  # model produced nothing: still return a well-formed message
            ev += self._open("text", {"type": "text", "text": ""})
        ev += self._close()
        stop = "tool_use" if self.saw_tool else STOP_MAP.get(self.finish_reason or "stop", "end_turn")
        out = self.out_tokens if self.out_tokens is not None else estimate_tokens("x" * self.out_chars)
        usage = {"output_tokens": out}
        if self.in_tokens is not None:
            usage["input_tokens"] = self.in_tokens
        ev.append(sse("message_delta", {"type": "message_delta",
                                        "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": usage}))
        ev.append(sse("message_stop", {"type": "message_stop"}))
        return ev
