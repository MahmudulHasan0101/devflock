#!/usr/bin/env python3
"""DevFlock Phase-0 probe for one submachine (llama-server behind a tunnel).

Usage:
  python devflock_probe.py TUNNEL_URL API_KEY [--model qwen] [--big] [--long]

  --big   also test 48k and 100k-token prefills (slow)
  --long  also test a generation that should run longer than 100 seconds

Stdlib only. Tests: metadata, auth styles, streaming, tool calls, prefill speed.
"""
import argparse, json, random, sys, time, urllib.error, urllib.request


def call(base, path, key, payload=None, auth="x-api-key", timeout=900):
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if auth == "x-api-key":
        headers["x-api-key"] = key
    elif auth == "bearer":
        headers["authorization"] = "Bearer " + key
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def sse(resp):
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:"):
            body = line[5:].strip()
            if body and body != "[DONE]":
                try:
                    yield time.time(), json.loads(body)
                except ValueError:
                    pass


def say(level, name, detail=""):
    print(f"[{level:4}] {name}: {detail}")


def msg(model, text, max_tokens=64, **extra):
    return {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": text}], **extra}


def t_meta(a):
    try:
        with call(a.url, "/health", a.key, auth="none", timeout=30) as r:
            say("PASS", "health", f"HTTP {r.status}")
    except Exception as e:
        say("FAIL", "health", str(e))
        return False
    try:
        with call(a.url, "/props", a.key, auth="bearer", timeout=30) as r:
            p = json.load(r)
        n_ctx = p.get("default_generation_settings", {}).get("n_ctx")
        slots = p.get("total_slots")
        level = "PASS" if slots == 1 else "WARN"
        say(level, "props", f"n_ctx={n_ctx} total_slots={slots} "
            "(if slots > 1, per-request context may be smaller: consider --parallel 1)")
    except Exception as e:
        say("WARN", "props", f"could not read /props ({e})")
    return True


def t_auth(a):
    for style in ("x-api-key", "bearer", "none"):
        try:
            with call(a.url, "/v1/messages", a.key, msg(a.model, "say hi", 8), auth=style, timeout=300) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code
        except Exception as e:
            code = str(e)
        expect_ok = style != "none"
        ok = (code == 200) == expect_ok
        say("PASS" if ok else "WARN", f"auth[{style}]", f"HTTP {code}")


def t_stream(a):
    payload = msg(a.model, "Write the numbers 1 to 300 separated by commas. Output only the numbers.",
                  900, stream=True)
    t0 = time.time()
    first = last = first_text = None
    events = out_tokens = 0
    try:
        with call(a.url, "/v1/messages", a.key, payload) as r:
            for ts, ev in sse(r):
                events += 1
                first = first or ts
                last = ts
                if ev.get("type") == "content_block_delta" and first_text is None:
                    first_text = ts
                if ev.get("type") == "message_delta":
                    out_tokens = ev.get("usage", {}).get("output_tokens", out_tokens)
    except Exception as e:
        say("FAIL", "stream", str(e))
        return
    total = time.time() - t0
    if first is None:
        say("FAIL", "stream", "no events received")
        return
    spread = last - first
    buffered = events > 20 and total > 5 and spread < 0.05 * total
    gen = out_tokens / max(last - (first_text or first), 1e-6)
    say("WARN" if buffered else "PASS", "stream",
        f"{events} events, first byte {first - t0:.1f}s, total {total:.1f}s, "
        f"~{gen:.1f} tok/s" + ("  <-- looks BUFFERED (tunnel not streaming)" if buffered else ""))


TOOL = {"name": "write_file", "description": "Write a text file to disk.",
        "input_schema": {"type": "object",
                         "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}


def one_tool_call(a):
    payload = msg(a.model, "Create hello.py that prints hello. You must use the write_file tool.",
                  400, stream=True, tools=[TOOL])
    name, parts = None, []
    with call(a.url, "/v1/messages", a.key, payload) as r:
        for _, ev in sse(r):
            if ev.get("type") == "content_block_start":
                cb = ev.get("content_block", {})
                if cb.get("type") == "tool_use":
                    name = cb.get("name")
            elif ev.get("type") == "content_block_delta":
                d = ev.get("delta", {})
                if d.get("type") == "input_json_delta":
                    parts.append(d.get("partial_json", ""))
    if name != "write_file":
        return False
    args = json.loads("".join(parts) or "{}")
    return isinstance(args.get("path"), str) and isinstance(args.get("content"), str)


def t_tools(a, runs=5):
    ok = 0
    for _ in range(runs):
        try:
            ok += one_tool_call(a)
        except Exception:
            pass
    say("PASS" if ok == runs else ("WARN" if ok else "FAIL"), "tool calls (streamed)", f"{ok}/{runs} valid")


def filler(tokens):
    salt = random.randrange(10**9)
    lines = [f"def f_{salt}_{i}(x): return x * {random.randrange(10**6)} + {i}" for i in range(tokens // 14)]
    return "Summarise in one word.\n" + "\n".join(lines)


def t_prefill(a, size):
    t0 = time.time()
    try:
        with call(a.url, "/v1/messages", a.key, msg(a.model, filler(size), 8)) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        say("FAIL", f"prefill ~{size}", f"HTTP {e.code} after {time.time() - t0:.0f}s")
        return
    except Exception as e:
        say("FAIL", f"prefill ~{size}", f"{e} after {time.time() - t0:.0f}s")
        return
    dt = time.time() - t0
    n = data.get("usage", {}).get("input_tokens", size)
    level = "WARN" if dt > 90 else "PASS"
    say(level, f"prefill ~{size}", f"{n} tokens in {dt:.0f}s (~{n / dt:.0f} tok/s)"
        + ("  <-- near the 100s tunnel limit" if dt > 90 else ""))


def t_long(a):
    payload = msg(a.model, "Write a very long, detailed essay about the history of computing.",
                  4000, stream=True)
    t0 = time.time()
    try:
        with call(a.url, "/v1/messages", a.key, payload, timeout=1200) as r:
            n = sum(1 for _ in sse(r))
        dt = time.time() - t0
        say("PASS" if dt > 100 else "WARN", "long generation",
            f"completed, {n} events, {dt:.0f}s" + ("" if dt > 100 else " (finished under 100s; not a real test)"))
    except Exception as e:
        say("FAIL", "long generation", f"{e} after {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("key")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--long", action="store_true")
    a = ap.parse_args()
    print(f"Probing {a.url} (model={a.model})\n")
    if not t_meta(a):
        sys.exit(1)
    t_auth(a)
    t_stream(a)
    t_tools(a)
    for size in [4000, 16000] + ([48000, 100000] if a.big else []):
        t_prefill(a, size)
    if a.long:
        t_long(a)
    print("\nDone. Save this output: it sets your session-recycle threshold and tunnel choice.")


if __name__ == "__main__":
    main()
