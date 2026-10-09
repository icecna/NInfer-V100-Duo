#!/usr/bin/env python3
"""重放 2026-10-08 两个真实 pi 会话（cortex-console 与 /root）的请求，复现 KV 串台。

请求内容从 pi 会话文件（session jsonl）重建：system sections 拼接、assistant 的
tool_calls + reasoning_content、toolResult -> role=tool。请求的顺序/内容来自会话树的
真实历史；requests.jsonl 只用于核对（它不含 messages）。

用法:
  python3 scripts/replay_kv_crosstalk.py --dry-run
  python3 scripts/replay_kv_crosstalk.py --base-url http://127.0.0.1:18080 \
      --start "2026-10-08T19:35" --end "2026-10-08T19:49" --max-tokens 32
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.request

SESSIONS = {
    "cortex": "/root/.pi/agent/sessions/--mnt-models-cortex-console--/"
              "2026-10-08T18-53-58-897Z_01a11cdd-78f0-72a9-97a3-0d6bdd1246e0.jsonl",
    "root": "/root/.pi/agent/sessions/--root--/"
            "2026-10-08T19-18-54-787Z_01a11cf4-4c42-74a1-8fe9-2c0605621cca.jsonl",
}

# 会话独有的强标记（用于判定串台）
MARKERS = {
    "cortex": ["cortex-console", "admin.html", "启动参数"],
    "root": ["已经reload", "sshpass", "ptycat"],
}


def content_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def system_text(m: dict) -> str:
    parts = [content_text(m)]
    for v in (m.get("sections") or {}).values():
        if v is not None and len(v) > 0:
            parts.append(v)
    return "\n\n".join(p for p in parts if p)


def system_update(m: dict) -> str:
    parts = []
    t = content_text(m)
    if t:
        parts.append(t)
    for name, v in (m.get("sections") or {}).items():
        parts.append(f'Removed system prompt section "{name}".' if v is None
                     else f'Updated system prompt section "{name}":\n\n{v}')
    return "\n\n".join(parts)


def convert_tools(added: list) -> list:
    out = []
    for t in added or []:
        out.append({"type": "function", "function": {
            "name": t.get("name"), "description": t.get("description") or "",
            "parameters": t.get("parameters") or {"type": "object", "properties": {}}}})
    return out


def build_requests(path: str, keep_thinking: bool = False):
    msgs: list[dict] = []
    reqs: list[dict] = []
    tools: list = []
    saw_system = False
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "message":
            continue
        m = e.get("message") or {}
        role, ts = m.get("role"), e.get("timestamp")
        if role == "system":
            if not saw_system:
                saw_system = True
                tools = convert_tools(m.get("toolsAdded"))
                msgs.append({"role": "system", "content": system_text(m), "_ts": ts})
            else:
                msgs.append({"role": "system", "content": system_update(m), "_ts": ts})
        elif role == "user":
            msgs.append({"role": "user", "content": content_text(m), "_ts": ts})
        elif role == "assistant":
            am = {"role": "assistant", "content": content_text(m), "_ts": ts}
            tcs = [b for b in (m.get("content") or []) if isinstance(b, dict) and b.get("type") == "toolCall"]
            if tcs:
                am["tool_calls"] = [{"id": b["id"], "type": "function", "function": {
                    "name": b["name"], "arguments": json.dumps(b.get("arguments") or {}, ensure_ascii=False)}}
                    for b in tcs]
            think = [b.get("thinking", "") for b in (m.get("content") or [])
                     if isinstance(b, dict) and b.get("type") == "thinking"]
            # 默认剥离历史 thinking：否则被污染过的 thinking 会被当作上下文喂回，模型只是复述，
            # 无法区分“引擎新污染”和“重放自我印证”。
            if think and keep_thinking:
                am["reasoning_content"] = "\n".join(think)
            msgs.append(am)
        elif role == "toolResult":
            msgs.append({"role": "tool", "tool_call_id": m.get("toolCallId"),
                         "content": content_text(m), "_ts": ts})
    # 请求点：每条 user，以及每组连续 toolResult 的末尾
    for i, x in enumerate(msgs):
        is_point = x["role"] == "user" or (
            x["role"] == "tool" and (i + 1 >= len(msgs) or msgs[i + 1]["role"] != "tool"))
        if is_point:
            reqs.append({"ts": x.get("_ts"),
                         "messages": [{k: v for k, v in y.items() if k != "_ts"} for y in msgs[:i + 1]]})
    return reqs, tools


def parse_ts(s: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d


def post(base: str, model: str, messages: list, tools: list, max_tokens: int,
         timeout: float) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False,
            "temperature": 0.7, "top_p": 0.95, "top_k": 20,
            "frequency_penalty": 0.0, "presence_penalty": 0.0}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    ch = out.get("choices", [{}])[0]
    msg = ch.get("message", {}) or {}
    return {"content": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "finish": ch.get("finish_reason"),
            "usage": out.get("usage") or {}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:18080")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--limit", type=int, default=0, help="只重放窗口内最后 N 个请求")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default="/tmp/kv-crosstalk-replay.jsonl")
    args = ap.parse_args()

    per_session = {}
    for name, path in SESSIONS.items():
        reqs, tools = build_requests(path)
        per_session[name] = reqs
        if reqs:
            print(f"[{name}] requests={len(reqs)} tools={[t['function']['name'] for t in tools]} "
                  f"first={reqs[0]['ts']} last={reqs[-1]['ts']} "
                  f"max_messages={max(len(r['messages']) for r in reqs)}", flush=True)

    merged = []
    for name, reqs in per_session.items():
        for r in reqs:
            r["session"] = name
            merged.append(r)
    merged.sort(key=lambda r: r["ts"])
    if args.start:
        lo = parse_ts(args.start)
        merged = [r for r in merged if parse_ts(r["ts"]) >= lo]
    if args.end:
        hi = parse_ts(args.end)
        merged = [r for r in merged if parse_ts(r["ts"]) <= hi]
    if args.limit:
        merged = merged[-args.limit:]
    print(f"merged requests to replay: {len(merged)}")
    if args.dry_run:
        for r in merged[:12]:
            print(f"  {r['ts']} {r['session']:6s} messages={len(r['messages'])}")
        print("  ...")
        for r in merged[-6:]:
            print(f"  {r['ts']} {r['session']:6s} messages={len(r['messages'])}")
        return 0

    # tools 取两个会话并集（同工具集）
    tools = convert_tools(
        [{"name": n, "description": "", "parameters": {}} for n in
         ["read", "bash", "edit", "write", "codemode", "subagent", "web_fetch"]])
    # 用会话里真实的工具定义
    for name, path in SESSIONS.items():
        _, t = build_requests(path)
        if t:
            tools = t
            break

    leaks = []
    outp = open(args.out, "w")
    t0 = time.time()
    for i, r in enumerate(merged):
        try:
            res = post(args.base_url, args.model, r["messages"], tools, args.max_tokens, args.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}] {r['session']} FAIL {e!r}", flush=True)
            continue
        text = (res["content"] + "\n" + res["reasoning"])
        prompt = (res["usage"] or {}).get("prompt_tokens")
        rec = {"i": i, "session": r["session"], "ts": r["ts"],
               "messages": len(r["messages"]), "prompt_tokens": prompt,
               "content": res["content"][:400], "reasoning": res["reasoning"][:400]}
        outp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        outp.flush()
        # 串台判定：本会话回复里出现另一会话的标记，且该标记不在本请求自己的 messages 中
        blob = "\n".join(
            (m.get("content") if isinstance(m.get("content"), str) else "")
            + ("\n" + m["reasoning_content"] if isinstance(m.get("reasoning_content"), str) else "")
            for m in r["messages"])
        for other, marks in MARKERS.items():
            if other == r["session"]:
                continue
            for mk in marks:
                if mk in text and mk not in blob:
                    leaks.append((i, r["session"], other, mk, text[:200]))
        if i % 10 == 0 or i == len(merged) - 1:
            print(f"[{i}/{len(merged)}] {r['session']} msgs={len(r['messages'])} "
                  f"prompt={prompt} dt={time.time()-t0:.0f}s", flush=True)
    outp.close()

    print("\n===== 重放判定 =====")
    if leaks:
        print("*** 复现串台 ***")
        for x in leaks:
            print(f"  请求#{x[0]} [{x[1]}] 出现 [{x[2]}] 的标记 {x[3]!r}: {x[4]!r}")
        return 10
    print("未复现串台")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
