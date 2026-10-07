"""Drive production-shaped multi-turn sessions against a running NInfer server.

One "session" is a system prompt plus `--turns` user turns; every turn appends the previous
assistant answer and a new user message, so each request is a strict extension of the one before
and the server should reuse the retained checkpoint instead of re-prefilling the whole history.

Records per (session, turn): prompt tokens, time to first streamed content delta, total seconds,
and the generated text, plus `/metrics` before and after. The text is what the paired control pass
compares: with and without `--context-host-cache-mib`, greedy decoding of the same conversation
must produce identical output.

Usage:
  python context_cache_session.py --base-url http://127.0.0.1:8081 --out /tmp/host.jsonl \
      --sessions 4 --turns 6 --concurrency 2
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable


TOPICS = (
    "prefix caching in an inference server",
    "batched decode scheduling",
    "int8 grouped quantisation of a KV cache",
    "speculative decoding with a draft head",
    "tensor parallel execution across two GPUs",
    "host offload of retained context",
)

FILLER = (
    "We are reviewing the design of a local inference engine. "
    "Answer briefly, then list the single most important risk. "
)


def post_json(url: str, payload: dict[str, Any], stream: bool, timeout: float) -> dict[str, Any]:
    """POST one chat completion. Returns {text, ttft, total} (streaming) or the parsed body."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not stream:
            body = json.loads(response.read())
            return {"text": body["choices"][0]["message"]["content"] or "",
                    "ttft": float("nan"), "total": time.monotonic() - started}
        ttft = float("nan")
        text: list[str] = []
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                if ttft != float("nan"):
                    ttft = time.monotonic() - started
                text.append(piece)
        return {"text": "".join(text), "ttft": ttft, "total": time.monotonic() - started}


def scrape(url: str, timeout: float) -> dict[str, float]:
    """Read the Prometheus surface into {name: value}."""
    out: dict[str, float] = {}
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
    except urllib.error.URLError:
        return out
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def session_messages(session: int, topic: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": FILLER * 6},
            {"role": "user", "content": f"Topic {session + 1}: {topic}. Start the review."}]


def run_session(args: argparse.Namespace, session: int) -> list[dict[str, Any]]:
    chat = f"{args.base_url}/v1/chat/completions"
    metrics = f"{args.base_url}/metrics"
    topic = TOPICS[session % len(TOPICS)]
    messages = session_messages(session, topic)
    rows: list[dict[str, Any]] = []
    for turn in range(args.turns):
        before = scrape(metrics, args.timeout)
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        try:
            result = post_json(chat, payload, True, args.timeout)
        except Exception as error:  # noqa: BLE001 - recorded, not raised
            rows.append({"session": session, "turn": turn, "error": str(error)})
            break
        after = scrape(metrics, args.timeout)
        reused = after.get("prompt_tokens_cached_total", 0.0) - before.get(
            "prompt_tokens_cached_total", 0.0)
        rows.append({
            "session": session,
            "turn": turn,
            "text": result["text"],
            "ttft": result["ttft"],
            "total": result["total"],
            "reused_tokens": reused,
            "checkpoint_selections": after.get("checkpoint_selections_total", 0.0)
            - before.get("checkpoint_selections_total", 0.0),
            "host_cache_occupied_bytes": after.get("context_host_cache_occupied_bytes", 0.0),
        })
        messages = messages + [
            {"role": "assistant", "content": result["text"]},
            {"role": "user", "content": f"Turn {turn + 2}: what breaks first under load?"},
        ]
    return rows


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(v for v in values if v == v)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
        for produced in pool.map(lambda s: run_session(args, s), range(args.sessions)):
            with lock:
                rows.extend(produced)

    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    hits = sum(1 for row in rows if row.get("checkpoint_selections", 0.0) > 0)
    reused = sum(row.get("reused_tokens", 0.0) for row in rows)
    ttft_hit = [row["ttft"] for row in rows if row.get("reused_tokens", 0.0) > 0]
    ttft_miss = [row["ttft"] for row in rows if row.get("reused_tokens", 0.0) == 0]
    errors = [row for row in rows if "error" in row]
    print(f"requests={len(rows)} errors={len(errors)} hits={hits} reused_tokens={int(reused)}")
    print(f"ttft_hit_p50={percentile(ttft_hit, 0.5):.3f} ttft_miss_p50={percentile(ttft_miss, 0.5):.3f}")
    for row in errors[:3]:
        print(f"error session={row['session']} turn={row['turn']}: {row['error']}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
