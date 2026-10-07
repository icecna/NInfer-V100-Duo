#!/usr/bin/env python3
"""Sweep prompt sizes against an already running `ninfer-serve` to show context-cache reuse.

Unlike `context_cache_session.py`, this script does NOT start or stop a server: it measures
whatever service is serving `--base-url`, so it can be run against the live deployment. For each
prompt size it runs `--sessions` sessions concurrently (bounded by `--concurrency`), each session
extending its own conversation turn by turn, and reports:

  * turns that reused a prefix and the tokens that came from the cache;
  * time to first token for reused turns versus recomputed ones;
  * the reuse share over the whole size, from the server's own `prompt_tokens_cached_total` and
    `computed_prefill_tokens_total` counters;
  * the host cache occupancy before and after the size.

The per-request token counters come from `/metrics`, so a concurrent neighbour can contribute to
the same window; `prompt_est` is therefore an estimate that is only exact when requests do not
overlap (it reads high otherwise). The per-size `reuse_share` uses whole-size counter deltas and
the TTFT values are measured per request, so those are unaffected by concurrency.

The server's reported `prompt_tokens` is authoritative: the padding is an estimate of the rendered
size, not the size itself. Every size gets its own nonce at the start of the system prompt so one
size's prefix cannot be a prefix of another's.

Usage:
  python3 tools/smoke/context_cache_sweep.py --base-url http://192.168.1.12:8080 \\
      --sizes 4096 8192 16384 32768 --sessions 3 --turns 3 --concurrency 2 --out /tmp/cc-sweep
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# The padding text below tokenizes at about 6 characters per token on this tokenizer, measured
# against a server-reported prompt of 42550 tokens. Only the server's count is authoritative.
CHARS_PER_TOKEN = 6.1
BLOCK = ("The quarterly access review covers authentication, authorization, audit, availability, "
         "and cost controls across every region and tier, for every team and every service. ")
COUNTERS = ("prompt_tokens_cached_total", "checkpoint_selections_total",
            "computed_prefill_tokens_total", "committed_decode_tokens_total",
            "context_host_cache_occupied_bytes", "context_host_cache_capacity_bytes")


def scrape(base_url: str, timeout: float) -> dict[str, float]:
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - metrics are observation only
        return {}
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.partition(" ")
        if name in COUNTERS:
            try:
                values[name] = float(value)
            except ValueError:
                pass
    return values


def chat(base_url: str, model: str, messages: list[dict[str, str]], max_tokens: int,
         timeout: float) -> dict[str, Any]:
    payload = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                          "temperature": 0.0, "stream": True}).encode()
    request = urllib.request.Request(f"{base_url}/v1/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    first: float | None = None
    pieces: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    delta = json.loads(body)["choices"][0]["delta"]
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.monotonic() - started
                pieces.append(piece)
    except Exception as error:  # noqa: BLE001 - recorded, not raised
        return {"error": str(error), "total": time.monotonic() - started}
    text = "".join(pieces)
    return {"text": text, "ttft": first if first is not None else time.monotonic() - started,
            "total": time.monotonic() - started, "empty": not text}


def pad_system(size: int, tag: str) -> str:
    filler = f"Review {tag}. "
    target = int(size * CHARS_PER_TOKEN)
    filler += BLOCK * max(1, target // len(BLOCK))
    return filler


def session_rows(args: argparse.Namespace, size: int, session: int) -> list[dict[str, Any]]:
    tag = f"size-{size}-session-{session}"
    messages = [{"role": "system", "content": pad_system(size, tag)},
                {"role": "user", "content": f"Topic {session + 1}: list the risks. Start now."}]
    rows: list[dict[str, Any]] = []
    for turn in range(args.turns):
        before = scrape(args.base_url, args.timeout)
        result = chat(args.base_url, args.model, messages, args.max_tokens, args.timeout)
        after = scrape(args.base_url, args.timeout)
        row = {"size": size, "session": session, "turn": turn, "tag": tag,
               "reused_tokens": after.get("prompt_tokens_cached_total", 0.0)
               - before.get("prompt_tokens_cached_total", 0.0),
               "checkpoint_selections": after.get("checkpoint_selections_total", 0.0)
               - before.get("checkpoint_selections_total", 0.0),
               "computed_prefill": after.get("computed_prefill_tokens_total", 0.0)
               - before.get("computed_prefill_tokens_total", 0.0),
               "host_occupied_bytes": after.get("context_host_cache_occupied_bytes", 0.0),
               "host_capacity_bytes": after.get("context_host_cache_capacity_bytes", 0.0)}
        row.update(result)
        rows.append(row)
        if "error" in result or result.get("empty"):
            break
        messages = messages + [{"role": "assistant", "content": result["text"]},
                               {"role": "user", "content": f"Turn {turn + 2}: narrow to one risk."}]
    return rows


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def sweep_size(args: argparse.Namespace, size: int) -> dict[str, Any]:
    start = scrape(args.base_url, args.timeout)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        batches = list(pool.map(lambda s: session_rows(args, size, s), range(args.sessions)))
    rows = [row for batch in batches for row in batch]
    end = scrape(args.base_url, args.timeout)
    cached = end.get("prompt_tokens_cached_total", 0.0) - start.get("prompt_tokens_cached_total", 0.0)
    computed = (end.get("computed_prefill_tokens_total", 0.0)
                - start.get("computed_prefill_tokens_total", 0.0))
    reused = [row for row in rows if row.get("reused_tokens", 0.0) > 0]
    cold = [row for row in rows if "ttft" in row and row.get("reused_tokens", 0.0) == 0]
    summary = {
        "size": size,
        "requests": len(rows),
        "errors": sum(1 for row in rows if "error" in row),
        "empty": sum(1 for row in rows if row.get("empty")),
        "reused_turns": len(reused),
        "reused_tokens": int(cached),
        "computed_prefill_tokens": int(computed),
        "reuse_share": cached / (cached + computed) if cached + computed else 0.0,
        "prompt_tokens": int(median([row.get("reused_tokens", 0.0) + row.get("computed_prefill", 0.0)
                                     for row in rows] or [0])),
        "ttft_reused": median([row["ttft"] for row in reused]),
        "ttft_cold": median([row["ttft"] for row in cold]),
        "occupied_gib": end.get("context_host_cache_occupied_bytes", 0.0) / 2 ** 30,
        "capacity_gib": end.get("context_host_cache_capacity_bytes", 0.0) / 2 ** 30,
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"size-{size}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"size={size:<6} prompt_est={summary['prompt_tokens']:<6} requests={summary['requests']} "
          f"errors={summary['errors']} empty={summary['empty']} reused_turns={summary['reused_turns']} "
          f"reuse_share={summary['reuse_share']:.2f} "
          f"ttft_hit={summary['ttft_reused']:.3f}s ttft_cold={summary['ttft_cold']:.3f}s "
          f"host_occupied={summary['occupied_gib']:.2f}/{summary['capacity_gib']:.0f}GiB", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://192.168.1.12:8080")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2,
                        help="concurrent sessions; must not exceed the server's --max-concurrency")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not scrape(args.base_url, args.timeout):
        print(f"no /metrics at {args.base_url}; is that the running service?", file=sys.stderr)
        return 2
    summaries = [sweep_size(args, size) for size in args.sizes]
    out = pathlib.Path(args.out)
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print("size      prompt_est  req  err  reused  share  ttft_hit   ttft_cold  host_GiB")
    for item in summaries:
        print(f"{item['size']:<9} {item['prompt_tokens']:<7} {item['requests']:<4} "
              f"{item['errors']:<4} {item['reused_turns']:<7} {item['reuse_share']:<6.2f} "
              f"{item['ttft_reused']:<10.3f} {item['ttft_cold']:<10.3f} {item['occupied_gib']:.2f}")
    return 1 if any(item["errors"] for item in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
