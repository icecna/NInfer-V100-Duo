#!/usr/bin/env python3
"""Discriminate why the third turn of an interleaved conversation loses its checkpoint.

Sends the shape A1 B1 A2 B2 A3 twice. The first pass submits everything back to back; the second
pauses before the last turn. Reuse is read per request from the service's own request log, which
is authoritative (the metrics counters are cumulative and only usable with one request in flight).

If the pause turns A3 from a full recompute into a hit, the cause is the in-flight D2H retain
being abandoned when its lane is reused too soon -- not the content of the history the client
replays, which is identical in both passes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request

CHARS_PER_TOKEN = 6.1
BLOCK = ("The quarterly access review covers authentication, authorization, audit, availability, "
         "and cost controls across every region and tier, for every team and every service. ")
FAKE_ASSISTANT = "Here is the summary you asked for."


def pad(tokens: int, tag: str) -> str:
    return f"Review {tag}. " + BLOCK * max(1, int(tokens * CHARS_PER_TOKEN) // len(BLOCK))


def post(base: str, model: str, messages: list[dict[str, str]], timeout: float) -> float:
    payload = json.dumps({"model": model, "messages": messages, "max_tokens": 4,
                          "temperature": 0.0, "stream": False}).encode()
    request = urllib.request.Request(f"{base}/v1/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
    return time.monotonic() - started


def new_requests(log: pathlib.Path, offset: int) -> list[dict]:
    """Completed requests appended to the service log after `offset` bytes."""
    if not log.exists():
        return []
    with log.open("rb") as handle:
        handle.seek(offset)
        appended = handle.read().decode("utf-8", "replace")
    events = []
    for line in appended.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "request_done":
            events.append(event["result"])
    return events


def run_pass(args, tag: str, pause_before_last: float) -> list[dict]:
    base, model = args.base_url, args.model
    system = {"role": "system", "content": pad(args.tokens, tag)}
    a_user, b_user = f"{tag} ALPHA: start the review.", f"{tag} BRAVO: start the review."
    a = [system, {"role": "user", "content": a_user}]
    b = [system, {"role": "user", "content": b_user}]
    a2 = a + [{"role": "assistant", "content": FAKE_ASSISTANT},
              {"role": "user", "content": f"{tag} ALPHA turn two."}]
    b2 = b + [{"role": "assistant", "content": FAKE_ASSISTANT},
              {"role": "user", "content": f"{tag} BRAVO turn two."}]
    a3 = a2 + [{"role": "assistant", "content": FAKE_ASSISTANT},
               {"role": "user", "content": f"{tag} ALPHA turn three."}]

    log = pathlib.Path(args.request_log)
    rows = []
    for name, messages, pause in (("A1", a, 0.0), ("B1", b, 0.0), ("A2", a2, 0.0),
                                  ("B2", b2, 0.0), ("A3", a3, pause_before_last)):
        if pause:
            time.sleep(pause)
        offset = log.stat().st_size if log.exists() else 0
        seconds = post(base, model, messages, args.timeout)
        time.sleep(0.5)  # let the log flush
        done = new_requests(log, offset)
        result = done[-1] if done else {}
        rows.append({"turn": name, "seconds": seconds,
                     "prompt": result.get("prompt_tokens"),
                     "cached": result.get("prefix_cache_hit_tokens"),
                     "path": result.get("prefix_reuse_path"),
                     "extra": len(done) - 1})
        print(f"  {tag}-{name:<2} {seconds:5.2f}s prompt={rows[-1]['prompt']} "
              f"cached={rows[-1]['cached']} path={rows[-1]['path']}"
              + (f"  (+{rows[-1]['extra']} concurrent request(s) in the window)"
                 if rows[-1]["extra"] > 0 else ""), flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://192.168.1.12:8080")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--request-log", default="/mnt/models/ai-launchers/requests.jsonl")
    parser.add_argument("--tokens", type=int, default=8000)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    print(f"pass 1: back to back (no pause before A3)")
    first = run_pass(args, "R1", 0.0)
    print(f"pass 2: {args.delay:.1f}s pause before A3")
    second = run_pass(args, "R2", args.delay)

    def verdict(rows):
        last = rows[-1]
        return "hit" if (last.get("cached") or 0) > 0 else "full recompute"

    print()
    print(f"A3 without a pause : {verdict(first)}  (cached={first[-1].get('cached')})")
    print(f"A3 with a pause    : {verdict(second)}  (cached={second[-1].get('cached')})")
    if first[-1].get("extra", 0) or second[-1].get("extra", 0):
        print("NOTE: another request completed inside a window; the attribution may be off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
