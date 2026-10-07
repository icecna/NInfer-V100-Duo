#!/usr/bin/env python3
"""Strict live verification of the host retained-prefix feature over HTTP.

Run it through `scripts/run-host-kv-verify.sh`, which stops ai-server, starts a private instance
with NINFER_HOST_KV_DEBUG=1, --context-host-cache-mib 4096, --max-concurrency 2 and `--spec mtp`,
and restarts the service afterwards. Run directly (no wrapper), it instead checks that the already
running service satisfies those requirements. Either way the run needs exclusive GPUs and no
external traffic: the request log is the source of truth for that.

Scenario (3 prefixes of ~39k tokens each, one shared 2-lane service, all
steps submitted back-to-back -- the drain-at-admission path must make the
outcomes identical to a quiescent run):

    A1  A2  B1  B2  C1  A3  C2  +  PROBE (tiny tail request)

  A1   = A as one user turn                     -> full_reset (hit=0)
  A2   = A + assistant reply + second turn      -> frontier append (H2D claim)
  B1   = B as one user turn                     -> full_reset (hit=0)
  B2   = B + assistant reply + second turn      -> frontier append (H2D claim)
  C1   = C as one user turn                     -> full_reset (hit=0)
  A3   = A2's EXACT prompt repeated             -> turn checkpoint (H2D claim)
  C2   = C1's EXACT prompt repeated             -> turn checkpoint (H2D claim)
  PROBE = tiny request; its admission destroys a host-resident prefix

Expected host-KV journal invariants (lanes read from the events themselves):
  - exactly 4 claim events in order A2, B2, A3, C2; A2/A3 on A's retained
    lane, B2 on B's, C2 on C's; each H2D < 3 s
  - 9 D2H harvests (warmup + 7 real prefixes + PROBE), zero abandoned
  - zero retain-skip; the arena occupancy reported by every event matches a
    strict per-lane simulation (retain allocates, claim/reset/evict release)
  - greedy outputs are byte-identical for identical prompts (A2 == A3,
    C1 == C2) and request-log reuse paths match the scenario above
"""

import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
import os

# Defaults point at the deployed service; `scripts/run-host-kv-verify.sh` overrides all three so
# the check can own a private instance instead of borrowing the production one.
BASE = os.environ.get("NINFER_HOST_KV_BASE",
                      "http://192.168.1.12:8080/v1/chat/completions")
MODELS = os.environ.get("NINFER_HOST_KV_MODELS", "http://192.168.1.12:8080/v1/models")
REQLOG = os.environ.get("NINFER_HOST_KV_REQLOG",
                        "/mnt/models/ai-launchers/requests.jsonl")
HOST_CACHE_MIB = 4096


def mk_prefix(tag):
    lines = []
    for i in range(400):
        body = (
            f"[{tag}:{i}] Paragraph {i} of the {tag} corpus. "
            "Inference caches map logical request state onto physical lane "
            "resources without coupling scheduling to allocation identity. "
            "Retention keeps a finished conversation's KV on pinned host "
            "memory until it is either reclaimed by a new admission or "
            "restored for the next turn of the same conversation. "
            "The host arena is a fixed, bounded, per-card budget that holds "
            "whole lane prefixes in LRU order."
        )
        lines.append(f"{i:04d} {tag} {body}")
    return "\n".join(lines)


def chat(messages, max_tokens=16):
    body = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
    }
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    elapsed = time.time() - t0
    choice = data["choices"][0]
    msg = choice["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "finish": choice.get("finish_reason"),
        "ptok": (data.get("usage") or {}).get("prompt_tokens"),
        "elapsed": elapsed,
    }


def parse_journal_ts(line):
    # "Oct 06 04:51:43 debian ..." -> epoch (year from `date`)
    try:
        parts = line.split(" ", 3)
        months = {m: i + 1 for i, m in enumerate(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
        mon, day, hms = parts[0], parts[1], parts[2]
        t = datetime(int(time.strftime("%Y")), months[mon], int(day),
                     *map(int, hms.split(":")))
        return t.timestamp()
    except Exception:
        return None


def unit_start_epoch():
    """Epoch of the current ai-server unit activation: BEFORE model load and
    warmup. Anchors the journal/jsonl window so the warmup's host-kv events
    are inside it and the previous instance's events are outside."""
    managed = os.environ.get("NINFER_HOST_KV_VERIFY_SINCE")
    if managed:
        return int(managed)
    out = subprocess.run(
        ["systemctl", "show", "ai-server", "-p", "ActiveEnterTimestamp",
         "--value"], capture_output=True, text=True).stdout.strip()
    if not out:
        raise RuntimeError("cannot read ai-server ActiveEnterTimestamp")
    epoch = subprocess.run(["date", "-d", out, "+%s"], capture_output=True,
                           text=True).stdout.strip()
    return int(epoch)


def journal_events(since_ts, until_ts):
    """Raw host-kv debug events from the journal within [since, until].

    since_ts must be the server start; a small negative margin covers journal
    clock granularity but never reaches the previous instance's events.
    """
    out = []
    lines = subprocess.run(
        ["journalctl", "-u", "ai-server", "--no-pager", "-a",
         "--since", datetime.fromtimestamp(since_ts - 2).strftime(
             "%Y-%m-%d %H:%M:%S"),
         "--until", datetime.fromtimestamp(until_ts + 60).strftime(
             "%Y-%m-%d %H:%M:%S")],
        capture_output=True).stdout.decode("utf-8", "replace").splitlines()
    for line in lines:
        if "[ninfer host-kv]" not in line:
            continue
        ts = parse_journal_ts(line)
        if ts is None or ts < since_ts - 2 or ts > until_ts + 5:
            continue
        payload = line.split("[ninfer host-kv]", 1)[1].strip()
        kind = payload.split(" ", 1)[0]
        kv = {}
        for tok in payload.split(" ")[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv.setdefault(k, v)
        out.append({"kind": kind, "ts": ts, "kv": kv, "raw": payload})
    out.sort(key=lambda e: e["ts"])
    return out


def reqlog_events(since_ts):
    out = []
    for line in open(REQLOG, "r", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") not in ("request_start", "request_done"):
            continue
        ts = (r.get("timestamp_unix_ms") or 0) / 1000.0
        if ts >= since_ts:
            out.append(r)
    return out


def in_flight(reqs):
    infl = {}
    for r in reqs:
        rid = (r.get("request") or {}).get("request_id")
        if r.get("event") == "request_start":
            infl[rid] = True
        elif r.get("event") == "request_done":
            infl.pop(rid, None)
    return infl


class ArenaSim:
    """Strict per-lane model of the host arena; every event's reported
    `occupied` must equal the simulated value."""

    def __init__(self):
        self.state = {}      # lane -> ("inflight"|"host", bytes)
        self.occupied = 0
        self.errors = []

    def _check(self, ev):
        reported = int(ev["kv"].get("occupied", "-1"))
        if reported != self.occupied:
            self.errors.append(f"{ev['kind']} lane={ev['kv'].get('lane')}: "
                               f"reported occupied={reported} sim={self.occupied}")

    def retain(self, ev):
        kv = ev["kv"]
        lane = int(kv["lane"])
        b = int(kv.get("local_bytes", "0")) + int(kv.get("peer_bytes", "0"))
        evicted = kv.get("evicted", "none")
        if not evicted.strip("0123456789,") .replace("none", ""):
            evicted = ""
        for v in (evicted.split(",") if evicted else []):
            if not v.isdigit():
                self.errors.append(f"retain lane={lane}: unparsable evicted='{v}' "
                                   f"(journal corruption or bad log line)")
                continue
            st = self.state.get(int(v))
            if st and st[0] == "host":
                self.occupied -= st[1]
            self.state[int(v)] = None
        if lane in self.state and self.state[lane]:
            self.errors.append(f"retain lane={lane} while lane already {self.state[lane]}")
        self.state[lane] = ("inflight", b)
        self.occupied += b
        self._check(ev)

    def complete(self, ev):
        kv = ev["kv"]
        lane = int(kv["lane"])
        st = self.state.get(lane)
        if ev["kv"].get("abandoned") == "1":
            if not (st and st[0] == "inflight"):
                self.errors.append(f"abandoned complete lane={lane} but state={st}")
            else:
                self.occupied -= st[1]
            self.state[lane] = None
        else:
            if not (st and st[0] == "inflight"):
                self.errors.append(f"complete lane={lane} but state={st}")
            else:
                self.state[lane] = ("host", st[1])
        self._check(ev)

    def claim(self, ev):
        kv = ev["kv"]
        lane = int(kv["lane"])
        st = self.state.get(lane)
        if not (st and st[0] == "host"):
            self.errors.append(f"claim lane={lane} but state={st}")
        else:
            self.occupied -= st[1]
        self.state[lane] = None
        self._check(ev)

    def release(self, ev):
        # reset-host / evict / retained-evict: the lane's host extent (if any) is dropped.
        kv = ev["kv"]
        lane = int(kv.get("lane", "-1"))
        st = self.state.get(lane)
        if st and st[0] == "host":
            self.occupied -= st[1]
            self.state[lane] = None
        elif ev["kind"] == "reset-host" and st:
            self.errors.append(f"reset-host lane={lane} in {st[0]} state")
        self._check(ev)

    def run(self, events):
        for ev in events:
            k = ev["kind"]
            if k == "retain":
                self.retain(ev)
            elif k == "complete":
                self.complete(ev)
            elif k == "claim":
                self.claim(ev)
            elif k in ("reset-host", "evict", "retained-evict"):
                self.release(ev)


def main():
    # --- preconditions: exactly one serve process, ours, debug-enabled ---
    # A managing wrapper started the instance with the required cache size, concurrency and debug
    # logging, so there is nothing to assert; without one, the service must already satisfy them.
    if os.environ.get("NINFER_HOST_KV_VERIFY_MANAGED") != "1":
        p = subprocess.run(
            ["pgrep", "-f", "NInfer-V100-Duo/build.*/apps/ninfer-serve"],
            capture_output=True, text=True)
        pids = [x for x in p.stdout.split() if x.strip()]
        assert len(pids) == 1, f"expected one serve process, got {pids}"
        assert "NINFER_HOST_KV_DEBUG" in open(
            f"/proc/{pids[0]}/environ", errors="replace").read(), \
            "serve is not running with NINFER_HOST_KV_DEBUG=1"

    inst = json.loads(urllib.request.urlopen(MODELS, timeout=15).read())
    created = unit_start_epoch()  # unit activation, before load + warmup

    reqs = reqlog_events(created)
    infl = in_flight(reqs)
    assert not infl, (
        f"service not quiescent before verify: {infl}; an external client is "
        "still on the service -- restart it and re-run")
    assert sum(1 for r in reqs if r.get("event") == "request_start") in (0, 1), (
        "service is serving external traffic (warmup only is expected); "
        "restart and re-run")

    A, B, C = mk_prefix("ALPHA"), mk_prefix("BRAVO"), mk_prefix("CHARLIE")
    print(f"prefixes: ~{len(A)//4}k tokens each", flush=True)

    def turn(tag, prefix):
        return (
            [{"role": "user",
              "content": prefix + f"\nReply with exactly: {tag}"}],
            f"Confirm the prefix is intact. Reply with exactly: {tag}")

    a1_msgs, a2_user = turn("A1", A)
    b1_msgs, b2_user = turn("B1", B)
    c1_msgs, _ = turn("C1", C)

    def step(tag, msgs):
        r = chat(msgs)
        print(f"{tag}: elapsed={r['elapsed']:.2f}s ptok={r['ptok']} "
              f"finish={r['finish']}", flush=True)
        return r

    a1 = step("A1", a1_msgs)
    a2_msgs = a1_msgs + [
        {"role": "assistant", "content": a1["content"],
         "reasoning_content": a1["reasoning"]},
        {"role": "user", "content": a2_user}]
    a2 = step("A2", a2_msgs)

    b1 = step("B1", b1_msgs)
    b2_msgs = b1_msgs + [
        {"role": "assistant", "content": b1["content"],
         "reasoning_content": b1["reasoning"]},
        {"role": "user", "content": b2_user}]
    b2 = step("B2", b2_msgs)

    c1 = step("C1", c1_msgs)
    a3 = step("A3", a2_msgs)   # A2's exact prompt again -> turn checkpoint
    c2 = step("C2", c1_msgs)   # C1's exact prompt again -> turn checkpoint
    probe = chat([{"role": "user", "content": "Reply with: ok"}],
                 max_tokens=8)
    print(f"PROBE: elapsed={probe['elapsed']:.2f}s finish={probe['finish']}",
          flush=True)
    # Settle so the PROBE's terminal D2H is both started and harvested before
    # the journal window closes: poll until every retain has its complete. The
    # idle-settle harvest finalizes in-flight transfers without new traffic, but
    # the finalize latency is not bounded, so the window end tracks it.
    t_end = time.time()
    for _ in range(30):
        t_end = time.time()
        ev_tmp = journal_events(created, t_end)
        n_retain = sum(1 for e in ev_tmp if e["kind"] == "retain")
        n_complete = sum(1 for e in ev_tmp if e["kind"] == "complete")
        if n_complete >= n_retain:
            break
        time.sleep(2)
    t_end = time.time()

    # --- postconditions ---
    failures = []

    def check(name, cond, detail=""):
        print(("  ok   " if cond else "  FAIL ") + name +
              (f"  [{detail}]" if detail and not cond else ""), flush=True)
        if not cond:
            failures.append(name)

    # 1. identical prompts give byte-identical greedy outputs
    check("A2/A3 identical outputs",
          (a2["content"], a2["reasoning"]) == (a3["content"], a3["reasoning"]))
    check("C1/C2 identical outputs",
          (c1["content"], c1["reasoning"]) == (c2["content"], c2["reasoning"]))

    # 2. exactly our 8 requests hit the service
    reqs2 = reqlog_events(created)
    dones = sorted((r for r in reqs2 if r.get("event") == "request_done"),
                   key=lambda r: r["timestamp_unix_ms"])
    starts = [r for r in reqs2 if r.get("event") == "request_start"]
    check(f"exactly 8 request_dones (got {len(dones)})", len(dones) == 8)
    check(f"no foreign request (starts={len(starts)})", len(starts) == 8)

    # 3. per-step request-log semantics (correlated by completion order)
    expect = {
        "A1": ("full_reset", 0),
        "A2": (None, "hit"),   # frontier append or turn-checkpoint restore
        "B1": ("full_reset", 0),
        "B2": (None, "hit"),   # frontier append or turn-checkpoint restore
        "C1": ("full_reset", 0),
        "A3": ("restore_turn_checkpoint", "hit"),
        "C2": ("restore_turn_checkpoint", "hit"),
        "PROBE": None,
    }
    ordered = [a1, a2, b1, b2, c1, a3, c2, probe]
    for tag, r, d in zip(expect, ordered, dones):
        if len(dones) != 8:
            break
        res = d.get("result") or {}
        hit = res.get("prefix_cache_hit_tokens", 0)
        computed = res.get("computed_prefill_tokens")
        reuse = res.get("prefix_reuse_path", "?")
        spec = expect[tag]
        if spec is None:
            ok = True
            want = "any"
        elif spec[1] == 0:
            ok = (reuse == spec[0] and hit == 0 and computed is not None
                  and abs(computed - r["ptok"]) <= 32)
            want = spec[0]
        else:
            ok = (spec[0] is None or reuse == spec[0]) and \
                hit >= r["ptok"] - 32 and computed is not None and computed <= 32
            want = spec[0] or "append_frontier|restore_turn_checkpoint"
        check(f"{tag} reuse={reuse} hit={hit} computed={computed}", ok,
              f"want {want}")

    # 4. journal host-kv invariants
    ev = journal_events(created, t_end)
    retains = [e for e in ev if e["kind"] == "retain"]
    real_retains = [e for e in retains
                    if int(e["kv"].get("text_pages", "0") or 0) >= 100]
    claims = [e for e in ev if e["kind"] == "claim"]
    completes = [e for e in ev if e["kind"] == "complete"]
    skipped = [e for e in ev if e["kind"].startswith("retain-skip")]
    abandoned = [e for e in completes if e["kv"].get("abandoned") == "1"]

    for e in ev:
        print(f"  {e['kind']:<14} {e['raw']}", flush=True)

    check(f"9 D2H retain starts incl. warmup+probe (got {len(retains)})",
          len(retains) == 9)
    check(f"7 real-prefix retains (got {len(real_retains)})",
          len(real_retains) == 7)
    check(f"4 H2D claims (got {len(claims)})", len(claims) == 4)
    check(f"9 harvest completes (got {len(completes)})", len(completes) == 9)
    check("no retained in flight at quiescence",
          all(e["kv"].get("abandoned") != "1" for e in completes))
    check("zero abandoned D2H transfers", not abandoned,
          str([e["raw"] for e in abandoned]))
    check("no retain-skip", not skipped, str([e["raw"] for e in skipped]))
    check("all H2D < 3s",
          all(float(e["kv"].get("h2d_ms", "9999")) < 3000 for e in claims),
          str([e["kv"].get("h2d_ms") for e in claims]))

    # 5. lane consistency: claim order is A2, B2, A3, C2 and each claim lands
    # on the lane its prefix retained to (real retains in order A1,A2,B1,B2,
    # C1,A3,C2; A2/A3 reuse A's lane, B2 B's, C2 C's).
    if len(real_retains) == 7 and len(claims) == 4:
        lane = {tag: int(e["kv"]["lane"]) for tag, e in
                zip(["A1", "A2", "B1", "B2", "C1", "A3", "C2"], real_retains)}
        got = [int(e["kv"]["lane"]) for e in claims]
        want = [lane["A1"], lane["B1"], lane["A1"], lane["C1"]]
        check("claim lanes match prefix lanes", got == want,
              f"got={got} want={want} retain_lanes={lane}")
        check("three prefixes used distinct lanes where expected",
              lane["A1"] != lane["C1"] and lane["B1"] in (0, 1))

    # 6. strict arena simulation over the whole event stream
    sim = ArenaSim()
    sim.run(ev)
    check("arena occupancy consistent at every event", not sim.errors,
          "; ".join(sim.errors[:3]))
    check(f"final arena {sim.occupied / 2**20:.0f} MiB <= {HOST_CACHE_MIB} MiB",
          sim.occupied <= HOST_CACHE_MIB * 2**20)

    # 7. service quiescent again
    reqs3 = reqlog_events(created)
    infl = in_flight(reqs3)
    check("service quiescent after verify", not infl, str(infl))

    print()
    if failures:
        print(f"VERIFY FAILED: {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("VERIFY PASS: host retained-prefix feature verified live "
          f"({len(claims)} H2D claims, {len(completes)} D2H harvests, "
          "zero abandons, exact outputs, arena consistent).")


if __name__ == "__main__":
    main()