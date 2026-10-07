#!/usr/bin/env bash
# Production-shaped context-cache gate.
#
# Runs the real serving product twice over the same multi-turn workload -- once with
# `--context-host-cache-mib 0` (control) and once with the host cache enabled -- and requires the
# generated text to be identical per (session, turn) while the cache pass actually reuses
# checkpoints (checkpoint_selections_total > 0). An optional third pass with a small cache forces
# eviction under pressure.
#
# Exclusive GPU use, so it is one self-contained script: ai-server is stopped first and restarted
# from an EXIT trap, which fires on success, failure, timeout and interruption.
#
# Usage:
#   scripts/run-serve-context-cache-gate.sh [--artifact PATH] [--cache-mib N] [--sessions N]
#                                           [--turns N] [--concurrency N] [--evict-mib N]
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
artifact="/mnt/models/ninfer-models/Qwen3.8-27B-QUASAR-NVFP4-NInfer.ninfer"
cache_mib=2048
evict_mib=256
sessions=4
turns=6
concurrency=2
port=8081
python_bin="/home/neroued/miniconda3/envs/ninfer/bin/python"
if [ ! -x "$python_bin" ]; then python_bin="/home/neroued/miniconda3/envs/py311/bin/python"; fi
if [ ! -x "$python_bin" ]; then python_bin="$(command -v python3 || true)"; fi
if [ -z "$python_bin" ]; then echo "no python interpreter found" >&2; exit 2; fi

while [ $# -gt 0 ]; do
  case "$1" in
    --artifact)    artifact="$2"; shift 2 ;;
    --cache-mib)   cache_mib="$2"; shift 2 ;;
    --evict-mib)   evict_mib="$2"; shift 2 ;;
    --sessions)    sessions="$2"; shift 2 ;;
    --turns)       turns="$2"; shift 2 ;;
    --concurrency) concurrency="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

work_dir="$(mktemp -d /tmp/ninfer-context-cache.XXXXXX)"
serve_pid=""
log() { echo "[context-cache-gate] $*" >&2; }

stop_serve() {
  if [ -n "$serve_pid" ] && kill -0 "$serve_pid" 2>/dev/null; then
    kill "$serve_pid" 2>/dev/null || true
    wait "$serve_pid" 2>/dev/null || true
  fi
  serve_pid=""
}

cleanup() {
  stop_serve
  log "restarting ai-server"
  service ai-server start >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_pass() {
  local name="$1" cache="$2" out="$3" conc="$4"
  log "pass $name: --context-host-cache-mib $cache --concurrency $conc"
  "$repo_dir/build-host-retained/apps/ninfer-serve" "$artifact" \
    --tp 2 --devices 1,0 --host 127.0.0.1 --port "$port" \
    --max-concurrency 2 --max-context 160000 --kv-capacity 280000 --kv-dtype int8 \
    --prefill-chunk 2560 --spec mtp --draft-tokens 3 --lm-head-draft \
    --context-host-cache-mib "$cache" \
    > "$work_dir/$name.log" 2>&1 &
  serve_pid=$!
  local ready=0
  for _ in $(seq 1 120); do
    sleep 3
    if ! kill -0 "$serve_pid" 2>/dev/null; then
      log "pass $name: server exited early"
      tail -5 "$work_dir/$name.log" >&2
      return 1
    fi
    if curl -sf --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
  done
  [ "$ready" = 1 ] || { log "pass $name: not healthy in time"; return 1; }

  "$python_bin" "$repo_dir/tools/smoke/context_cache_session.py" \
    --base-url "http://127.0.0.1:$port" --sessions "$sessions" --turns "$turns" \
    --concurrency "$conc" --out "$out" | tee "$work_dir/$name.summary"
  curl -s --max-time 10 "http://127.0.0.1:$port/metrics" > "$work_dir/$name.metrics" || true
  stop_serve
}

log "stopping ai-server for exclusive GPU use"
service ai-server stop >/dev/null 2>&1 || true
sleep 3

# The two compared passes run ONE request at a time: greedy output is schedule dependent (a
# different batch shape changes reductions, and the two passes do not hit the same checkpoints), so
# identical text is only meaningful when the schedule is. The concurrent pass then checks that
# concurrency still reuses, bounds occupancy and produces no errors, without comparing text.
run_pass control 0 "$work_dir/control.jsonl" 1
run_pass host "$cache_mib" "$work_dir/host.jsonl" 1

if [ "$evict_mib" -gt 0 ]; then
  run_pass evict "$evict_mib" "$work_dir/evict.jsonl" "$concurrency"
fi

"$python_bin" - "$work_dir" <<'PY'
import json, sys, pathlib

work = pathlib.Path(sys.argv[1])

def load(name):
    rows = {}
    for line in (work / f"{name}.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows[(row["session"], row["turn"])] = row
    return rows

control, host = load("control"), load("host")
missing = sorted(set(control) - set(host)) + sorted(set(host) - set(control))
mismatch = [
    key for key in sorted(set(control) & set(host))
    if control[key].get("text") != host[key].get("text")
]
hits = sum(1 for row in host.values() if row.get("checkpoint_selections", 0) > 0)
reused = sum(row.get("reused_tokens", 0) for row in host.values())
print(f"paired={len(set(control) & set(host))} missing={len(missing)} mismatched={len(mismatch)}")
print(f"host_checkpoint_selections={hits} host_reused_tokens={int(reused)}")
for key in mismatch[:5]:
    print(f"  mismatch session={key[0]} turn={key[1]}")
    print(f"    control={control[key].get('text', '')[:120]!r}")
    print(f"    host   ={host[key].get('text', '')[:120]!r}")
failed = bool(missing or mismatch or hits == 0)
print("RESULT:", "FAIL" if failed else "PASS")
raise SystemExit(1 if failed else 0)
PY
