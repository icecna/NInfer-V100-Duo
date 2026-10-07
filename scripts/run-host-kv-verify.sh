#!/usr/bin/env bash
# Live verification of the host retained-prefix feature, owning the instance it tests.
#
#   scripts/run-host-kv-verify.sh --artifact <model.ninfer> [--port 8099] [--mib 4096]
#
# The verification needs exclusive GPUs and a service with host-KV debug logging, a 4096 MiB host
# cache, concurrency 2 and MTP drafts, so this wrapper stops ai-server, starts its own ninfer-serve
# with those settings, runs `scripts/verify_host_kv_serve.py` against it, and restarts ai-server.
# The restart runs from an EXIT trap that TERM/INT/HUP also route through, so the service comes
# back even when the run fails, times out, or is killed.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact="${NINFER_V100X2_ARTIFACT:-}"
port=8099
mib=4096
serve_bin="$repo_dir/build-host-retained/apps/ninfer-serve"

while [ $# -gt 0 ]; do
  case "$1" in
    --artifact) artifact="$2"; shift 2 ;;
    --port)     port="$2"; shift 2 ;;
    --mib)      mib="$2"; shift 2 ;;
    -h|--help)  sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$artifact" ] || { echo "--artifact (or NINFER_V100X2_ARTIFACT) is required" >&2; exit 2; }
[ -x "$serve_bin" ] || { echo "missing $serve_bin; build the ninfer-serve target" >&2; exit 2; }
[ -f "$artifact" ] || { echo "no such artifact: $artifact" >&2; exit 2; }

work_dir="$(mktemp -d /tmp/ninfer-host-kv-verify.XXXXXX)"
serve_pid=""
cleaned=0
cleanup() {
  [ "$cleaned" = 1 ] && return 0
  cleaned=1
  if [ -n "$serve_pid" ] && kill -0 "$serve_pid" 2>/dev/null; then
    kill -TERM "$serve_pid" 2>/dev/null || true
    wait "$serve_pid" 2>/dev/null || true
  fi
  echo "[host-kv-verify] restarting ai-server"
  systemctl start ai-server || true
}
trap 'exit 143' TERM INT HUP
trap cleanup EXIT

echo "[host-kv-verify] stopping ai-server for exclusive GPU use"
systemctl stop ai-server

# The verification anchors its window on this epoch, which is before model load and warmup.
since="$(date +%s)"
reqlog="$work_dir/requests.jsonl"
NINFER_HOST_KV_DEBUG=1 "$serve_bin" "$artifact" \
  --tp 2 --devices 1,0 --host 127.0.0.1 --port "$port" \
  --max-concurrency 2 --max-context 160000 --kv-capacity 280000 --kv-dtype int8 \
  --prefill-chunk 2560 --spec mtp --draft-tokens 3 --lm-head-draft \
  --context-host-cache-mib "$mib" --request-log-jsonl "$reqlog" \
  > "$work_dir/serve.log" 2>&1 &
serve_pid=$!

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$serve_pid" 2>/dev/null; then break; fi
  if curl -sf --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  echo "[host-kv-verify] private instance never became healthy; last log lines:" >&2
  tail -20 "$work_dir/serve.log" >&2
  exit 1
fi
echo "[host-kv-verify] private instance healthy on 127.0.0.1:$port (mib=$mib, log=$work_dir/serve.log)"

set +e
NINFER_HOST_KV_VERIFY_MANAGED=1 \
NINFER_HOST_KV_VERIFY_SINCE="$since" \
NINFER_HOST_KV_REQLOG="$reqlog" \
NINFER_HOST_KV_BASE="http://127.0.0.1:$port/v1/chat/completions" \
NINFER_HOST_KV_MODELS="http://127.0.0.1:$port/v1/models" \
  python3 "$repo_dir/scripts/verify_host_kv_serve.py"
status=$?
set -e

echo "[host-kv-verify] python exit=$status"
exit "$status"
