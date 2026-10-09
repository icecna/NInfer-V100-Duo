#!/usr/bin/env bash
# 重放 2026-10-08 两个真实 pi 会话的请求，复现 KV 串台（方案 C）。
# 安全：stop ai-server -> 独立引擎(18080) -> 重放 -> EXIT trap 恢复 ai-server。
set -uo pipefail

REPO=/mnt/models/NInfer-V100-Duo
SERVE=$REPO/build-v100-duo/apps/ninfer-serve
MODEL=/mnt/models/ninfer-models/Qwen3.8-27B-QUASAR-NVFP4-NInfer.ninfer
PORT=${PORT:-18080}
LOG=${LOG:-/tmp/kv-crosstalk-replay-engine.log}
GRACE=${GRACE:-15}
HOST_CACHE_MIB=${HOST_CACHE_MIB:-20480}
START=${START:-}
END=${END:-}
LIMIT=${LIMIT:-0}
MAX_TOKENS=${MAX_TOKENS:-32}
RUN_ID=$(date +%Y%m%d-%H%M%S)

cleanup() {
  local rc=$?
  if [ -n "${ENGINE_PID:-}" ]; then
    kill "$ENGINE_PID" 2>/dev/null
    wait "$ENGINE_PID" 2>/dev/null
  fi
  service ai-server start >/dev/null 2>&1
  echo "== engine log: $LOG (run $RUN_ID) =="
  echo "== ai-server 已恢复 =="
  exit $rc
}
trap cleanup EXIT

echo "== 宽限 ${GRACE}s =="
sleep "$GRACE"
echo "== stop ai-server =="
service ai-server stop >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$n" = "0" ] && break
  sleep 2
done

echo "== start engine (port $PORT, host-cache ${HOST_CACHE_MIB}MiB) =="
: > "$LOG"
NINFER_HOST_KV_DEBUG=1 NINFER_HOST_KV_PROBE=1 \
"$SERVE" "$MODEL" \
  --tp 2 --devices 1,0 --host 127.0.0.1 --port "$PORT" \
  --max-concurrency 2 --max-context 160000 --kv-capacity 280000 --kv-dtype int8 \
  --prefill-chunk 2048 --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --vision-max-tokens 8192 --context-host-cache-mib "$HOST_CACHE_MIB" \
  --pending-timeout-ms 600000 --temperature 0.7 \
  >> "$LOG" 2>&1 &
ENGINE_PID=$!

ok=""
for _ in $(seq 1 450); do
  if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "engine died during startup:"; tail -30 "$LOG"; exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ok=1; break; fi
  sleep 2
done
[ -n "$ok" ] || { echo "engine not ready:"; tail -30 "$LOG"; exit 1; }
echo "engine ready"

ARGS=(--base-url "http://127.0.0.1:$PORT" --max-tokens "$MAX_TOKENS")
[ -n "$START" ] && ARGS+=(--start "$START")
[ -n "$END" ] && ARGS+=(--end "$END")
[ "$LIMIT" != "0" ] && ARGS+=(--limit "$LIMIT")

echo "== replay =="
python3 "$REPO/scripts/replay_kv_crosstalk.py" "${ARGS[@]}"
rc=$?

echo
echo "== probe 事件摘要 =="
grep -oE "lane=[0-9]+" "$LOG" | sort | uniq -c
grep -oE "running=[0-9]+" "$LOG" | sort | uniq -c
grep -cE "^.*retained-evict" "$LOG" | xargs echo "retained-evict:"
grep -cE "claim lane=" "$LOG" | xargs echo "claim:"
grep -cE "retain lane=" "$LOG" | xargs echo "retain:"
echo "--- 尾部 claim/retain ---"
grep -E "claim lane=|claim-host-hash|retain lane=|retain-host-hash|retained-evict|evict entry" "$LOG" | sed -E 's/^\[[^]]*\] //' | tail -50
echo "完整日志: $LOG"
[ $rc -eq 10 ] && echo "== 结果: 复现串台 ==" || echo "== 结果: 未复现 (rc=$rc) =="
