#!/usr/bin/env bash
# Stop the running ai-server (systemd), run the host KV round-trip real-artifact gate, then
# restart the service. The service is down while the test runs; the restart happens even when
# the test fails (trap). Manage the service exclusively via `service ai-server`.
set -u

ROOT=/mnt/models/NInfer-V100-Duo
ARTIFACT=/mnt/models/ninfer-models/Qwen3.8-27B-QUASAR-NVFP4-NInfer.ninfer
TEST_BIN=$ROOT/build-host-retained/tests/ninfer_qwen3_8_27b_v100-duo_host_kv_real_test
LOG=$ROOT/build-host-retained/host-kv-gate.log
SERVE_LOG=/mnt/models/ai-launchers/serve-restart.log
HEALTH_URL=http://192.168.1.12:8080/health

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

wait_gpus_free() {
  local i busy
  for i in $(seq 1 120); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
    if [[ "$busy" == "0" ]]; then log "both GPUs free"; return 0; fi
    sleep 1
  done
  log "ERROR: GPUs still busy after 120s"
  nvidia-smi
  return 1
}

wait_healthy() {
  local i
  for i in $(seq 1 240); do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
      log "ai-server healthy (after ~${i}s)"
      return 0
    fi
    sleep 1
  done
  log "ERROR: ai-server not healthy after 240s; journalctl -u ai-server -n 40"
  journalctl -u ai-server -n 20 --no-pager || true
  return 1
}

stop_serve() {
  service ai-server stop
  # Belt and suspenders: catch any stray unmanaged ninfer-serve still holding the port/GPUs.
  local pids
  pids=$(pgrep -f "NInfer-V100-Duo/build.*/apps/ninfer-serve" || true)
  if [[ -n "$pids" ]]; then
    log "killing unmanaged ninfer-serve pid(s): $pids"
    kill $pids
    sleep 3
    pkill -9 -f "NInfer-V100-Duo/build.*/apps/ninfer-serve" 2>/dev/null || true
  fi
  wait_gpus_free
}

restart_serve() {
  service ai-server start || { log "ERROR: service ai-server start failed"; return 1; }
  wait_healthy
}

trap 'if [[ "${STOPPED:-0}" == "1" ]]; then restart_serve || true; fi' EXIT

stop_serve || { log "ERROR: service stop failed; NOT restarting (GPUs still owned by something)"; exit 1; }
STOPPED=1

cmake --build "$ROOT/build-host-retained" -j 32 --target ninfer_qwen3_8_27b_v100-duo_host_kv_real_test > "$LOG" 2>&1
build_rc=$?
if [[ $build_rc -ne 0 ]]; then
  log "ERROR: test build failed (rc=$build_rc); see $LOG"
  grep -m5 "error:" "$LOG" || true
  exit "$build_rc"
fi

log "running host KV round-trip gate (artifact: $ARTIFACT)"
NINFER_V100_DUO_ARTIFACT="$ARTIFACT" "$TEST_BIN" 2>&1 | tee -a "$LOG"
test_rc=${PIPESTATUS[0]}
exit "$test_rc"