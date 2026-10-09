#!/usr/bin/env bash
# Closed-loop growth probe: one conversation, every turn waits for the full decode and appends
# the generated assistant message, so the next prompt replays what the model actually produced
# (the shape a real client has). Reports the host cache occupancy per turn to separate a real
# retention leak from the replay artifact where truncated history never matches an entry.
set -uo pipefail

REPO=/mnt/models/NInfer-V100-Duo
SERVE=$REPO/build-v100-duo/apps/ninfer-serve
MODEL=/mnt/models/ninfer-models/Qwen3.8-27B-QUASAR-NVFP4-NInfer.ninfer
PORT=${PORT:-18080}
LOG=${LOG:-/tmp/host-growth-engine.log}
GRACE=${GRACE:-5}
ROUNDS=${ROUNDS:-40}
GEN=${GEN:-96}
THINK=${THINK:-0}
CONVS=${CONVS:-1}
RUN_ID=$(date +%Y%m%d-%H%M%S)

cleanup() {
  local rc=$?
  [ -n "${ENGINE_PID:-}" ] && { kill "$ENGINE_PID" 2>/dev/null; wait "$ENGINE_PID" 2>/dev/null; }
  service ai-server start >/dev/null 2>&1
  echo "== engine log: $LOG (run $RUN_ID) =="
  echo "== ai-server 已恢复 =="
  exit $rc
}
trap cleanup EXIT

sleep "$GRACE"
service ai-server stop >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$n" = "0" ] && break
  sleep 2
done

: > "$LOG"
NINFER_HOST_KV_DEBUG=1 "$SERVE" "$MODEL" \
  --tp 2 --devices 1,0 --host 127.0.0.1 --port "$PORT" \
  --max-concurrency 2 --max-context 160000 --kv-capacity 280000 --kv-dtype int8 \
  --prefill-chunk 2048 --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --vision-max-tokens 8192 --context-host-cache-mib 20480 \
  --pending-timeout-ms 600000 --temperature 0.7 >> "$LOG" 2>&1 &
ENGINE_PID=$!

for _ in $(seq 1 450); do
  kill -0 "$ENGINE_PID" 2>/dev/null || { echo "engine died:"; tail -20 "$LOG"; exit 1; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
echo "engine ready"

python3 - "$PORT" "$ROUNDS" "$GEN" "$THINK" "$CONVS" <<'PY'
import json, sys, urllib.request, threading
port, rounds, gen, think = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
convs = int(sys.argv[5])
BASE = f"http://127.0.0.1:{port}"

def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        txt = r.read().decode()
    out = {}
    for line in txt.splitlines():
        for key in ("host_context_used_bytes", "host_context_peak_bytes", "host_context_entries"):
            if line.startswith(key):
                out[key] = int(float(line.split()[1]))
    return out

def chat(msgs):
    body = {"model": "qwen3.8-27b", "messages": msgs, "max_tokens": gen,
            "stream": False, "temperature": 0.7}
    if not think:
        body["reasoning_effort"] = "none"
    body = json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"].get("content") or ""

lock = threading.Lock()

def worker(wid):
    msgs = [{"role": "system", "content": f"你是助手{wid}，回答控制在三句话以内。"}]
    for i in range(rounds):
        msgs.append({"role": "user",
                     "content": f"[{wid}] 第{i+1}轮：用一个具体例子说明熵池、引用计数和写时复制的区别。"})
        answer = chat(msgs)
        msgs.append({"role": "assistant", "content": answer})
        m = metrics()
        with lock:
            print(f"w{wid} r{i+1:>3} {m.get('host_context_used_bytes',0)/2**30:>7.2f} GiB "
                  f"peak={m.get('host_context_peak_bytes',0)/2**30:>6.2f} "
                  f"entries={m.get('host_context_entries',0)}", flush=True)

threads = [threading.Thread(target=worker, args=(w,)) for w in range(convs)]
for t in threads: t.start()
for t in threads: t.join()
PY
echo "== done =="