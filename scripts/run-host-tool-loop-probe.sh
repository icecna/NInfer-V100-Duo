#!/usr/bin/env bash
# Closed-loop tool-loop growth probe: a conversation whose history replays the model's own
# outputs, including tool_calls and tool results (the shape a coding agent produces). Every turn
# waits for the full decode and feeds it back, so the next prompt is a strict continuation.
# Reports host cache occupancy per turn to check that stale checkpoints are reclaimed.
set -uo pipefail

REPO=/mnt/models/NInfer-V100-Duo
SERVE=$REPO/build-v100-duo/apps/ninfer-serve
MODEL=/mnt/models/ninfer-models/Qwen3.8-27B-QUASAR-NVFP4-NInfer.ninfer
PORT=${PORT:-18080}
LOG=${LOG:-/tmp/host-toolloop-engine.log}
GRACE=${GRACE:-5}
ROUNDS=${ROUNDS:-10}
GEN=${GEN:-256}
CONVS=${CONVS:-2}
HOST_CACHE_MIB=${HOST_CACHE_MIB:-12288}
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
NINFER_HOST_KV_DEBUG=1 NINFER_PROMPT_DEBUG=1 "$SERVE" "$MODEL" \
  --tp 2 --devices 1,0 --host 127.0.0.1 --port "$PORT" \
  --max-concurrency 2 --max-context 160000 --kv-capacity 280000 --kv-dtype int8 \
  --prefill-chunk 2048 --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --vision-max-tokens 8192 --context-host-cache-mib "$HOST_CACHE_MIB" \
  --pending-timeout-ms 600000 --temperature 0.7 >> "$LOG" 2>&1 &
ENGINE_PID=$!
for _ in $(seq 1 450); do
  kill -0 "$ENGINE_PID" 2>/dev/null || { echo "engine died:"; tail -20 "$LOG"; exit 1; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
echo "engine ready"

python3 - "$PORT" "$ROUNDS" "$GEN" "$CONVS" <<'PY'
import json, sys, threading, urllib.request
port, rounds, gen, convs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
BASE = f"http://127.0.0.1:{port}"
TOOLS = [{"type": "function", "function": {
    "name": "lookup",
    "description": "Look up a stored value by key.",
    "parameters": {"type": "object",
                   "properties": {"key": {"type": "string"}},
                   "required": ["key"]}}}]

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
    body = json.dumps({"model": "qwen3.8-27b", "messages": msgs, "max_tokens": gen,
                       "stream": False, "temperature": 0.7, "tools": TOOLS,
                       "tool_choice": "auto"}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]

lock = threading.Lock()

def worker(wid):
    msgs = [{"role": "system", "content": f"你是助手{wid}。需要信息时调用 lookup 工具。"}]
    for i in range(rounds):
        msgs.append({"role": "user",
                     "content": f"[{wid}] 第{i+1}轮：先用 lookup 查 key_{wid}_{i}，再用两三句话总结。"})
        # tool loop: keep answering until the model stops asking for tools
        for _ in range(4):
            m = chat(msgs)
            if m.get("tool_calls"):
                msgs.append({"role": "assistant", "content": m.get("content") or "",
                             "tool_calls": m["tool_calls"]})
                for tc in m["tool_calls"]:
                    msgs.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": f"{tc['function']['name']}: value_{wid}_{i}"})
            else:
                msgs.append({"role": "assistant", "content": m.get("content") or ""})
                break
        mm = metrics()
        with lock:
            print(f"w{wid} r{i+1:>3} used={mm.get('host_context_used_bytes',0)/2**30:>6.2f} GiB "
                  f"peak={mm.get('host_context_peak_bytes',0)/2**30:>6.2f} "
                  f"entries={mm.get('host_context_entries',0)} msgs={len(msgs)}", flush=True)

threads = [threading.Thread(target=worker, args=(w,)) for w in range(convs)]
for t in threads: t.start()
for t in threads: t.join()
PY
echo "== done =="