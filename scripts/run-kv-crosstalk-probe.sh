#!/usr/bin/env bash
# KV/GDN 串台复现 probe（一次性诊断，非回归 gate）
#
# 目标：复现 2026-10-08 生产串台（一个会话的模型上下文里出现另一个会话的内容）。
#
# 怀疑路径：
#   retain() 按 sequence.claimed_entry 共享页面（program_impl.h:1705-1740）
#   claimed_entry 只在 clear_lane() 复位（program_impl.h:2191）
#   start_prefill_lane 的 FullReset 分支不复位（program_impl.h:754-762）
#   executor 第二轮 admission 可把 reuse==0 的新请求原地 FullReset 到 retained lane，
#   且不对该 lane 调 evict_retained_lane（concurrent_executor.h:889-915）
#   => 新请求 retain 时共享了上一条请求/别的会话的 entry 页面。
#
# 场景（并发 worker，制造 lane0/lane1 真正的交错）：
#   worker A / worker B：各自多轮长会话，输出足够长以形成 decode 重叠（占满两个 lane）
#   injector：持续注入全新短会话（reuse==0），落进 retained lane 触发 FullReset
#   判定：任一全新会话/任一 worker 的验证轮里出现别的会话独有的代号 => 串台
#
# 安全：stop ai-server -> 测试 -> EXIT trap 保证 start ai-server。
set -uo pipefail

REPO=/mnt/models/NInfer-V100-Duo
SERVE=$REPO/build-v100-duo/apps/ninfer-serve
MODEL=/mnt/models/ninfer-models/Qwen3.8-27B-QUASAR-NVFP4-NInfer.ninfer
PORT=${PORT:-18080}
LOG=${LOG:-/tmp/kv-crosstalk-engine.log}
PY=${PY:-/tmp/kv-crosstalk-test.py}
GRACE=${GRACE:-15}
TRIALS=${TRIALS:-1}
ROUNDS=${ROUNDS:-3}
GEN=${GEN:-192}
INJECT=${INJECT:-4}
WORKERS=${WORKERS:-2}
HOST_CACHE_MIB=${HOST_CACHE_MIB:-20480}
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

echo "== 宽限 ${GRACE}s（让正在进行的会话回合结束）=="
sleep "$GRACE"

echo "== stop ai-server =="
service ai-server stop >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$n" = "0" ] && break
  sleep 2
done

echo "== start engine with probe (port $PORT) =="
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

echo "== wait for readiness =="
ok=""
for _ in $(seq 1 450); do
  if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "engine died during startup; last log lines:"; tail -30 "$LOG"; exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ok=1; break; fi
  sleep 2
done
[ -n "$ok" ] || { echo "engine not ready in 900s; last log lines:"; tail -30 "$LOG"; exit 1; }
echo "engine ready"

cat > "$PY" <<PYEOF
import json, os, sys, time, threading, urllib.request

PORT = ${PORT}
TRIALS = ${TRIALS}
ROUNDS = ${ROUNDS}
GEN = ${GEN}
INJECT = ${INJECT}
WORKERS = ${WORKERS}
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODELS = f"http://127.0.0.1:{PORT}/v1/models"

with urllib.request.urlopen(MODELS, timeout=30) as r:
    MODEL = json.loads(r.read())["data"][0]["id"]
print(f"model id: {MODEL}", flush=True)

_lock = threading.Lock()
def log(*a):
    with _lock:
        print(*a, flush=True)

def chat(messages, max_tokens=64, timeout=900, no_think=True):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": False}
    if no_think:
        body["reasoning_effort"] = "none"
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    ch = out.get("choices", [{}])[0]
    msg = ch.get("message", {}) or {}
    return {"content": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "finish": ch.get("finish_reason"), "dt": time.time() - t0}

def filler(seed, n=40):
    return "\n".join(
        f"模块{i:03d}(seed={seed}): 负责请求调度与资源预留, 不变量是 frontier 单调递增、"
        f"page 归属唯一; 风险窗口在并发 retain 的 host 拷贝与 device 更新之间。" for i in range(n))

def worker(name, code, rounds, gen):
    """长会话：多轮、每轮长输出，制造 decode 重叠并反复命中自己的 host 前缀。"""
    sysp = f"你是资深 C++/CUDA 代码审查助手。本项目代号 {code}。回答用中文，尽量详细。"
    msgs = [{"role": "system", "content": sysp},
            {"role": "user", "content": f"请记住项目代号 {code}。下面开始逐轮审查，尽量详细。"
                                        f"\n初始清单：\n" + filler(name, 50)}]
    try:
        for i in range(rounds):
            msgs.append({"role": "user", "content":
                         f"[第{i+1}轮] 继续深入讨论 seed={i}，详细展开至少 150 字。"})
            r = chat(msgs, gen)
            msgs.append({"role": "assistant", "content": r["content"] or r["reasoning"]})
            log(f"[{name}{i+1}] dt={r['dt']:.1f}s finish={r['finish']} len={len(r['content'])}")
        msgs.append({"role": "user", "content":
                     "只回答一行：你在本对话上下文里见过的所有形如 XXXX-0000 的代号或暗号。没有就回复'无'。"})
        r = chat(msgs, 128)
        verify = (r["content"] + "\n" + r["reasoning"]).strip()
        log(f"[{name}V] {verify[:160]!r}")
        return {"name": name, "code": code, "verify": verify}
    except Exception as e:
        log(f"[{name}] FAIL {e!r}")
        return {"name": name, "code": code, "verify": ""}

def injector(trial, n):
    """全新会话（reuse==0）：应落进 retained lane 并原地 FullReset。"""
    outs = []
    for i in range(n):
        try:
            msgs = [{"role": "system", "content": f"你是通用助手 C{trial}-{i}，回答简短。"},
                    {"role": "user", "content": filler(f"C{trial}-{i}", 8) + "\n以上只是背景，只回复'好的'。"}]
            r = chat(msgs, 16)
            msgs.append({"role": "assistant", "content": r["content"]})
            msgs.append({"role": "user", "content":
                         "只回答一行：你在本对话上下文里见过的所有形如 XXXX-0000 的代号或暗号。没有就回复'无'。"})
            r2 = chat(msgs, 128)
            text = (r2["content"] + "\n" + r2["reasoning"]).strip()
            log(f"[C{trial}-{i}] {text[:160]!r}")
            outs.append(text)
        except Exception as e:
            log(f"[C{trial}-{i}] FAIL {e!r}")
        time.sleep(0.3)
    return outs

leaks = []
for trial in range(TRIALS):
    codes = [f"CODE{i}-{5000 + i * 13 + trial}" for i in range(WORKERS)]
    log(f"\n########## trial {trial} codes={codes} ##########")
    res = {}
    inj = []
    ths = []
    for i, code in enumerate(codes):
        name = f"W{i}"
        ths.append(threading.Thread(
            target=lambda n=name, c=code: res.__setitem__(n, worker(n, c, ROUNDS, GEN))))
    ths.append(threading.Thread(target=lambda: inj.extend(injector(trial, INJECT))))
    for t in ths: t.start()
    for t in ths: t.join()

    for who, r in res.items():
        for code in codes:
            if code != r["code"] and code in r["verify"]:
                leaks.append((trial, f"{who} 验证轮", code))
    for i, text in enumerate(inj):
        for code in codes:
            if code in text:
                leaks.append((trial, f"新会话 C{trial}-{i}", code))

print("\n===== 最终判定 =====")
if leaks:
    print("*** 复现串台 ***")
    for x in leaks: print("  泄漏:", x)
    sys.exit(10)
print("本次未复现")
sys.exit(0)
PYEOF

echo "== run crosstalk probe (trials=$TRIALS rounds=$ROUNDS gen=$GEN inject=$INJECT) =="
python3 "$PY"
rc=$?

echo
echo "== probe 日志摘要 =="
grep -c "host-kv" "$LOG" 2>/dev/null | xargs echo "host-kv 行数:"
echo "--- lane 使用 / running 峰值 ---"
grep -oE "lane=[0-9]+" "$LOG" | sort | uniq -c
grep -oE "running=[0-9]+" "$LOG" | sort | uniq -c
echo "--- capture / claim / retain / evict 序列 ---"
grep -E "prefix-capture|claim lane=|claim-host-hash|retain lane=|retain-host-hash|retained-evict|evict entry|rewrite-plan" "$LOG" \
  | sed -E 's/^\[[^]]*\] //' | tail -120
echo
if [ $rc -eq 10 ]; then
  echo "== 结果: 复现串台 =="
else
  echo "== 结果: 本次未复现 (rc=$rc) =="
fi
echo "完整日志: $LOG"
