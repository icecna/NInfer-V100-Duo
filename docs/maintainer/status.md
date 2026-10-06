# 项目状态(V100-Duo)

> 临时文档:记录进行中的工作,host KV cache 功能落地提交后删除本文件。

更新时间:2026-07-18
分支:`port-x2-prefill`

## 已交付(近期主线)

- 双 V100 NVLink 推理(`7028d96a` 起步),主线 NVFP4 性能调优:
  sm70 volta decode kernel、TP2 attention decode、V100 duo prefill、
  sm70 CUTLASS prefill 路由 + prepacked QPN kernel(attn/gdn input projections)
- TP2 vision 支持与可配置 visual budget
- DFlash2 系列优化(control/MLP epilogue 融合、top-k 融合、octet sliding attention)
- nvfp4 支持:quasar-nvfp4 trial artifact、qwen3.8 nvfp4 inventory/binding 派生修复、
  volta nvfp4 swiglu epilogue 融合

## 开发中:host KV cache 系列

**Host 后备的 retained prefix(host KV cache)** —— **P0 已完成**(2026-07-18):

- 验收结果(2026-07-18,`scripts/run-host-kv-gate.sh`):
  - arena 单测修复 2 个测试 bug(exact-fit 临时 allocation 立即析构、main 双重计数),全绿
  - live gate(Qwen3.8-27B QUASAR NVFP4):冷算 3.06s vs host claim 0.090s TTFT(256 MiB 预算),
    device-resident 0.074s;exact 输出/frontier/占用量 PASS
  - 已提交 `f17345c8 feat: retain terminal prefixes in a pinned host KV cache`(22 文件)
- 能力:terminal 请求 paged KV 异步 D2H 到 per-device pinned arena,device 页立即归还池;
  后续 admission 命中时同步 H2D 恢复。serve 选项 `--context-host-cache-mib`(默认 0 关闭);
  放不下的前缀保持 device 驻留。GDN 线性状态不移动,保持 lane 亲和。

- 已结束请求的长前缀 KV 不再强制占用 device KV 池,可异步搬到 pinned host RAM
  长期保留;再次命中时一次同步 H2D 恢复(150k token ≈ 3.6 GiB,PCIe 约 0.2~0.5s),
  替代重算整段 prefill(约 90s)。
- 主要改动:
  - `src/core/host_kv_arena.{h,cpp}`:per-device pinned host arena + 页几何 layout +
    原子迁移原语(复用上游单 GPU 版,按 TP2 扩展为 per-device)
  - `program.{h,impl}`:`SequenceState` 增加 host 驻留状态、host LRU、retain/claim 流程
  - `concurrent_executor.h`:admission boundary 的驱逐 / host 收割逻辑
  - 新 serve 选项 `--context-host-cache-mib`(默认 0 关闭,预算按 tp rank 均分)
  - `include/ninfer/types.h`:`memory_summary` 增加 `host_kv_cache_bytes` /
    `host_kv_cache_occupied_bytes` / `host_kv_retain_complete`
- 测试与验收:
  - `tests/test_host_kv_arena.cpp`(新增)、`tests/test_concurrent_executor.cpp`(扩展)
  - `tests/targets/qwen3_6_27b/test_engine_v100-duo_host_kv_real.cpp`(真实 artifact)
  - `scripts/run-host-kv-gate.sh` + `scripts/verify_host_kv_serve.py`
- **状态:实现完成,待跑验收脚本验证后提交。**

## P1(表核心,已完成 2026-07-18,待提交)

- `HostPrefixTable<Payload>`(纯 CPU):insert/find/find_best(内容匹配 + 最深边界
  平局裁决 + LRU 盖戳)/oldest_evictable(最小非零戳,在途 claim 不可驱逐)/
  evict(回调)/begin-end_claim(防下溢);单测 `test_host_prefix_table.cpp` 6 组全绿
- **接线(行为等价改道)**:arena 分配的所有权从 lane(`host_kv` 4 个 HostKVAllocation)
  移到表 entry(`HostPrefixResidency` payload:kv + origin_lane + reuse_boundary);
  lane 保留控制面(ledger/identity/checkpoint/retained)不变,planner/executor 接口零改动;
  claim 经表查找 + begin/end_claim 守卫,完成后按 lane 语义驱逐 entry 归还 arena;
  LRU 容量驱逐改走 `oldest_evictable()`
- live gate(build-host-prefixes 目录,不影响线上服务二进制)与改道前基线逐字节一致:
  cold 3.06s / append 0.074s / host_append 0.090s / occupied=121786368,exact PASS
- gate 脚本已指向 `build-host-prefixes`

## 规划(草案,未开工)

- **内容寻址的跨 lane host 前缀表**(`docs/maintainer/plan-content-addressed-host-prefixes.md`):
  host 前缀不再绑定 lane,2 lane 服务可驻留 3+ 个长前缀(受 arena 预算约束),
  消除多会话乒乓;device-resident 前缀保持 lane 绑定不变。GDN 线性注意力状态随 entry
  持久化(D2H 快照 + claim 时 H2D 到目标 lane 槽),设计已补入该文档。
  目标已定案:多对话(> lane 数)排队时上下文完整保留、回来免 prefill;预算手工配
  (`--context-host-cache-mib`,默认关闭),会话识别靠 prompt 内容匹配(无会话 ID、无 TTL,
  纯容量 LRU 驱逐),miss/驱逐一律降级为冷算不报错。
  依赖上述 host KV cache 功能落地并验证完成后开工;两份 plan 文档完成后一并删除。