# 设计草案:Host 后备的 retained prefix 层

状态:草案,未开工。完成后删除本文件。

## 1. 目标与非目标

**目标:** retained prefix(已结束请求留下的可复用 `SequenceState`)不再必须驻留 device KV 池。
空闲 lane 的 retained state 可以异步搬到 pinned host RAM 长期保留,新的 admission 命中 host
副本时在 boundary 回传 device,再走现有的 exact-hit / suffix-prefill 路径。

效果(针对 Duo 280k 池的场景):显存 KV 池只服务**正在执行**的序列;已结束的长会话前缀
(100k~150k token)在 RAM 里保留,恢复付一次 PCIe 传输(150k token ≈ 3.6 GiB,PCIe 3.0
约 0.2~0.5s),而不是重算整段 prefill(按 1600 tok/s 约 90s)。

**非目标(本方案不动的东西):**

- 无抢占、one-prefill-owner、boundary-only membership 全部保持;
- 不做 paused-request resume(那是抢占栈的语义,不在范围内);
- 不引入 shared prefix 去重(保留当前 single-owner retained 语义);
- 不改变 prefix identity / rewrite checkpoint 的数学;
- 不做 batched prefill。

## 2. 现状(移植基础)

- `SequenceState::retained`(`src/targets/qwen3_6/impl/runtime/program.h`)在请求 terminal 时
  置位,state 留在原 physical lane(lane-affine),device KV pages 被占用直到被
  `evict_retained_lane` 释放。
- admission 可行性失败时,executor 在同一 boundary 驱逐可驱逐的 retained lane
  (`concurrent_executor.h:837`)。
- KV 物理层:`PagedKVPool/PagedKVAllocation`(`src/core/paged_kv_cache.h`),device 侧
  `materialize_pages` 从 pool 取页并发布 block-table 行;每个序列有 text KV 和 MTP backend KV
  两个 pool,TP2 下每卡各有一份(peer 侧 `text_peer`/`backend_peer`)。
- 历史参考:上游 8 月版 `src/core/host_kv_arena.{h,cpp}`(本树 `7028d96a~1` 里,939 行)提供
  pinned host arena、页几何 layout、recipe/apply 原子迁移原语,单 GPU 语义,已在线上验证 6 周。
  本方案复用其设计,按 TP2 需要扩展为 per-device。

## 3. 架构

### 3.1 新增 `HostKVArena`(per-device)

`src/core/host_kv_arena.{h,cpp}`,从 `7028d96a~1` 版本移植并扩展:

```cpp
class HostKVArena {
public:
    HostKVArena(std::size_t capacity_bytes, std::span<const HostKVPageLayout> layouts);
    // 与历史版本相同:allocate / can_allocate / plan_after_releases / apply_recipe / split
    std::size_t occupied_bytes() const;
    std::size_t free_bytes() const;
};
```

- backing 为单块 `PinnedHostBuffer`(host pinned),按 device 分 arena:TP2 下 rank0/rank1
  各一个 arena,各自只映射本卡 stream 的传输目标。每 arena 容量 = 配置总量 / 卡数。
- layout 复用 `KVPageGeometry`:text KV pool 一份、MTP backend KV pool 一份,启动时固定。
- recipe/apply 语义(先原子规划后整体应用)保留,保证驱逐/恢复在 boundary 原子完成。

### 3.2 Residency 状态机(扩展 `SequenceState`)

```
                request terminal
                     │
                     ▼
         retained (device-resident)      ← 现状行为,立即生效
                     │ retain D2H 完成
                     ▼
         retained (host-resident)        ← 新增:device pages 已归还 pool
                     │ admission claim + H2D 完成
                     ▼
              active (device)
```

`SequenceState` 增加:

```cpp
enum class RetainedResidency : uint8_t { None, Device, Host, HostTransferIn };
struct HostKVResidency {
    std::optional<HostKVAllocation> text_kv;      // 每 arena 一份;TP2 存两份(per-rank)
    std::optional<HostKVAllocation> backend_kv;
    std::uint32_t text_pages = 0, backend_pages = 0;
};
RetainedResidency retained_residency = RetainedResidency::None;
HostKVResidency host_kv;
```

约束:

- `retained == false` 时 `retained_residency == None`,`host_kv` 无效;
- `RetainedResidency::Host` 时 device 侧 `text_kv_valid`/`mtp_kv_valid` 对应的 pages 已归还
  (logical extent 记在 `host_kv`,block-table 行保持 installed 但 pages 已释放——
  需要 `PagedKVAllocation` 支持 unbind pages 而保留 row,或 claim 时重新 bind);
- exact-hit 跳过 suffix prefill 的条件从"device 驻留"改为"本 boundary 后 device 驻留":
  host-resident 的 exact-hit 必须先在 claim 时完成 H2D,之后语义与现状一致。

### 3.3 传输路径

**Retain(D2H)** — 触发点:请求 terminal commit(`program_impl.h:1050` 置 `retained=true`
之后,同一 boundary):

1. 在 per-device arena 按 `text_kv_valid`/`mtp_kv_valid` 页数分配 host allocation(可能失败);
2. 成功:逐页 `cudaMemcpyAsync` D2H(text KV 和 backend KV 各自 layout 的 page stride),
   每卡 stream 上提交,`cudaEventRecord` 标记完成;
3. 失败(容量不足):保留现状行为——留在 device-resident,后续 eviction 直接释放;
4. event 完成后(下一个 boundary 检查)归还 device pages、`trim` 到 0,置 `Host`。
   传输中状态对 admission 视为 device-resident(保守,不提前计 host 收益)。

**Claim(H2D)** — 触发点:admission 命中 retained lane 且该 lane 为 `Host`:

1. 分配 device pages(与现状 `materialize_sequence_kv` 相同的 entitlement 检查);
2. 逐页 H2D,双卡各自 stream 提交,两个 event 都 complete 后才把 lane 视为
   device-resident,继续走现有 suffix prefill / exact-hit finalization;
3. 全部发生在 admission boundary 内,该 boundary 的 executor 串行纪律不变;
   H2D 是同步等待(GPU 此时无该 lane 的 in-flight work)。

**Evict** — 两层:

- device-resident:与现状相同,admission 可行性失败时驱逐(释放 device pages,直接丢弃);
- host-resident:仅当新的 D2H 分配需要容量时,按 LRU(以最后 claim/terminal 时间戳排序)
  驱逐 host 条目;驱逐是纯 host free,无 GPU 参与,任意 boundary 可做。

host 驱逐不会破坏 active request(它只作用于 `retained==true` 的 state),不触碰
"admission 保证完成" 的契约。

## 4. 配置

serve/CLI 新参数:

```
--context-host-cache-bytes N   # host pinned 总量,0 = 关闭功能(默认)
```

- 0:完全保持现状(所有 retain 留 device-resident);
- N:双卡各 N/2。
- 启动日志在现有 KV capacity 行后追加:
  `host-cache=24.00 GiB (12.00 GiB/device) retained_host_slots=max-by-geometry`。
- 推荐值公式:`N ≥ 预期同时保留会话数 × 最大前缀 token 数 × 24.3 KiB/token + 余量`。
  例:8 个 150k 会话 ≈ 8×3.6 GiB = 29 GiB,配 48 GiB 留余量。

## 5. 容量与性能预估(V100-PCIe 基线)

锚点:启动日志 `resolved=280000 tokens runtime=6.42 GiB` → 24.3 KiB/token(两卡合计)。

| 项 | 数值 |
|---|---|
| KV 每 token(两卡合计) | ≈ 24.3 KiB |
| 150k token 序列 host 占用(两卡合计) | ≈ 3.6 GiB(1.8 GiB/卡) |
| D2H retain 成本(PCIe 3.0 x16, ~12 GiB/s,双卡并行) | ≈ 0.15s + 逐页开销,总计 <0.5s |
| H2D claim 成本 | 同量级 |
| 对照:重算 150k prefill(实测 ~1600 tok/s) | ≈ 94s |

即恢复一个 150k 会话:约 0.5s 传输 vs ~94s 重算(约 180 倍),且 retain 的 D2H 发生在
会话结束后的空闲窗口,用户不可见。逐页 memcpy 开销可控:device pool 页连续分配
(`materialize_pages` preferred=last+1),host arena 传输规划合并 contiguous runs。

## 6. 契约与文档更新

- `docs/maintainer/concurrent-inference-architecture.md`:
  - §1.2 Non-goals 不变(pause/resume 仍不做);
  - §2.3 "fixed slots, shared state memory" 增补 host-resident retained 语义;
  - §2.5 one-prefill-owner 不变;
  - "retained state 不搬到另一 lane" 保持,host 后备不改变 lane-affinity(回传仍到原 lane);
  - eviction 语义一节扩展为 device/host 两层。
- `docs/serving.md` / `docs/cli.md` / `--help`:新参数。
- `AGENTS.md`:无变化(产品目标不变,只是 retained state 的物理位置扩展)。

## 7. 构建隔离

正在运行的服务依赖 `build-v100-duo`(以及旧的 `build`),本次全部编译/测试使用独立的新构建目录:

- `cmake -S . -B build-host-retained` / `cmake --build build-host-retained -j`
- 不触碰 `build/`、`build-v100-duo/` 及其产物;
- 验证通过后,由用户决定何时用新产物替换运行中的服务。

## 8. 验证面

按 op-development 契约,本方案不引入新数学,验证以行为/生命周期为主:

1. **单元**:`HostKVArena` 移植版——allocate/split/recipe/apply 原子性、容量边界、
   LRU 驱逐顺序(纯 host,快)。
2. **行为(Engine 级,现有测试扩展)**:
   - retain→D2H→evict-device 后,pool free bytes 恢复到 admit 前水平(现状做不到);
   - claim 回传后,exact-hit 跳过 suffix prefill、suffix-hit 只 prefill 增量,
     输出 token 序列与 device-resident 路径**逐 token 一致**(用固定 seed);
   - host 驱逐后再次到达的请求退化为全量 prefill,输出仍一致;
   - 并发 2:会话 A 结束 retain 到 host 期间,会话 B 的 admission/decode 不受影响;
   - D2H in-flight 时强制驱逐:按保守语义验证不释放 device pages。
3. **TP2 专项**:双卡 egress parity(现有 `check_peer_mtp_egress` 模式)扩展到 host 传输——
   两卡的 host 副本独立但内容同源;kill -9 后重启(无持久化)确认启动路径不受影响。
4. **数值**:不新增 kernel;H2D/D2H 是逐页 memcpy,page layout 与 device pool 几何一致,
   用一次 roundtrip 校验(host→device 读回与源比对)覆盖。
5. **性能**:保留场景下第二个会话的 TTFT 对比(重算 vs 回传),记录到
   `bench/`(现有 MTP decode/context decay 报告模式)。

## 8. 改动清单(预估)

| 文件 | 改动 |
|---|---|
| `src/core/host_kv_arena.{h,cpp}` | 新增(移植 `7028d96a~1` 版本 + per-device 扩展) |
| `src/core/paged_kv_cache.{h,cpp}` | 小幅:pages 归还但保留 logical extent 的接口 |
| `src/targets/qwen3_6/impl/runtime/program.h` | `SequenceState` 增加 residency 字段 |
| `src/targets/qwen3_6/impl/runtime/program_impl.h` | retain/claim/evict 三处 boundary 逻辑 + D2H/H2D 提交 |
| `src/runtime/engine/concurrent_executor.h` | admission 可行性计入 host-resident claim 成本;统计/日志 |
| `src/serve/serve_options.{h,cpp}`、request_log | 新参数 + 启动/运行日志 |
| docs(3 份)+ AGENTS 相关段落 | 契约更新 |
| tests | arena 单测 + engine 行为测试扩展 |

工作量评估:arena 移植 1~2 天,executor/program 集成 3~5 天,TP2 专项与测试 3 天,
共约 1.5~2 周。

## 9. 风险与决策点

1. **exact-hit 的 TTFT 变化**:现状 exact-hit TTFT ≈ 0 suffix;host 化后 exact-hit 付一次
   H2D(150k 前缀 <0.5s,短前缀 <50ms),TTFT 从 ~400ms 变为 ~1s 以内。v1 直接接受;
   若实测不可接受,再做"claim 后回写 device 缓存一份"优化(v2 可选)。
2. **pinned 内存不可用**(容器/虚拟化限制 `cudaHostAlloc` 失败):启动时检测,功能自动
   降级为关闭并告警。
3. **host 传输与 executor 单线程的交互**:claim 的 H2D 同步等待发生在 executor 线程,
   期间其他请求也停,但单次 <0.5s,且是低频事件(会话恢复),C=2 场景可接受;若实测
   不可接受,改为 event 轮询跨 boundary,列为 v2。
4. **与上游 10 月新栈的关系**:上游的 `b9114396` 是另一条更完整的路线(含抢占)。本方案
   独立演进;若将来决定跟上游,本层被其 `host_kv_arena`+reclaim 事务取代,本设计作废。