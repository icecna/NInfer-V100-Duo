# 设计草案:内容寻址的 host 前缀表(跨 lane 前缀缓存)

状态:草案,未开工。依赖 plan-host-retained-prefix.md 的功能落地并验证完成后开工;两文件完成后删除。

## 1. 目标与非目标

**目标:** host 前缀不再绑定 lane。2 lane 服务可驻留 3+ 个长前缀(受 arena 预算约束,16 GiB 下约 6 个
150k),任意空闲 lane 的 admission 可按内容匹配 claim 任意驻留前缀。消除三会话乒乓:非匹配请求
落在空 lane 时不再销毁驻留前缀,前缀只由全局 LRU 按容量驱逐。

效果:3 个长会话 + 2 lane 时,稳态是"第 3 个排队"而不是"三方轮流冷算";前缀驻留数量由
`--context-host-cache-mib` 单一预算控制,与 lane 数解耦。

**非目标:**

- 不改 lane 数、KV 池布局、调度语义(无抢占、FIFO、one-prefill-owner 全保持);
- **device-resident 前缀保持 lane 绑定**:retain-skip(放不下 host arena)的前缀留在原 lane 的
  device KV 里,只可被同一 lane 复用——这是现状行为,跨 lane 复用 device 前缀需要 D2D 拷贝或
  page-remap,不在本方案;
- 不做 shared prefix 去重(两个会话共享同一段开头仍各存一份)；
- **分支轨迹独立驻留**:客户端从同一分叉点走出多个分支(或回切旧轮次另走)时,每个分支
  轨迹各存一份完整 entry,公共段重复计入 arena 预算,不去重。每个 entry 自包含(含公共段
  的 KV 拷贝),公共 entry 被 LRU 驱逐不影响分支 entry 的复用;预算按"预期同时活跃分支数"
  规划,挤占后表现为复用率下降(性能悬崖),不是正确性问题
- 复用 miss = 冷算,永远不报错:客户端改写/截断历史、不回带 reasoning_content 等不规范行为
  的代价是 prefill 重算,不产生错误输出或失败请求
- 不改 prefix identity / rewrite checkpoint / 拷贝 kernel 的数学——D2H/H2D 仍是同一 per-page
  memcpy,变的只是"哪个前缀"和"H2D 到哪个 lane"。

## 2. 现状(本方案改什么)

全部状态 lane-affine,是本方案的移除对象:

- `SequenceState`(`src/targets/qwen3_6/impl/runtime/program.h:186`):`retained` /
  `retained_residency` / `host_kv`(4 个 HostKVAllocation)/ `host_lru_stamp` /
  `ledger` / `prefix_identity` / `rewrite_checkpoint`
- 匹配:`plan_request_for_lane(lane, ...)` 按 lane 调
  `prefix_matches(prompt, sequence.ledger, sequence.prefix_identity, reuse_base)`
  (`program_impl.h:660-684`);executor `find_admission_lane` 逐 lane 规划,优先可复用 lane,
  非匹配的 retained lane 留给驱逐路径(`concurrent_executor.h:777-800`)
- 生命周期:terminal → `retain_host_sequence` D2H → harvest boundary 的
  `complete_host_transfer` 释放 device 页、lane 变 Host 驻留;非匹配 admission 落该 lane →
  `reset-host reason=full-reset` 销毁前缀;retain 容量不足 → 驱逐最旧 host 驻留 **lane**
- executor 集成:`wait_host_retain(lane)`(admission 前 drain)、`harvest_host_retains()`
  (boundary 收割 + `invalidate_lane_plans`)

## 3. 设计

### 3.1 HostPrefixTable(Program 层新组件)

```
struct HostPrefixEntry {
    std::uint64_t id;
    std::vector<TokenId> tokens;               // 完整前缀(从 lane ledger 移出)
    ResidentPrefixIdentity identity;           // 快速匹配指纹(从 lane 移出)
    HostKVResidency residency;                 // 4 个 HostKVAllocation(从 lane 移出)
    RewriteCheckpoint checkpoint;              // 前缀携带的 checkpoint(复用路径选择需要)
    std::uint32_t reuse_boundary;              // 最深可复用边界(prompt_end-2 / frontier)
    // GDN 线性注意力的递归状态(原 lane 槽在 retain 时刻的快照,local + peer 两个 device 各一份)。
    // 跨 lane claim 时 H2D 到目标 lane 自己的 per-lane 槽;若不持久化,原 lane 被复用后状态即
    // 丢失,线性注意力层无法从中断点继续。快照内容 = retain 时刻 lane 槽的字节(与 device-
    // resident 路径保留的状态同一份数据,rewrite checkpoint 语义不变)。
    GdnStateResidency gdn;                     // 每 device: {conv, recurrent} 两个 pinned extent
    std::uint64_t lru_stamp;                   // 全局 LRU 序,claim 时刷新
    std::uint32_t in_flight_claims = 0;        // 进行中的 H2D 读,>0 不可驱逐
    // 每 entry 的 H2D 完成事件(用于 drain)
};
```

- **插入**:D2H 完成(harvest boundary)时从 lane 状态 **move** 进表;lane 的 `retained` 清除,
  lane 立即可用。每 entry 内存开销 ≈ tokens(600 KB)+ identity(~2 MB)+ GDN 状态(约几 MB),
  相对 GB 级 KV 可忽略
- **查找**:给定 prompt 扫表(N 很小,16 GiB 下 ≤ ~6):`prefix_matches` 直接复用现有 matcher,
  先做长度/指纹预筛再逐 token。选最深可复用边界。**不需要索引结构**
- **驱逐**:全局 LRU(最小 `lru_stamp`),victim 必须 `in_flight_claims == 0`;进行中则 drain 该
  entry 的 H2D 事件(与 `wait_host_retain` 同构)或跳过选次旧。容量检查沿用
  `HostKVArena::can_allocate_all`(已修的顺序模拟);GDN 状态与 KV 共用同一 arena 预算,
  容量检查时计入其 extent
- 硬上限 = arena 容量;是否加显式 entry 数上限见 §7

### 3.2 规划路径改写

- `plan_request_base`(lane 无关)做内容查找:命中 → base plan 携带
  `(entry_id, reuse_base, reuse_path)`;不命中 → FullReset
- `plan_request_for_lane` 退化为"把已选 entry 绑定到指定空 lane"的 claim 计划;host 路径不再
  逐 lane 重算匹配。lane 版本缓存相应简化:内容失效由"被驱逐的 entry id"触发,executor 对
  引用该 entry 的排队请求重规划(替代现有 per-lane `invalidate_lane_plans`)
- device-resident retained lane 的匹配保持现状(按 lane,`prefix_matches` 走 lane 状态)——
  作为内容查找的第二个来源,命中优先级:同 lane device 前缀 > host 表前缀

### 3.3 claim 与 terminal

- **claim**:`claim_host_prefix(entry, lane)`:为 lane 分配 device KV(现状),H2D
  entry 字节 → 新 device 页;GDN 状态 H2D → 目标 lane 的 per-lane 槽(`conv`/`recurrent`
  逐层 extent,TP2 下 peer device 各一份,与 KV 的 local/peer 分支同构)。lane 槽从此是该请求
  的工作副本(后续 `zero_slot`/decode 更新只动 lane 槽),entry 内的快照保持只读——
  转移期间 `entry.in_flight_claims++`,事件完成后 --,多个 lane/请求可先后 claim 同一 entry
- **terminal**:新内容插入新 entry(不复用旧 entry);lane 状态清空。现有
  `complete_host_transfer` 的"替换 lane 上的旧 host_kv"逻辑消失——旧 entry 由 LRU 管理
- **非匹配 admission**:落空 lane 无事发生(没有 `reset-host` 可触发)——乒乓消失

### 3.4 executor 接口变化

| 现有 | 变化 |
|---|---|
| `wait_host_retain(lane)` | 改为 admission 规划前 drain **全部**在途 D2H(数量 ≤ lane 数,耗时 ms 级),保证匹配时表完整 |
| `harvest_host_retains() -> evicted lanes` | `harvest_host_retains() -> {inserted ids, evicted ids}`;evicted ids 触发引用它们的排队请求重规划 |
| `has_retained_lane(lane)` | 语义收窄为 device-resident retained(host 驻留不再是 lane 状态) |
| `memory_summary` / debug 事件 | host 占用 = 表总字节;事件改为 entry 维度(`retain entry=#` / `claim entry=# h2d_ms=` / `evict entry=#`) |

### 3.5 不变量与并发

- 仍全单线程(executor),arena 无锁契约不变,不引入新同步
- entry 驱逐前其 H2D 读必须完成(事件 drain);D2H 写完成才插入(事件 gate 不变)
- arena 的 extent 生命周期:entry 持有 HostKVAllocation,驱逐即释放——与现状一致

## 4. 数值正确性论证

拷贝路径(布局、per-page memcpy、plane 对齐)一字不改,只改选择与目的地。gate 需新增场景:

1. **跨 lane claim**:前缀在 lane 0 产生 → lane 0 被其他请求占用 → 同一 prompt 在 lane 1
   claim,输出与冷算逐字节一致
2. **3 前缀共存**:2 lane 下 A/B/C 各保留一轮,第三个 prompt 仍能 claim(16 GiB 预算或
   gate 用小 arena + 短前缀构造)
3. **LRU 驱逐**:小 arena 强制 A 被 C 驱逐;A 再来 → FullReset 冷算,输出逐字节一致;
   驱逐不得命中 `in_flight_claims > 0` 的 entry
   - **3b. 排队请求引用被驱逐 entry**:请求规划时命中 entry X,随后 X 在 harvest boundary 被
     LRU 驱逐 → 该请求**静默降级到 FullReset**(不抛错、不卡死),冷算输出与直接冷算一致
4. 现有 gate 场景(同 lane 往返)全保留,作为回归
5. **GDN 状态跨 lane 往返**:前缀在 lane 0 retain(含 GDN 快照)→ lane 0 被另一请求占用
   (其 prefill 覆盖 lane 0 的 GDN 槽)→ 原 prompt 在 lane 1 claim,输出与冷算逐字节一致;
   同一 entry 被两个 lane 先后 claim,两次结果一致且 entry 快照不变

## 5. 阶段

- **P0 收尾现有功能**(前置):重跑 live verify(脚本 3 个 bug 已修)、提交 host-retained 全部
  改动。当前代码未测完,不做在未完成的地基上
- **P1 表核心**:`HostPrefixTable` + 单测(插入/查找/LRU/在途保护,纯 CPU,假 layout);
  retain/claim 改道走表但行为等价(单 lane 场景),gate 全绿
- **P2 规划改写**:内容查找进 base plan、跨 lane claim、executor 重规划语义;gate 场景 1-2
- **P3 驱逐与保护**:全局 LRU、在途 drain、事件/summary 改 entry 维度;gate 场景 3-4 +
  三会话 live 压测(bench 会话 + 两个长会话,确认零乒乓) + gate 场景 5
- **P4 文档与观测**:README/serving 的能力描述(前缀数量由预算决定,与 lane 解耦;复用率
  取决于客户端历史管理行为)、观测项加**复用命中率 + 驱逐率 + retain-skip 率**
  (碎片/挤占的性能悬崖可观测),删两份 plan

## 6. 风险

- 规划路径是改动最大的面(planner/executor/Program 三方接口),P1 先做行为等价改道,把
  "改接口"和"改语义"拆开验证
- 排队请求的重规划:entry 驱逐后持有旧 base plan 的请求必须能降级到 FullReset 而不是抛错——
  现有 `invalidate_lane_plans` 已覆盖 per-lane 形态,推广到 entry id 需要把版本戳从 lane 维度
  改成 plan 内容维度
- vision 前缀:identity 含 vision items,entry 原样携带;匹配逻辑不变,但 gate 场景应加一个
  带图会话确认 identity 在 move 后仍等价
- **预算挤占与 arena 碎片**(多分支/长短不一轨迹反复 allocate/release):总剩余够但连续块
  不够时 `retain-skip reason=capacity`,前缀留在 device 池占 KV 页——性能降级不是故障,
  靠 `can_allocate_all` 顺序模拟兜底 + 驱逐率/retain-skip 率观测消化
- **PCIe 竞争**:terminal D2H 与 claim H2D 可同时进行,单 PCIe 3.0 带宽下串行排队,
  TTFT 方差变大;不预先限速,P3 压测看数据再定
- **GDN 跨 lane gate 的构造成本**:需精确控制"原 lane 被占用"时机,live 环境难构造,
  可能需要 test engine 加可控的 lane 抢占钩子

## 7. 开放问题(实现前定)

1. entry 数上限:仅靠 arena 容量,还是加 `--max-host-prefixes`?(倾向不加:预算即上限)
2. 匹配平局(两个 entry 都能复用同一 prompt 的不同深度):取最深边界,确认与现有
   turn-checkpoint 优先语义一致
3. 表在进程关停时的清理:entry 析构归还 arena,需确认无在途 D2H(现有 shutdown 顺序已 drain,
   复核即可)