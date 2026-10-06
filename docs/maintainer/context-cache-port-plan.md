# Context cache port plan (target: upstream-parity features on V100 / TP2)

状态：active port plan。实现完成后并入 `concurrent-inference-architecture.md` 与
`paged-kv-cache.md`，本文件随后删除。

## 0. Deliverable, scope, completion

Deliverable: reproduce the observable behaviour of upstream `Neroued/ninfer` context cache on the
local V100-Duo product — explicit and automatic prompt-cache markers, cross-request prefix
checkpoints, host-resident replicas with fork sharing, session lineage, and cost-based
retain/reclaim — adapted to `sm_70`, TP2, and the `qwen3_6/3.8` hybrid GDN + DFlash + MTP targets.

In scope: the four layers below (protocol, frontend, storage, engine) and the public type surface
that carries request cache hints.

Out of scope (hardware/product adaptation, not upstream parity):

- Copying upstream class names or its single-GPU device-pool assumptions. Local storage may be a
  semantics-equivalent re-implementation.
- Any change to Op kernels, artifact formats, or the model-conversion toolchain.
- Multi-GPU beyond the existing TP2 mirror.

Completion conditions: every feature in §1 is implemented and observable; §4 layers' acceptance
criteria pass; the real-artifact gate covers marker-driven reuse across sessions; serving docs and
schema tests describe the accepted protocol fields.

## 1. Upstream feature inventory (the parity contract)

### 1.1 Public request surface

- `kMaximumContextCacheSessionKeyBytes = 256`, `kMaximumExplicitPromptCacheMarkers = 4`.
- `PromptCacheMarkerKind`: `SharedStablePrefix` (others may hit it), `PrivateLongAnchor` (own only).
- `SharedCandidateEvidence` bits: `ExplicitBoundary`, `RequestedAutomatic`, `DefaultAutomatic`,
  `EngineStructural`.
- `PromptCacheMarkerLocation`: `MessageBoundary`, `MessagePartBoundary`,
  `LeadingInstructionBoundary`, `ToolBoundary`.
- `PromptCacheMarker`: `after_message_count` (1-based, boundary after that many messages),
  `leading_instruction_bytes` (within the untrimmed leading System/Developer message),
  `after_tool_count`, `after_message_part_count`.
- `ContextCacheHints`: `session_key`, `markers`, `allow_engine_automatic_shared_prefixes`,
  `update_session_index`.

### 1.2 Protocol layer

- OpenAI: `prompt_cache_key` (string, ≤64) and `safety_identifier` (string, ≤64); `user` (string);
  `prompt_cache_retention` ∈ {`in_memory`, `24h`}; `prompt_cache_options` object
  (`mode` ∈ {`implicit`,`explicit`}, `ttl` = `30m`); `prompt_cache_breakpoint` must be
  `{mode:'explicit'}` else 400 `invalid_cache_breakpoint`. `apply_openai_prompt_cache_policy`
  synthesizes the automatic target (last turn with `tool_calls`, else last turn with content, else
  `tools.back()`), keeps the newest ≤4 explicit boundaries as `SharedStablePrefix +
  ExplicitBoundary`, and disables engine structural candidates.
- Anthropic: block-level `cache_control` must be `{type:'ephemeral', ttl? ∈ {5m,1h}}` else 400
  `invalid_cache_control`; allowed on system parts, user text/image, tool_result, assistant text,
  and the last tool_use (otherwise advisory); rejected on thinking/redacted_thinking; ≥5 distinct
  write slots → 400 `too_many_cache_breakpoints`; TTL conflict → 400 `conflicting_cache_ttl`.
- Responses: `previous_response_id` inherits session lineage; a stored response's `session_key`
  defaults to its `response_id`; `cache_hints.session_key`/`update_session_index` follow
  `store`; `session_key` is validated non-empty ≤256.

### 1.3 Frontend layer

- The renderer resolves each marker to a byte offset only at a *unique* output position, then the
  tokenizer maps it to an *exact* token frontier; a boundary without an exact frontier is dropped.
- Resolution is trim-aware (`unicode trim`), macro/JSON-aware, and media-aware
  (`exact_vision_frontier` never splits a multi-token vision item).
- Structural candidates (`EngineStructural`) when the protocol did not set an automatic policy:
  the last tool boundary and the leading-instruction boundary.
- Input validation: >4 markers, invalid kind/location combinations, `SharedStablePrefix` with
  `evidence=None`, and bad `session_key` are rejected.

### 1.4 Storage layer

- Checkpoint = KV history (`text` + optional `backend` address spaces) + `StateImage`
  (linear-attention conv/recurrent pool, continuation hidden, optional DFlash local cyclic KV) +
  resident identity + `{frontier, backend_frontier, rope_delta, tail_hidden_valid}` + `role` +
  `pins`.
- `StateImageRole` ∈ {`Free`, `ActiveMutable`, `CheckpointImmutable`, `ReservedDestination`};
  `StateReplicaResidency` ∈ {`None`, `DeviceOnly`, `HostOnly`, `Both`}; transfer APIs with RAII
  abort; fork (`begin_fork`/`commit_fork`/`abort_fork`) for shared prefixes.
- Logical KV pages carry `references`/`active_references`/`source_pins`/`writer_references`/
  `commit coverage`/host-replica binding; address spaces carry `committed_frontier` and
  `checkpoint_frontier` with `set_checkpoint_requirement` → `protect_coverage`, and
  `truncate_inactive_prefix` to shrink unreferenced checkpoint prefixes.
- `HostKVExtentStore` owns typed host extents over logical-page memberships, pins pages while
  copying, and can split/release zero-reference runs.
- Releasability needs KV address spaces releasable *and* the state owner releasable; a pinned
  checkpoint is never evictable.

### 1.5 Engine layer

- `PrefixIndex<Model>`: radix tree keyed by `checkpoint_key(handle, frontier)`; `matches` walks
  frontier ≤ `prompt_tokens` and returns deepest-first candidates; the model still performs the
  authoritative exact match.
- `ResourceManager<Model>`: candidate ranking by transfer+prefill cost, then bytes, reused tokens,
  session hint, publication order; `adopt` creates the owner/shared reference; `publish` adds
  checkpoints; `finish` updates the session lineage (one owner per `session_key`);
  `victims`/`admits`/`gain_limit`/`balance_reused` drive reclaim; demand is remembered per
  frontier; host shortage releases redundant host replicas first.
- Cost model: `context_cost` Q32 transfer (batch + operations, or bytes/ns) and prefill
  (chunk + token + attention pair + vision item/patch) presets, resolved generic → builtin →
  override.
- Scheduler holds no cache policy; it only supplies membership, fairness, and victim selection
  (including pausing a resident to snapshot it).

### 1.6 Program contract consumed by the engine

`plan_request`, `inspect_source`, `checkpoint_key`, `checkpoint_footprint`,
`checkpoint_summary/metadata`, `checkpoint_matches`, `checkpoint_recovery_frontier/loss`,
`release_checkpoint`, `host_bytes_released`, `pause_host_bytes`, `release_redundant_host`,
`start_binding`, `plan_reclaim`, `plan_releases`, `start_demote`, `start_pause`, `poll_context`,
`has_context_transaction`, `physical_usage`.

## 2. Local state and gaps

| Upstream | Local now | Verdict |
|---|---|---|
| `PromptCacheMarker`, `ContextCacheHints` | absent | add (L1) |
| OpenAI/Anthropic/Responses cache fields | Responses whitelists then 400s them; Chat/Anthropic ignore them | add (L1) |
| Renderer emits per-boundary byte offsets | single `RewriteCheckpointByteSpec` (`TurnClosure`/`ResponseReplay`) | extend (L2) |
| Tokenizer `encode_with_boundaries` | `Tokenizer::encode` only; single-prefix re-encode for one checkpoint | extend (L2) |
| Rolling shortlist digest + `identity_tag` | `ResidentPrefixIdentity` exact match; no shortlist | add (L3) |
| Checkpoint store with roles/refs/pins | single host prefix residency per terminal (`HostPrefixResidency`) | generalize (L4) |
| Ref-counted logical KV + address-space frontier protection | `PagedKVPool` entitlement+mapped, no refcounts | extend (L4) |
| `HostKVExtentStore` | `HostKVArena` (page + raw extents, no membership/pin semantics) | extend (L4) |
| Generic `StateImageStore` (roles/refs/fork/host) | `LinearAttentionStatePool` with `2*max_concurrency` slots and host transfer | generalize (L4) |
| `PrefixIndex` + `ResourceManager` | none; `concurrent_executor` + `admission_policy` | port/adapt (L5) |
| `context_cost` | none | port (L5) |
| `engine_core` + `scheduler` | `concurrent_executor` lane model | integrate (L5) |

Gaps that must be closed first: G1 multi-boundary rendering, G2 exact token frontier,
G3 identity shortlist + `identity_tag`, G4 KV reference counting + checkpoint protection,
G5 host-extent membership/pin, G6 state-image roles/refs, G7 resource policy, G8 protocol fields.

Dead local leftovers to delete with L0: `src/runtime/engine/shared_capture_planner.h`,
`context_portfolio_value.h`, `resource_search.h` (they include the non-existent
`runtime/engine/context_cost.h` and are not compiled).

## 3. Local target architecture (decisions)

- **D1 identity.** Keep `ResidentPrefixIdentity` as the authoritative exact comparison; extend it
  with the media item digest set (modality, grid, patch spans, media content hash, timestamps).
  Add a rolling 2×u64 shortlist digest per frontier and an `identity_tag` =
  (speculative backend, proposal head, KV storage kind, DFlash geometry, TP degree). The shortlist
  is rank0-owned and identical on both ranks.
- **D2 checkpoints.** `Checkpoint{role, frontier, backend_frontier, kv_refs, state_image, identity,
  pins}`. Roles: `InputReplay`, `LongAnchor`, `SharedPrefix`, `Continuation`. KV stays on the
  existing `PagedKVPool`; add per-page reference/active-reference/source-pin counters, a
  `committed_frontier`, and a `checkpoint_frontier` protection watermark, plus a
  `truncate_inactive_prefix` that only shrinks above the highest live checkpoint.
- **D3 host replicas.** Build an extent store over `HostKVArena`: reserve a contiguous extent for
  an ordered logical-page membership, pin sources while copying, publish a host replica capability,
  and support releasing/splitting zero-reference runs. Non-page state (GDN conv/recurrent, tail
  hidden, DFlash local) uses raw extents in the same arena.
- **D4 state image.** Generalize `LinearAttentionStatePool` from two fixed slots to N logical slots
  with role/ref/pin metadata plus a host pool; include continuation hidden and DFlash local state
  (currently excluded from host snapshots). Host bytes stay 256B-aligned and layer-major.
- **D5 frontend boundaries.** `render()` returns an ordered boundary list
  (`MessageBoundary`/`MessagePartBoundary`/`ToolBoundary`/`LeadingInstructionBoundary`) as byte
  offsets, each valid only when its output position is unique. Token frontiers come from a
  boundary-aware encode (token→byte map) or, until that exists, from per-boundary prefix encodes
  (≤6 boundaries). The automatic target is the end of the last content/tool turn, so the reusable
  boundary sits *before* the assistant `<think>` prologue.
- **D6 protocol.** Add the §1.1/§1.2 surface; `translate.cpp` maps protocol boundaries to
  `PromptCacheMarker`s and forwards `session_key`; `PromptInput` gains `ContextCacheHints`.
- **D7 scheduling.** Keep the lane-based `concurrent_executor`; add a
  `ResourceManager`-equivalent policy layer that the executor drives for
  admit → bind → adopt → publish → finish → reclaim. Structural candidates stay disabled when a
  protocol supplies its own automatic policy.
- **D8 TP2.** Rank0 owns digest/shortlist/index decisions; rank1 mirrors checkpoint and host
  allocation through the existing identical-call-order → identical-page-id invariant. Host arena
  exists per device; budgets are reported as the sum and admission uses the bottleneck device.
- **D9 interim.** The existing content-addressed `HostPrefixTable` (P1/P2) remains the working
  reuse path until L4/L5 land, then is removed rather than kept as a parallel cache.

## 4. Layers and acceptance

- **L0 — baseline.** Delete the dead leftovers; land this plan. Acceptance: build unchanged,
  `git diff --check` clean.
- **L1 — protocol and hints.** Public `PromptCacheMarker`/`ContextCacheHints`; OpenAI/Anthropic/
  Responses parsing with the exact error codes; `translate` mapping; no engine behaviour change.
  Acceptance: schema tests for every accepted/rejected field and boundary synthesis.
- **L2 — frontend boundaries.** Multi-boundary rendering + exact token frontiers; automatic and
  structural candidates; vision/trim/Unicode/BPE regressions. Acceptance: frontend tests asserting
  boundary→token-frontier for each location, including a boundary that must be dropped when the
  frontier is not exact.
- **L3 — identity.** Shortlist digests + `identity_tag`; `prefix_shortlist_key` exposed on the base
  plan. Acceptance: digest equality iff exact identity equal, across speculative backend / KV
  storage / media variants.
- **L4 — checkpoint storage.** Reference-counted KV + checkpoint protection/truncate; host extent
  store with membership/pin/fork; generalized state image with host pool. Acceptance: host-arena
  and state-transfer unit tests (lifetime, fork, reclaim, truncate) plus a real-artifact replay of
  a checkpoint at a message boundary.
- **L5 — engine integration.** `PrefixIndex` + resource policy + cost model wired into the
  executor; session lineage; reclaim/balance; pause-to-snapshot. Acceptance: cross-session reuse on
  the real artifact under TP2, reclaim under a constrained host budget, and deterministic
  `reused_prompt_tokens`/`prefix_reuse_path` reporting.
- **L6 — serving surface.** Retention/session fields end to end, `cached_input_tokens` reporting,
  docs and capability tests. Acceptance: `docs/serving.md` + schema tests + a live HTTP check.

## 4.1 Progress

- L0 baseline — done (plan committed, dead leftovers removed).
- L1 protocol and hints — done: `feat(serve): accept protocol prompt-cache hints`.
- L2 frontend boundaries — done: `feat(frontend): resolve prompt cache markers to token
  frontiers`.
- L3 identity shortlist digests — done: `feat(runtime): add prefix shortlist digests and identity
  tag`.
- L4 checkpoint storage.
  - L4a `KvPageRegistry` — done: `feat(runtime): add the KV page ownership registry`. Per-page
    references, active references, source pins, writer, protected columns, committed columns, host
    replica, plus `KvAddressSpace` frontier/checkpoint-protection/inactive-truncate and page
    sharing.
  - L4b `HostKvExtentStore` over `HostKVArena` — done: `feat(runtime): add the host KV extent
    store`. Ordered page membership, source pin on reserve, host-replica publish with pin release,
    abort, and whole-extent release of unreferenced runs. Partial-run splitting is deferred until a
    checkpoint can truncate the suffix of a shared extent.
  - L4c state image — done: `feat(runtime): add the checkpoint state image store`. A role,
    checkpoint-reference, and transfer-pin registry over logical state slots plus a pinned raw
    host-image pool sized for the GDN conv/recurrent state, the continuation hidden, and an
    optional DFlash local region. Wiring the D2H/H2D transfer of the hidden and DFlash region into
    the program belongs to L5, where the program owns the device slots.
- L5 engine integration.
  - L5a `PrefixIndex` — done: `feat(runtime): add the prefix shortlist index`.
  - L5b cost model — done: `feat(runtime): add the context cost model`.
  - L5c resource policy (retention, reclaim, session lineage) — done:
    `feat(runtime): add the context retention policy`. Owner/session lineage, shared vs private
    candidate visibility, bounded demand LRU, demand-ordered victims, and the gain-above-loss
    admission rule.
  - L5d session/demand retention in the working host-prefix path — done:
    `feat(runtime): tag retained host prefixes with the request lineage`. The host prefix table
    tags an entry with the publishing request's session key, prefers a same-lineage entry when the
    request names one while keeping anonymous entries reusable, records the last request that
    reused an entry, and evicts never-reused entries before demanded ones. Real-artifact gate
    re-run passes.
  - L5e `PrefixIndex` + cost model + multi-checkpoint integration — pending.
- L6 serving surface (retention/session end to end, live verification) — pending.

L1/L2 are observable only through the parsed markers and resolved frontiers; Engine reuse
behaviour, and therefore `docs/serving.md`'s cache description, is unchanged until L5.

## 5. Budgets and risks

- Host bytes: checkpoint KV extent + (GDN conv/recurrent + hidden + DFlash) per checkpoint; count
  both ranks. Budget gate uses the bottleneck device; `--context-host-cache-mib` stays the knob.
- Digest cost: one rolling pass per prefill; vision items add a per-item hash.
- Frontier resolution: prefer a tokenizer token→byte map; fall back to ≤6 prefix encodes.
- Risks: TP2 replica divergence, non-exact frontier silently disabling a boundary, DFlash host
  snapshot growth, and reclaim latency during prefill. Each has a fallback: reject the marker,
  drop DFlash from host replicas, and pause-to-snapshot respectively.
