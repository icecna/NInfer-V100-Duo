# Context cache port plan (target: upstream-parity features on V100 / TP2)

状态：active port plan。实现完成后并入 `concurrent-inference-architecture.md` 与
`paged-kv-cache.md`，本文件随后删除。

Route decision: implement the upstream context-cache **functionality** locally rather than porting
upstream's code structure. The pinned upstream revision, the feature-by-feature alignment, and the
deviation register live in [`context-cache-parity.md`](context-cache-parity.md); that file is the
acceptance basis and must be updated with any new deviation.

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
  - L5e `PrefixIndex` shortlist integration — done: `feat(runtime): shortlist host prefixes by
    content digest`. The hosted residency index is keyed by (identity_tag, frontier, digest); a
    residency's digest table is recomputed from the ledger's real token ids after generation so
    generated-region frontiers stay matchable; the base plan shortlists candidates before the
    authoritative token/media match, preserving the session filter and preference. Real-artifact
    gate re-run passes both phases.
  - L5f retention-policy integration — done: `feat(runtime): make the retention policy the host
    cache authority`. Hosted residencies register with `ContextRetentionPolicy` (owner/session
    lineage, demand recorded on reuse, victims order), which is now the single retention authority;
    the table's session/demand writes are gone. Real-artifact gate re-run passes both phases.
  - L5g multi-checkpoint capture — attempted and reverted. A second residency at the
    rewrite-checkpoint frontier (KV shared with its terminal, owning only the GDN snapshot and
    hidden) reused correctly but failed a new schedule-matched output comparison in the real gate:
    the restored state decoded differently from a cold prefill, so the rewrite-slot snapshot is not
    a verified restore source on TP2. The feature was removed and the gate returned to its
    verified terminal-only behaviour. Reinstating it requires establishing where the frontier GDN
    and hidden can be read for both ranks, and a schedule output for the hidden at an arbitrary
    frontier.
  - L5g diagnosis (reproduced) — the acceptance comparison was re-added and the retry reproduced
    the divergence. Instrumented evidence: the rewrite-checkpoint capture is at the right frontier
    (`[prefill-capture] abs=3274 rel=202 len=202 base=3072`, i.e. the final 202-token chunk ends
    exactly at the prompt frontier), and the prefix residency is published at `frontier=3274` and
    claimed at `boundary=3274`. So the frontier bookkeeping is not the fault. What differs from the
    verified terminal claim is the state source: the terminal reads the lane's CURRENT linear-state
    slot at retain time (after decode) and its phase-1 output matches the cold control, while the
    prefix reads the REWRITE-CHECKPOINT slot, which was written during prefill and is only read
    again at retain, after ~30 decode/MTP rounds. The rewrite slot therefore has to be proven
    stable through decode before it can be a restore source; the existing rewrite-recovery test
    only validates it immediately after capture. Next experiment: snapshot the checkpoint GDN into
    a dedicated per-lane buffer at capture time (or read it before decode starts) and copy that
    buffer at retain, then re-run the comparison. Also observed: filler prompts (frontier 21-24)
    publish prefix residencies; capture should be restricted to real prompt boundaries.
    Additional pointer from the TP2 prefill order: `mtp_prefill_chunk_tp2` runs before
    `copy_slot(linear_state_current_slot_, linear_state_rewrite_checkpoint_slot_)` in the same
    checkpoint chunk (`text_context_impl.h:2233` vs `:2254`), and the GDN updates for both the
    target and the MTP go through `linear_state_current_slot_`. If the draft prefill advances that
    slot beyond the checkpoint, the captured rewrite slot is not the target's state at the frontier
    at all. Either way the fix is the same: capture the frontier state explicitly at the checkpoint
    (before any draft advance) into a dedicated per-lane buffer rather than reading the rewrite
    slot at retain.
    Tested: moving the capture to immediately after the target layers (before the draft prefill) does
    NOT fix the divergence. So the capture point is not the fault either; the rewrite slot simply
    does not hold the target's frontier state when retain reads it, whether because another consumer
    owns or advances it during decode or because it is reused. The fix is therefore to stop relying
    on the rewrite slot: capture the frontier state explicitly into a dedicated per-lane buffer at
    the checkpoint and read that buffer at retain. That needs a new persistent per-lane state
    layout (`layouts.h`/`layouts_impl.h`), a `set_*` on `TextContext`, and the retain path to read
    the buffer instead of `rewrite_checkpoint_state_slot`; the claim needs no change.
    Tested and ruled out: a dedicated third per-lane state slot (`LinearStateSlots::frontier_state_slot`,
    captured with `copy_slot(current, frontier)` right after the target layers and read by retain
    instead of the rewrite slot) still reproduces the divergence. So the restore really does get the
    target's state at the frontier, and the GDN is not the fault. Two more facts pin the cause:
    the two sessions are comparable (a cold round is byte-identical across them), and the restored
    prefix diverges from the cold prefill starting at the SECOND token (index 1), i.e. after the
    first decode step, while the first token (from the frozen tail hidden) matches. What the prefix
    residency has and no other entry has is KV content PAST its frontier: its KV is shared with the
    terminal, so whole pages past the frontier still hold the terminal's tokens, and the sub-page
    tail page keeps the terminal's columns. On the terminal claim the frontier is the last valid
    index, so there is nothing stale to read; on the prefix there is. The fix is therefore exactly
    the column-level coverage route B was chosen for: give the checkpoint an address space that
    commits only `frontier % tokens_per_page` columns of its tail page (B2 shared extent plus owned
    tail), instead of sharing whole terminal pages unmodified. Verifying this cheaply needs a
    prefix whose last page is either frontier-exclusive or zeroed past the frontier; page-aligning
    the frontier is not enough on its own because the GDN is only captured at the real frontier.
    Retracted hypothesis: a scrub that zeroed those columns in the claimed device KV does change
    phase 1 too, but that does NOT show a read past the valid length. Reading the kernel settles it:
    `gqa_attention_decode_bf16.cuh` bounds its key loop by `window = pos[tokens - 1] + 1`, and the
    launcher's `logical_capacity = envelope.max_visible_keys` is only the upper bound for the
    device-side ACTIVE-SPLIT policy (`gqa_attention_decode.cu:361`, and the file comment at
    `:29-31`: eager calls pass an exact window, graph calls pass their replay interval). Cached
    decode passes `valid_columns = nullptr` because a single-query decode needs no mask; the
    append/prefill path supplies `valid_columns` for its `remaining = valid_columns - column_begin`.
    So the reader does stop at the position, and neither narrowing `logical_capacity` nor adding a
    decode mask would be a fix (the former would break graph replay: the guard
    `last_pos >= logical_capacity` would `write_neutral`). The scrub result must have perturbed
    something else, so it is not evidence.
    KV is now also excluded. The gate runs `options.kv_cache = KvCacheStorage::Int8Group64`, and the
decode kernel's comment and code show the scale is per `(row, 64-group)`, i.e. the group runs along
the HEAD dimension, not along tokens (`gqa_attention_decode_i8.cuh:7`, `:134-135`, `:268`). A
token's quantised record therefore cannot be affected by later tokens in its page, so the prefix's
valid-range KV is bit-identical to what a cold prefill writes, and a private prefix copy would not
change anything. With the attention window bounded by `pos` and the frontier GDN captured
faithfully, the remaining suspects are the speculative (MTP) path and any per-request field the
claim does not restore. Discriminators tried since then, all of which change the next steps:
(1) MTP cannot be disabled for this comparison -- the engine refuses the host retained-prefix cache
without `--spec mtp` ("the non-speculative reuse route has no qualified path"), so the divergence
takes place inside the MTP-path by construction. (2) The divergence is not artifact specific: it
reproduces byte-for-byte on `Qwen3.8-27B-nvfp4.ninfer`, and both artifacts produce identical
CONTROL outputs while the wrong host value differs per artifact (1472 on QUASAR, 63 on nvfp4),
which means the restored state is wrong rather than shifted. (3) A claim-field print shows the
prefix entry lands with `exec == ledger.size()` while the terminal entry lands with
`exec + 1 == ledger.size()`, but testing the "boundary is an index" reading (`reuse_boundary =
prefix_frontier - 1`) moves the divergence to token 0, i.e. the count reading is the correct one and
this asymmetry is not the cause. (4) The terminal claims were never compared before; comparing them
now shows the claim path itself drifts from a cold replay: `A_next` first differs at token 26,
`C_next` at 12, `B_next` not at all (32/32), while the prefix claim differs at token 1. So there is
an accepted small claim-path difference (the reason the original test deliberately does not compare
claim outputs across sessions) whose noise floor is tokens 12-32, and the prefix divergence at
token 1 is far below that floor -- it is a real, much larger error rather than that drift. Any
future acceptance for the checkpoint must therefore compare against a claim-path baseline (for
example the same prompt claimed at the terminal frontier, or a cold recompute inside the host
session), not against the control session. (5) That in-session baseline was then run: the host
session's own cold prefill of A's prompt versus the prefix claim of the same prompt (both single
requests in one session, identical schedule) differ at token 1. The restore is therefore
DETERMINISTICALLY wrong, not drift, and the acceptance for any future fix is that in-session
comparison. Remaining unexplored surface: which restored input the first verify round disagrees on
-- with the state captured in its own frontier slot the candidates are the claimed KV (coverage
and table length) and the peer rank's state, and upstream's answer is to fork both the state image
and the KV address space so that no claim-time field list or whole-page copy exists at all.
  - L5g upstream mechanism (read at `Neroued/ninfer@68c54356`, `src/models/qwen3_5/program/storage/`).
    A checkpoint is a FORK of two things, and the decode addresses them directly:
    `sequence.state` is a `{read, write}` pair of state-image handles; `state_selectors()` maps them
    to physical slots (`sequence.cpp:62`), `settle_state_fork()` commits a fork
    (`sequence.cpp:72`), and `refresh_state_views()` points `sequence.tail_hidden` at the read
    slot's continuation hidden (`sequence.cpp:67`). Prefill writes the checkpoint's state into
    `selectors.destination` and takes the capture hidden from that slot's continuation hidden
    (`prefill.cpp:500`). KV is a `kv_address_space` handle with committed columns
    (`text_kv_addresses->active(...)`, `sequence.cpp:93-101`). The consequence for this port is
    architectural: upstream has no "claim restores a list of fields" step at all -- the decode reads
    the checkpoint's own state slot and its own KV address space, so coverage ends at the frontier
    and no reader can see past it. The local design instead copies the terminal's host KV into the
    claiming lane and re-derives the sequence fields, which is why the state image (B4), the
    boundary-free restore path, and the committed-column coverage (B2) are the parts that still
    have to be built. Note that the local attention window is exact in all three readers (decode
    `pos+1`, prefill `base_i + len`, MTP verify `frontier + extent + 1`), so the earlier scrub
    result is NOT evidence of a page-granular read and may have been a bug in the scrub itself;
    upstream's coverage matters because it removes the possibility, not because a local reader is
    known to run past the frontier.
  - L5h cost-model ranking — pending.

### 4.2 Route B implementation plan (chosen)

The frontier checkpoint is rebuilt on upstream's model instead of the claim-time field list. The
acceptance for every slice is the same three checks: the in-session `cold prefill vs claim`
comparison for the same prompt (must report no divergence), phase 1 + phase 2 of
`ninfer_qwen3_8_27b_v100-duo_host_kv_real_test`, and the real-artifact gate on both artifacts.

B4 first, because it is the structural difference that makes the local restore a field list:

ROOT CAUSE LOCATED: the checkpoint's GDN state is not the state a cold prefill reaches at the same
frontier. A calibrated probe (hash of the per-layer RECURRENT slots of the claimed lane, taken in
the first decode round, `NINFER_HOST_KV_PROBE=1`) gives the same value for two cold prefills of the
same prompt (`8a3e90281c629cf2`) and a different value for the claim of that prompt
(`6e0eee7c2d52bd3b`), while the same round's ingress fields are byte-identical
(`frontier=3274 text_valid=3274 mtp_valid=3274 ledger=3275 drafts=3 extent=3`). So neither KV
coverage, KV content, nor ingress is at fault: the state captured at the capture point, or the
slot it is restored into, is wrong. Next: log the slot indices at the capture
(`linear_state_current_slot_`, `linear_state_frontier_slot_`, `state_.slot_count()`) and hash the
state at the capture moment, to decide between a wrong capture point and a wrong slot mapping.

DONE, and it narrows the defect further. The capture point is CORRECT: with
`NINFER_HOST_KV_PROBE=1` the capture prints `current=0 frontier=2 slots=3` (the mapping
`lane`/`2n+lane`/`3n` is right) and its per-layer recurrent hash is `8a3e90281c629cf2`, which is
exactly what the first decode round of a cold prefill of the same prompt reports -- twice. The
claim of that same prompt reports `6e0eee7c2d52bd3b`. Capture, slot mapping, ingress, KV coverage
and KV content are therefore all correct, and the defect is in the STATE TRANSPORT between the
capture and the claim: either the frontier slot is overwritten between the capture and the retain
snapshot, or `snapshot_linear_state_slot_to_host` / `restore_linear_state_slot_from_host` lose or
misplace it. The next measurement is the same per-layer recurrent hash taken (a) in
`retain_host_sequence` right where the prefix GDN is snapshotted and (b) after the claim restores
it; (a) != `8a3e...` means the slot was clobbered, (a) == `8a3e...` with (b) != `8a3e...` means the
transport is wrong. Note: snapshotting a raw slot from inside `TextContext` faults with
`cudaErrorIllegalAddress`, so the recurrent view (used by the working probe) is the safe sampler.

DONE, and it pins the defect to the CLAIM's restore. The same per-layer recurrent hash taken in
`retain_host_sequence` right where the prefix GDN is snapshotted reports `8a3e90281c629cf2 slot=2`
-- the capture's value, unchanged, so the frontier slot is NOT clobbered between the capture and
the retain and the snapshot reads the right data. The claim of that entry then decodes with
`6e0eee7c2d52bd3b`. Everything before the restore is therefore correct, and what remains is either
the claim's `restore_linear_state_slot_from_host` call (destination slot, byte count, or stream) or
the identity of `payload.gdn_local` for the checkpoint entry. One measurement left: hash the same
recurrent view immediately after the claim restores and before any round runs -- `8a3e...` there
means something advances the state after the restore, anything else means the restore itself.

DONE, and the restore itself is at fault. The claim prints
`claim-restored-hash=6e0eee7c2d52bd3b slot=0 boundary=3274` immediately after its restore, and the
first decode round reports the same value, so nothing advances the state afterwards. The device
source of the snapshot is verified correct (`8a3e...` at the capture and again at the retain), and
the restore is a symmetric `transfer_slot` helper the working terminal claim uses. What is left
unverified is the checkpoint's HOST extent: the bytes the snapshot wrote into the arena are not the
bytes the claim reads back, i.e. the checkpoint's GDN host allocation is aliased, moved, or
re-issued in between. That matches the shape of the defect -- the terminal entry works while the
checkpoint entry does not, and the checkpoint is published first and reclaimed first (its retention
age precedes the terminal's by construction). Next: hash the checkpoint's host extent right after
the snapshot and again right before the restore, and audit the arena extent ownership for a
residency that is published before its terminal (the run needs the checkpoint to coexist with the
terminal and with a third entry).

CORRECTION: the host extent is NOT corrupt. Its raw bytes hash identically right after the snapshot
and right before the restore (`retain-host-hash=ec183aba1fe84574` ==
`claim-host-hash=ec183aba1fe84574`), so the arena keeps the checkpoint's bytes intact. The earlier
state conclusions are also NOT trustworthy, because the probe they used is biased:
`linear_state_transfer.cu` documents that the slot is the FASTEST axis of each layer tensor
("element i of slot s lives at `i * slot_count + s`", `slot_stride = pool.slot_count()`), while the
probe copied `recurrent_slot(layer, slot).bytes()` linearly from `.data`. A strided slot view read
as if contiguous mixes neighbouring slots, and the capture probe sampled slot `2n+lane` while the
claim probe sampled slot `lane`, so the two numbers were never comparable. What remains established:
capture-time slot indices are correct (`current=0 frontier=2 slots=3`), the host extent round trips
intact, ingress fields are identical, KV coverage and KV content are excluded. The state question is
OPEN again. The correct sampler is `snapshot_linear_state_slot_to_host` (it honours the stride) for
BOTH sides, or an explicit strided read; the earlier text-context fault when calling it was a
host-buffer/device-context problem, not a reason to avoid it.

1. **B4a state image.** Give every lane that can own a checkpoint its own state-image record
   (`{read, write}` slot pair) in the existing linear-state pool; the pool already reserves the
   third slot group introduced for the frontier snapshot, so no new allocation shape is needed.
   `state_selectors()` (`program_impl`) returns the pair, `settle_state_fork()` commits it, and
   `refresh_state_views()` rebinds `sequence.tail_hidden` to the read slot. Files:
   `program.h`, `program_impl.h`, `linear_state_slots.h`.
2. **B4b decode reads the image.** The compact/MTP decode takes the row's state slot from the
   selector instead of `current_state_slot(lane)`, so a claim needs no GDN copy at all: it only
   points the lane's `{read, write}` at the checkpoint's image and rebuilds the KV address space.
   This removes the `snapshot_linear_state_slot_to_host`/`restore_linear_state_slot_from_host`
   round trip for checkpoints. Files: `program_impl.h`, `schedule.h`, `mtp_impl.h`,
   `text_context{,_impl}.h`, `text_prefill_impl.h`.

B2 second, to make coverage structural rather than a safety margin:

NOTE: B2 is NOT required for the current defect. A scoped experiment that cleared exactly the tail
page's columns past the frontier for checkpoint entries only (terminal entries untouched) left the
in-session cold-vs-claim comparison completely unchanged (`first diff at 1`). Coverage past the
frontier therefore does not reach the decode path, and the divergence stays in B4. B2 remains in
the plan as the structural guard upstream has, but it is not the fix and must not be started
before B4 is finished.

3. **B2a committed columns in the KV address space.** `KvAddressSpace` (already built, B1) becomes
   the authority for a sequence's coverage: whole pages are shared by reference, the tail page is
   owned and commits only `frontier % tokens_per_page` columns, and the PagedKV pool pinning added
   in this branch keeps a referenced page from being re-issued. Files:
   `kv_page_registry.{h,cpp}`, `program.h`, `program_impl.h`.
4. **B2b readers use the address space.** The claimed row's KV view/table length comes from the
   address space's committed columns, so no reader can address past the frontier regardless of
   which kernel reads it. Files: `program_impl.h`, `schedule.h`.

B5 last, and only once B4+B2 pass:

5. **B5 capture at the frontier.** Re-enable the checkpoint capture (published at the prompt
   frontier, not the terminal) and the multi-turn acceptance; the current capture code, the
   in-session comparison, and the upstream mechanism notes above are the starting material.

While B is in flight the feature stays off: the branch keeps the verified terminal-only behaviour
and a green gate, so any slice can stop without leaving a half-wired path in use.
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
