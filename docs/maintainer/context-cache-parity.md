# Context-cache parity contract (route: function-aligned local implementation)

Status: active. This file is the acceptance basis for the context-cache work. It records exactly
what "aligned with upstream" means for this branch, so a change can be judged without re-litigating
the route.

## Pinned upstream revision

- Repository: `https://github.com/Neroued/ninfer`
- Revision: `68c54356fd490ab329bd1475d48957f886bb7dd1` (2026-10-05), subject
  "fix(frontend): preserve cache boundaries through template trimming"

"Aligned" means aligned with this revision only. Upstream still moves; re-pin and re-audit when it
does. No claim is made that this branch can merge upstream changes.

## Route

This branch implements the upstream context-cache **functionality** locally, not its code
structure (see the port plan's route decision). Upstream's storage/engine layers are single-GPU and
built around a reference-counted KV address space and an execution-transaction model that this
branch does not have; this branch's TP2 V100 target has no upstream counterpart. Porting the code
would still leave the TP2 boundary self-designed and would replace the local, real-artifact-verified
execution path with an unverifiable one.

## Alignment by feature

Legend: **aligned** = observable behaviour matches; **approximate** = same intent, different
internals that can change an observable ordering or frontier; **deviates** = intentionally
different; **not implemented**.

| Area | Status | Evidence / note |
|---|---|---|
| OpenAI `prompt_cache_key`/`options`/`breakpoint`/`retention` validation and error codes | aligned | schema tests |
| OpenAI automatic boundary target (last tool-call turn → last content turn → tools) | aligned | schema tests |
| Explicit-boundary budget (keep newest four) | aligned | schema tests |
| Anthropic `cache_control` (ephemeral, `5m`/`1h`, thinking rejection, `>4`, TTL conflict) | aligned | schema tests |
| Responses cache fields accepted + validated | aligned | schema tests |
| Responses session lineage (own id on first store, parent key on continuation) | approximate | local generates the response id per request; no persistence |
| `prompt_cache_retention: "24h"` | deviates | accepted for compatibility; the cache is process-local in memory only |
| Marker types and locations (`MessageBoundary`, `MessagePartBoundary`, `ToolBoundary`, `LeadingInstructionBoundary`) | aligned | cache-boundary test |
| Marker → exact token frontier | approximate | local hand-written ChatML renderer, not upstream's jinja source-range + unique-output-position resolution; local maps offsets directly and drops offsets that are not exact token boundaries |
| Engine structural candidates (last tool boundary, leading instruction) | aligned | cache-boundary test |
| Vision frontier handling | approximate | local shifts/drops offsets through media-placeholder expansion; no upstream `exact_vision_frontier` rule |
| Shortlist digest + `identity_tag` | approximate | same `identity_tag` formula; digest is a local dual-lane FNV over id/type/positions/media, so values differ while equality semantics match |
| Digest from the ledger's real ids after generation | deviates (local-only) | needed because this branch matches generated-region frontiers |
| KV page ownership (references/pins/protection/truncate) | deviates | simplified; no directory-chunk address space, no COW prefix view |
| Host extent store | deviates | whole-extent release; no zero-reference run splitting |
| State image roles/references/pins + host pool | deviates | no fork/transfer/residency state machine; DFlash/hidden transfer not wired |
| Shortlist lookup on the real path | aligned | host residency matching via `PrefixIndex` + authoritative exact match; real-artifact gate |
| Session lineage + demand retention | approximate | policy is authoritative, but the ordering/weights differ from upstream's `ResourceManager` |
| Cost model | deviates | priced, but no calibrated presets; not used for ranking yet |
| Multi-checkpoint capture at declared boundaries | partial | one extra residency at the rewrite-checkpoint frontier (the assistant opener / prompt frontier); it shares the terminal's KV pages and owns only its GDN snapshot and hidden, and is published opportunistically from spare capacity (skipped when the arena is full, so it never evicts a terminal) |
| Serving surface (`prompt_cache_retention` semantics, `cached_input_tokens` alignment, docs) | not implemented | pending |

## Deviation register (must stay explicit)

1. Frontend renderer: hand-written ChatML, not jinja; boundary resolution is direct-offset based.
2. Storage is not a reference-counted address space; no prefix-view sharing or COW.
3. Retention is a simplified owner/session/demand policy, not `ResourceManager`.
4. Cost model is uncalibrated and unused for ranking.
5. TP2, MTP, and DFlash handling is local; upstream has no TP counterpart.
6. Multi-checkpoint capture is partial: only the rewrite-checkpoint frontier is captured, and
   prompt-internal boundaries resolved by the frontend (`context_cache.boundaries`) are not. The
   captured prefix shares the terminal's KV pages; it owns its own GDN snapshot and hidden, which
   is the state image upstream also stores per checkpoint.
7. Prefill is not split at capture frontiers, and the prefill schedule only materializes the
   continuation hidden at the rewrite frontier (`rewrite_checkpoint_hidden`), not at an arbitrary
   frontier. Capturing a prompt-internal boundary therefore first needs a schedule output for the
   hidden at that frontier: the GDN snapshot alone is insufficient because the MTP bridge and
   zero-suffix sampling need the continuation hidden. Until then, only the rewrite-checkpoint
   frontier (which equals the prompt frontier for the default render) is captured.

## Conformance evidence

- Protocol: `ninfer_openai_schema_test`, `ninfer_anthropic_schema_test`,
  `ninfer_responses_schema_test`.
- Frontend boundaries: `ninfer_qwen3_6_cache_boundary_test`.
- Identity digests: `ninfer_prefix_identity_test`.
- Storage primitives: `ninfer_kv_page_registry_test`, `ninfer_kv_extent_store_test`,
  `ninfer_state_image_store_test`.
- Engine primitives: `ninfer_prefix_index_test`, `ninfer_context_cost_test`,
  `ninfer_retention_policy_test`.
- Local integration: `ninfer_host_prefix_table_test`, `ninfer_host_kv_arena_test`,
  `ninfer_concurrent_executor_test`, `ninfer_linear_state_transfer_test`.
- Real artifact: `ninfer_qwen3_8_27b_v100-duo_host_kv_real_test` (phase 1 same-lane, phase 2
  cross-lane content-addressed claims).

A feature may be called aligned only when a test in this list exercises it and passes. When a new
deviation is introduced, add it to the register and to the table in the same change.
