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

Route B (chosen): align the local storage with upstream rather than patching the page-granular
local KV. The local branch's `PagedKVPool` is page-granular with no reference counts and no
column-level coverage; upstream's context cache rests on a reference-counted KV address space with
per-page coverage, prefix views, and a state-image store. Every attempt to capture a frontier
inside a page on the local KV failed for that reason, so the fix is the storage model, not another
capture path.

Staged plan (each stage keeps the build green and lands with tests):

- **B1 logical pages and address spaces.** Logical page metadata (references, active references,
  source pins, writer, protected/committed columns, device/host replica) and an ordered address
  space with a committed frontier, a checkpoint frontier, whole-page prefix views/fork, and
  inactive-prefix truncate. Host-only tests. Builds on `KvPageRegistry` as the starting point.
  Status: prefix view (`KvAddressSpace::fork_prefix`) and device-pool adoption
  (`KvPageRegistry::adopt` / `KvAddressSpace::adopt`) landed with host-only tests.
- **B2 host extents with sharing.** Host extents whose membership can be referenced by more than
  one residency, plus an owned sub-page tail for a frontier inside a page; the release/partition
  rules that let a checkpoint drop without freeing the terminal's pages. Host-only tests.
  Status: extent reference counting (`HostKvExtentStore::retain`) landed with a host-only test;
  the shared-run plus owned-tail composition is still to build.
- **B3 program adoption.** `SequenceKVBundle` becomes an address-space handle; retain builds a
  checkpoint as a prefix view (share whole pages, own the sub-page tail) and the claim restores
  through the address space. Real-artifact gate must pass.
- **B4 state image.** Route the checkpoint GDN snapshot and hidden through `StateImageStore`,
  mirroring both ranks, with the retention policy already in place.
- **B5 capture re-enable.** Re-enable the frontier/marker capture on the new storage; acceptance is
  the real gate plus the schedule-matched output comparison that caught the earlier failure.

Until B5 lands, multi-checkpoint capture stays reverted and the parity table keeps its current
status. B1/B2 are pure host modules and do not touch the verified execution path.

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
| Multi-checkpoint capture at declared boundaries | not implemented | only the terminal residency is published; a frontier residency was implemented and reverted (see deviation 6) |
| Serving surface (`prompt_cache_retention` semantics, `cached_input_tokens` alignment, docs) | not implemented | pending |

## Deviation register (must stay explicit)

1. Frontend renderer: hand-written ChatML, not jinja; boundary resolution is direct-offset based.
2. Storage is not a reference-counted address space; no prefix-view sharing or COW.
3. Retention is a simplified owner/session/demand policy, not `ResourceManager`.
4. Cost model is uncalibrated and unused for ranking.
5. TP2, MTP, and DFlash handling is local; upstream has no TP counterpart.
6. Multi-checkpoint capture is not implemented. A second residency at the rewrite-checkpoint
   frontier was implemented (its KV pages shared with the terminal, owning only the GDN snapshot
   and hidden) and then reverted: the real-artifact gate gained a schedule-matched output
   comparison for a replay of the same rendered prompt, and the restored state produced a
   different decode than the control's cold prefill.

   The snapshot source was checked and is correct: both ranks write their rewrite GDN slot
   (`text_context_impl.h` copies `current -> rewrite` per rank when the chunk ends at the
   checkpoint frontier) and rank 0 writes the frontier hidden there.

   A decisive experiment settled the nature of the failure: replaying the same base prompt twice
   in one session, adjacent, once with reuse (restoring the prefix residency) and once cold
   (`allow_prefix_reuse=false`), produced different outputs while the first generated token was
   identical. Same session and adjacent positions rule out cross-session schedule drift, so the
   restore is genuinely wrong. The matching first token means the restored continuation hidden is
   correct; the divergence that follows points at the accumulated state (the GDN snapshot or the
   MTP backend KV tail), not at the initial hidden or the attention mask. The next step is to
   separate those two, for example by testing the target backend without MTP or by restoring a
   frontier whose KV is page-aligned. The feature stays reverted until then.
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
