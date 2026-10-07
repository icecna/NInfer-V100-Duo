# Prefix reuse at an interior boundary is not exact (active fix)

Branch: `fix/prefix-reuse-boundary-exactness`, based on the tip of `feat/context-cache-port`
(`395f751f`). Temporary work plan for active work; delete it when the fix lands or is abandoned.

## Symptom

A multi-turn conversation whose second turn reuses the previous turn's checkpoint
(`PrefixReusePath::AppendAtFrontier`) does not reproduce a full recompute: with greedy decoding the
generated text differs on some turns. Same rendered prompt, same single-request schedule, reuse
versus recompute. Nothing about it needs the host cache.

## Evidence (all reproducible on this branch)

Serve-level, two passes over the same workload, ONE request at a time, control with
`--no-prefix-reuse`:

| turn | control prompt / hit / path | host prompt / hit / path |
|---|---|---|
| 0 | 200 / 0 / full_reset | 200 / 0 / full_reset |
| 1 | 268 / 0 / full_reset | 268 / 198 / append_frontier |
| 2 | 336 / 0 / full_reset | 336 / 266 / append_frontier |
| 3 | 404 / 0 / full_reset | 404 / 334 / append_frontier |

`prompt_tokens` match turn by turn, so both passes rendered the same token sequence, and the first
differing answer is at a turn whose input is built from turns that already matched. 3 of 8 turns
differ at 2 sessions x 4 turns, 7 of 24 at 4 sessions x 6 turns.

Isolation: control `--no-prefix-reuse` with 0 hits versus host `--context-host-cache-mib 0`, which
reuses only DEVICE-resident entries (6 hits, all `append_frontier`) -- same 3 of 8 turns. So this is
not the host cache, not its transfers and not concurrency.

Pre-existing: with `text_context_impl.h` and `linear_state_slots.h` restored byte-identical to the
base commit `7b2a0ada`, the isolation run still fails on the same three turns. The only device-path
changes left on this branch are the inert `PagedKVPool::pin_pages/unpin_pages` additions and the
host-only `linear_state_transfer.cu` fix, neither of which this path calls.

Boundary detail worth exploiting: reuse is 198 where the previous turn's prompt was 200 and its
terminal frontier would be about 247, i.e. the checkpoint sits two tokens before the end of the
previous turn's prompt (the assistant opener), NOT at the terminal frontier. The engine gate's
existing phase-1 append case reuses the terminal frontier and is bit-exact, which is why it never
caught this.

## Reproduce (fastest first)

Engine gate, once a case exists (see step 1): the test binary is
`build-host-prefixes/tests/ninfer_qwen3_8_27b_v100-duo_host_kv_real_test`, run with
`NINFER_V100_DUO_ARTIFACT` set, and it needs the GPUs exclusively, so wrap it in the
stop/test/start script pattern.

Serve level, ~4 minutes, self-contained (stops and restarts `ai-server` from an EXIT trap):

```
scripts/run-serve-context-cache-gate.sh --cache-mib 0 --sessions 2 --turns 4 --concurrency 1 --evict-mib 0
```

expect `mismatched=3` and `RESULT: FAIL` today. Per-request evidence lands in
`<work-dir>/{control,host}.requests.jsonl` (`result.prompt_tokens`,
`result.prefix_cache_hit_tokens`, `result.prefix_reuse_path`).

## Plan

1. **Move the reproducer into the engine gate.** Add a turn-boundary case to
   `tests/targets/qwen3_6_27b/test_engine_v100-duo_host_kv_real.cpp`: turn 1 prompt, then turn 2 =
   turn 1 prompt + the generated answer + a new user message, with `allow_prefix_reuse` on, and
   require `prefix_reuse_path == append_frontier` rather than `full_reset`. Compare it in-session
   against a cold prefill of the same turn-2 prompt (the `in-session cold-vs-claim` comparison that
   the `prefix_exact` round already uses) so the schedule is identical.
2. **Instrument the boundary.** At the restore, hash and compare against a cold prefill at the same
   frontier: the GDN state (use `snapshot_linear_state_slot_to_host`, which honours the slot stride
   -- reading `recurrent_slot(...).bytes()` linearly is a trap, see the lessons), the tail hidden,
   and the restored `execution_frontier` versus the checkpoint frontier. Two ranks.
3. **Ranked suspects.**
   - The capture into the rewrite-checkpoint slot happens AFTER the MTP draft prefill in the base
     ordering, so the slot may hold a state up to `draft_tokens` past the boundary. `git log` for
     the reverted attempt (`text_context_impl.h`) shows the capture moved before the draft did not
     change the host-cache symptom, but it was never measured on this boundary.
   - The two-token offset: anything that assumes "reuse length == the previous prompt's length"
     recomputes the wrong tail.
   - The MTP backend KV valid length (`mtp_kv_valid`, `pages_for_tokens(base - 1)`) at an interior
     boundary; the terminal-frontier cases may have been hiding an off-by-one.
4. **Fix, then accept.** A claim must be numerically exact against a same-schedule recompute, at any
   frontier.

## Acceptance

- The new engine-gate case is bit-exact against its in-session cold prefill.
- `scripts/run-serve-context-cache-gate.sh --cache-mib 0 --sessions 2 --turns 4 --concurrency 1 --evict-mib 0`
  reports `mismatched=0`.
- `scripts/run-serve-context-cache-gate.sh --cache-mib 2048 --sessions 4 --turns 6 --concurrency 2`
  reports `mismatched=0` for its compared passes, `0` errors and at least one checkpoint hit.
- Real-artefact gate stays green: `ninfer_qwen3_8_27b_v100-duo_host_kv_real_test` phases 1 and 2.

## Related but separate: "unexplained cold fills"

Not this defect (that one answers wrongly; this one answers coldly). Production request log at
`/mnt/models/ai-launchers/requests.jsonl` over 1495 requests: overall 85.1% of prompt tokens were
reused, but by family

| family | reuse ratio |
|---|---|
| multi-message with tools | 88.9% |
| single message (cold start) | 19.8% |
| multi-message with thinking dropped (`preserve_thinking=false`) | 12.5% |
| multi-message with `preserve_thinking=true` | 99.3% |

so the dominant cause of a `full_reset` where one was not expected is that a re-render which drops
earlier thinking is not a strict extension of the retained prefix. Upstream's pinned commit is
literally about keeping cache boundaries through template trimming; the local engine instead gives
up entirely. That is a frontend/reuse-matching gap, worth its own work item, and the numbers above
are the motivation.

## Lessons carried into this branch

- Diagnostics that mutate state must be environment-guarded: an unguarded "coverage experiment"
  scrub corrupted every measurement for a whole session.
- Do not treat an input as excluded by argument: the KV bytes were "excluded" that way and were the
  corrupt one.
- `LinearStateSlots::state_slot_count` must stay `2 * max_concurrency`; a third per-lane slot costs
  73.4 MiB per slot per GPU on the 27B target and broke the production reservation.
- `transfer_slot` copies each layer's slot contiguously (fixed in `9d22a5e8`); the device-resident
  exactness of terminal claims and of the checkpoint feature depends on it.
