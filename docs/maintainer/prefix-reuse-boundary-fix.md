# Prefix reuse at an interior boundary is not bit-exact: root cause and decision

Status: **not fixed, deliberately deferred.** The instance is still present on the branch and on
`master` (it is pre-existing). The change that would fix it is a numeric-contract change in the
dense prefill attention and was judged too large to land in this pass; the reuse gates are to be
downgraded to the contract the engine can actually keep (see *Decision*). This document is the
authority for those findings and for the reproducer, so the work can be resumed without redoing the
investigation.

## Symptom

A multi-turn conversation whose later turns reuse an earlier turn's checkpoint
(`PrefixReusePath::AppendAtFrontier` / `RestoreTurnCheckpoint`) does not reproduce a full recompute:
with greedy decoding the generated text differs on some turns. Same rendered prompt, same
single-request schedule, reuse versus recompute. It does not need the host cache; the
device-resident path shows it too.

## Root cause (confirmed by reading the path, not by suspecting a kernel)

The dense causal prefill attention's *numeric configuration depends on the shape of the chunk it
runs in*, and the chunk shape for the same absolute positions differs between turns.

1. `TextContext::prefill_impl_tp2` (`text_context_impl.h`) cuts each prefill chunk at the
   rewrite-checkpoint frontier:

   ```cpp
   int len = std::min(chunk, T);
   if (checkpoint_rel > 0 && len > checkpoint_rel) { len = checkpoint_rel; }
   ```

   and the attention envelope is `{base_i + len, base_i + len}`.

2. Therefore the same absolute positions are processed with different `len`, hence different
   envelope, in different turns. Two independent causes, both unavoidable:
   - the frontier cut ends the chunk at an arbitrary position that changes per turn; and
   - the **last** chunk ends at the prompt end, which grows every turn
     (`[256,310)` for a 310-token prompt versus `[256,346)` for a 346-token prompt).

3. The attention reads that shape:
   - `gqa_small_t_launch_capacity(envelope, tokens, dtype)` derives the **split-K count** from the
     envelope (e.g. the i8 `tokens == 5 && window in (128,512]` case is `div_up(window, 32)`, so
     window 308 gives 10 splits and 344 gives 11);
   - `logical_capacity` / `implementation_window` are `envelope.max_visible_keys`;
   - the VoltaFlash path uses `n_kv_total = envelope.max_visible_keys`, `base = n_kv_total - width`
     and `n_kv = round_up_keys(base + begin + tokens)`, which is the key extent that drives the
     fattn **stream-K** partition.

Different split/stream-K grouping over the same valid keys is a different floating-point
summation order. The prefix hidden and KV therefore differ below the greedy margin, and the
divergence surfaces as a flipped token after some tens of steps.

**Consequence worth stating plainly:** the KV a position gets is not a function of
`(position, KV[0..q])` alone, so the engine's prefix is **not reproducible across prompt lengths**.
That is why a retained prefix can never equal a fresh recompute while the kernels stay
chunk-shape dependent. It is also why every "route" change below failed: the input shape moved, not
the route.

## Evidence

Reproducer and knobs (temporary diagnostics; removed with the rest of the work-in-progress):

- Engine gate turn-boundary case in `test_engine_v100-duo_host_kv_real.cpp` (an in-session cold
  prefill versus a claim, per turn). At the default 32 output tokens it passed; that is what made
  the first bisect wrong. With a stretched shared prefix
  (`NINFER_BOUNDARY_LONG_PREFIX=1`) and `NINFER_BOUNDARY_OUTPUT=200` it reports
  `mismatched=3` on both the host and the device path.
- Prefill logits probe (`NINFER_BOUNDARY_LOGITS=1 NINFER_BOUNDARY_OUTPUT=1`,
  `NINFER_HOST_GATE_NO_MTP=1`, `NINFER_BOUNDARY_DEVICE_ONLY=1`): the claim's and the cold's
  last-position prefill logits differ in ~95% of the 248320 entries by a few bf16 ULP
  (`max_abs ~ 0.35-0.53`) while the argmax still agrees. So the prefill state differs; greedy only
  amplifies it later.
- `NINFER_BOUNDARY_REUSE_LAST_ONLY=1` (intermediate checkpoints established by cold prefills):
  identical divergence. So it is not the provenance of the checkpoint.
- Changing `NINFER_BOUNDARY_CHUNK` changes the divergence. So it is the chunking.
- **Pre-existing:** `master` (`774061c4`, the parent of `6e3c7298`) fails the same way once the
  master `prefix_real` case is run with `kOutput = 200` (`append vs append_cold: greedy outputs
  differ`). The earlier "first bad commit is `6e3c7298`" bisect was an artifact of 32-token turns:
  that commit's A16 T-keyed routes make the near-tie surface sooner, they do not create it.

Negatives (each checked, each insufficient):

| Trial | Result |
|---|---|
| Unify the A16 routes (attn/gdn input proj, generic linear, swiglu, linear_add) to the wide route during prefill | real contributor: serve gate 3 → 2 differences, engine reproducer green at 32 output tokens; not sufficient |
| Route the GQA attention by envelope instead of width (VoltaFlash versus ChunkedSmallT) | changes which tokens flip, does not remove the logit divergence |
| Force every prefill `causal_conv1d_silu` to one route (its `T<=16 / T<=64 / T>64` split) | numerically equivalent; counts byte-identical |
| Force GDN `launch_recurrent_prepared` versus `launch_recurrent_inout` (`T>=64` split) | numerically equivalent |
| Force the small-T attention split-K to 1 | still differs; the chunk shape feeds more than the split count |
| Explain it by the host transfer, the decode schedule, CUDA graphs, MTP, or W4A4 | each excluded separately |

Also audited and *not* the cause for the observed widths: `rope.cu`/`add_bias` block choices
(elementwise), `context_kv_materialize` page gates, `linear/ggml_k` (only the Q4_K_M artifact),
`dynamic_grouped_conv`, `bidirectional_gqa_attention` (vision). Latent and worth a separate look if
shorter suffixes ever occur: `gdn_gating_proj` `k27Routes` (`cols <= 8` selects `SmallTSplit10`,
split-K 10, versus `Split8`), and `bf16_gdn_norm_gating_proj_27` (`tokens <= 2/14/28`).

## The fix that would be required

**Make the dense causal prefill attention's reduction a function of the query's absolute position
alone.** Concretely, in the attention launcher:

1. align the Q-block (VoltaFlash) and small-T piece grids to **absolute** positions instead of the
   chunk base, so the same absolute interval lands in the same block in every turn;
2. derive `n_kv` / the window from the block's **nominal** absolute end
   (`block_abs_start + nominal_block_tokens`), not from the truncated end;
3. derive the split count from that same nominal end, so the reduction grouping is position-keyed;
4. rely on the (already index/position-based) causal mask to zero the keys past the valid prefix;
   those keys are still *read*, so confirm the read is safe (i8 codes dequantize finite, mask
   contributes exactly 0).

Affected: `src/ops/launcher/gqa_attention_volta_flash.cu`,
`src/ops/launcher/gqa_attention_decode.cu`, `src/ops/wrapper/gqa_attention.cpp`. Cost: short chunks
lose some partition parallelism; prefill at long context pays a little.

Rejected alternatives, for the record:

- *Stop cutting the chunk at the checkpoint frontier.* Does not help: the last chunk still ends at
  the prompt end, which grows.
- *Round the frontier cut to a `prefill_chunk` (128) boundary.* Breaks the checkpoint position
  semantics and still does not address prompt growth.

## Decision: downgrade the gates instead

The divergence is a numeric reproducibility gap, not a correctness defect:

- the reuse still produces a coherent continuation, not garbage, a bad format, or a crash;
- production samples at `temperature = 1.0`, `top_k = 20`, where the trajectories are not
  reproducible in the first place, so this is not observable to a user;
- there is no memory, performance, or protocol impact (cache hits, reuse ratio, and throughput are
  unaffected).

So the gates should assert what the engine can keep, and report the rest:

- **Engine turn-boundary case:** require the **first generated token** of the claim to equal that
  of the same-session cold prefill, plus the reuse actually being taken (expected path, positive
  reused tokens). Report full-sequence divergences instead of failing on them. This still catches
  the real failure modes: wrong lane, missing KV, stale or absent checkpoint.
- **`scripts/run-serve-context-cache-gate.sh`:** require `0` errors, `0` empty responses, a
  positive host reused-token count and at least one checkpoint hit, plus a shared leading response
  prefix per compared turn; report the full-text divergence count. Drop the "identical text per
  (session, turn)" assertion.
- Keep the strict checks that *are* achievable: phase 1 (host D2H/H2D round trip of the same
  computation, no re-prefill) and the cross-lane round trip should stay bit-exact.

## Reproducer recipe (to rebuild when the fix is attempted)

The diagnostics that produced the evidence above were temporary and are not in the tree. To rebuild
them:

1. In `tests/targets/qwen3_6_27b/test_engine_v100-duo_host_kv_real.cpp` add a turn-boundary session:
   a fixed 4-turn conversation (system prefix plus one user message per turn, the assistant replies
   taken from literal bodies so the prompt length does not depend on the generated output), with
   `allow_prefix_reuse` on for the claim and a cold prefill of the identical prompt in the same
   session as the comparator. Env knobs used:
   `NINFER_BOUNDARY_LONG_PREFIX` (stretch the shared system prefix),
   `NINFER_BOUNDARY_OUTPUT` (output tokens per turn; 200 needed to surface the near-tie),
   `NINFER_BOUNDARY_CHUNK` (engine `prefill_chunk`; must stay a nonzero multiple of 128),
   `NINFER_BOUNDARY_LOGITS` (compare `debug_enable_logit_capture` /
   `debug_last_round_logits_bf16`; with `output == 1` that is the prefill's own logits),
   `NINFER_BOUNDARY_DEVICE_ONLY`, `NINFER_HOST_GATE_NO_MTP`, `NINFER_BOUNDARY_REUSE_LAST_ONLY`,
   `NINFER_HOST_GATE_BOUNDARY_ONLY`.
2. Prefer `scripts/run-host-kv-gate.sh` (it does the `ai-server stop` / run / `start` trivially) and
   `scripts/run-serve-context-cache-gate.sh` for the production shape.

## Related but separate: "unexplained cold fills"

Not this defect (that one answers *wrongly*; this one answers *coldly*). Production request log at
`/mnt/models/ai-launchers/requests.jsonl` over 1495 requests: overall 85.1% of prompt tokens were
reused, but by family

| family | reuse ratio |
|---|---|
| multi-message with tools | 88.9% |
| single message (cold start) | 19.8% |
| multi-message with thinking dropped (`preserve_thinking=false`) | 12.5% |
| multi-message with `preserve_thinking=true` | 99.3% |

A later pass over the same log found 266/1582 requests are `full_reset`, and that they correlate
strongly with idle time (`<30 s` 10%, `30 s-5 m` 37%, `>5 m` 88%), with
`prefix_cache_hit_tokens=0` and the whole prompt recomputed. Two distinct causes, neither the
numeric bug above:

- A retained host entry is claimable **only at its exact frontier** (`request_plan_impl.h`), and a
  re-render that drops earlier thinking is not a strict extension of it, so a conversation that
  loses its device lane (claim consumes the entry, the lane is later reused or reclaimed) has
  nothing to fall back to.
- After an idle gap the conversation is typically no longer the resident one on any lane, which is
  the expected "reclaimed or displaced by another conversation" behaviour.

The first is a frontend/reuse-matching gap worth its own work item.

## Lessons carried into this branch

- A near-tie needs a long enough greedy trajectory to show up. The first bisect declared `master`
  clean and blamed `6e3c7298` because the test generated 32 tokens. Always check that the
  reproducer's horizon can actually exhibit the failure before trusting a bisect.
- Route-keyed diagnostics answer the wrong question. Forcing every T-keyed kernel to one route
  changed which tokens flipped but never moved the logits, because the divergence entered through
  the *chunk shape* (the envelope), not the route. Reading the computation path found it; the
  probing had circled it for a long time.
- Diagnostics that mutate state must be environment-guarded, and remove them before the final diff.
- `LinearStateSlots::state_slot_count` must stay `2 * max_concurrency`; a third per-lane slot costs
  73.4 MiB per slot per GPU on the 27B target and broke the production reservation.
- `transfer_slot` copies each layer's slot contiguously (fixed in `9d22a5e8`); the device-resident
  exactness of terminal claims and of the checkpoint feature depends on it.
