# ISSUE-009: remaining performance opportunities after v11av

> **Update (2026-10-23):** Experiment A's acceptance result is now recorded — v11ay passes
> both mandatory `1e6` spread gates and both totals. See the acceptance/verification
> sections in
> [`ISSUE-009`](../done/ISSUE-009-output-cell-ownership-cartesian-prototype.md), the
> `## Gate result (2026-10-23)` section in
> [`ISSUE-006`](../done/ISSUE-006-prototype-success-protocol.md), and the session log
> `history/2026/10/23.md`. The v11ay implementation is `d1c5fc7a`; the results note is
> `fd5fb705`; the tracker/acceptance update is `f4922aaa`. The opportunities below remain
> **unmeasured next-step proposals**, not blockers for Experiment A.

## Scope and baseline

Offline review of the experiment history, captured source/SASS, benchmark logs, and
preprocessing trace. No CUDA builds or GPU measurements were performed for this note;
proposals below are unmeasured unless explicitly attributed to existing artifacts.
No kernel or library code was changed by this investigation.

**The checkout changed during the investigation:** newer v11ax/v11ay experiments were
added by separate work. Do not repeat those changes or present their measurements as
measurements from this investigation. The current source was advanced to v11ay in
`d1c5fc7a`; its results were recorded in `fd5fb705`.

Let `A = profile-artifacts/expA-output-owned/opt-2026-10-22/`.
The v11av result is approximately:

| Workload | Preprocessing | Metadata | Spread, including split reduction | Total |
|---|---:|---:|---:|---:|
| Uniform 1e6, 64³, qb=1 radix | 3.082 ms | 0.444 ms | 12.575 ms | 16.101 ms |
| Clustered 1e6, 64³, qb=4 counting | 1.467 ms | ~0.130 ms | 13.714 ms | 15.311 ms |
| Sparse 1e5, 128³, qb=4 radix | 0.284 ms | 0.273 ms | 1.534 ms | 2.092 ms |

Source: `A/v11av/bench.txt:13–24,37–48,61–72`.
Historical 6.2 ms preprocessing / 20 ms uniform-total figures are superseded.

The newer interleaved v11ay benchmark reports:

| Workload | v11av spread → v11ay spread | v11av total → v11ay total |
|---|---:|---:|
| Uniform | 12.528 → **10.912 ms** | 16.058 → **14.446 ms** |
| Clustered | 13.714 → **11.232 ms** | 15.314 → **12.834 ms** |
| Sparse | 1.536 → **1.393 ms** | 2.089 → **1.956 ms** |

Source: `A/v11ay/README.md`, `bench.txt`, `ab_v11av_v11ay.txt`.
These are existing four-process-run A/B results, not new runs here. v11ay removes y/z
coefficient-shift selects with contiguous padded loads and factors strength × z outside
the cell updates. The latter changes FP32 multiplication association; recorded relL2
remains at the reordering noise floor. Both mandatory spread targets now pass.

## Why the shared-load flamegraph is misleading

The detailed sourced investigation is
[ISSUE-009-v11av-profile-analysis.md](ISSUE-009-v11av-profile-analysis.md).
Independent CSV aggregation in this investigation confirmed:

- v11av distinct-PC executed warp instructions: **671,898,962 uniform / 868,613,915
  clustered**, matching the report totals. Naively summing source-correlated SASS rows
  gives **716,843,518 / 921,479,579** because 70 / 92 PCs appear repeatedly.
- Distinct-PC all-samples: **171,630 / 201,895**, not the naive source-row totals.
- **No v11av LDS instruction has excessive shared wavefronts.** Its shared-bank excess
  belongs to stores. Broadcast scan loads are not a bank-conflict bottleneck.
- The v11av accumulation arithmetic executes with only **6.426 / 4.122 participating
  lanes per warp instruction**. The fixed 16-cell update also computes zero-weight cells.
- Short-scoreboard is about **6.4% / 6.9% of scan all-samples**, usually on consumers
  waiting for metadata/coefficients, not on the triggering LDS source lines.
- The archived clustered tree omits `prepare_two_coefficients` under block staging.
  Corrected staging attribution is ~7.6%, not 4.7%; neither is a wall-time saving bound.

Sources: `A/v11av/{uniform_qb1,clustered_qb4}_{source.csv,details.txt}` and the analysis
note's counting method and primary NVIDIA documentation links. A sample-attribution
flamegraph is not a device execution timeline or a partition of elapsed kernel time.

**Tooling next step:** deduplicate SASS metrics by `(kernel, PC)` before aggregating,
check their instruction total against Nsight's kernel total, and make source-range/call
maps version-specific. Current `devel/morton_spread/ncu_samples_to_chrome.py:173–218`
still aggregates source occurrences without PC deduplication; updating v11ay line ranges
and the block callee alone does not fix that. Retain dropped/unmapped samples visibly.

Do not generalize v11av's conflict-free LDS finding to v11ay: the latter changes the row
stride and has some LDS excess wavefronts. Its saved report also shows higher L1/TEX
utilization (~78% clustered). Re-profile the current kernel before declaring its new
bottleneck. Neither a throughput percentage nor a source sample share is useful FLOP/s
or an independently recoverable speedup.

## Prioritized remaining experiments

### 1. Reduce setup work rather than repeat shared-layout tuning

Uniform setup remains ~3.53 ms with v11ay, about a quarter of its total.
The v11at preprocessing code is unchanged in the v11av/v11ay benchmarks.
Its nsys trace attributes approximately:

- **1.35 ms** to the random packed-payload gather;
- **0.9–1.0 ms** to radix sorting;
- **0.585 ms** to qbin computation plus input packing;
- ~0.2 ms to bin-start construction; ~0.039 ms to index initialization.

Sources: `A/v11at/preprocessing_uniform_nsys.txt:14–29`, `A/v11at/README.md:35–44`.
These are approximate per-kernel profiling measurements, not an exactly additive wall
breakdown. The trace mixes prototype/library kernels and has no usable NVTX summary.

**First substantial test: fused rank-based counting at the same qb.** In one pass compute
`qbin[i]` and `rank[i] = atomicAdd(counts[qbin[i]], 1)`. Scan counts, then scatter using
`binstart[qbin[i]] + rank[i]`, without a second cursor atomic. This removes a separate
histogram launch, cursor clearing, and the second reservation. The library already uses
this pattern in `include/cufinufft/spreadinterp.hpp:299–325`. The prototype currently
discards the first rank and repeats the reservation (`A/v11av/exp_a_cartesian.cu:95–98,
129–146,1300–1313`). Keep rank storage separate from any destination permutation array.
Compare uniform qb=1 against radix and clustered qb=4 against current counting; contention
means a win cannot be assumed. Keep the no-atomics requirement scoped to spreading.

**Alternative:** direct sorting of `(qbin, PackedPoint)` followed by linear deinterleave
removes the random gather but increases sort payload from 4 to 32 bytes. Sort plus
conversion must beat the existing ~2.3 ms sort-plus-gather budget; the 1.35 ms gather is
not a free saving. Consuming packed sorted records is a separate experiment: account for
extra bytes and any spread regression from changing global staging loads.

Small independent fusions: initialize indices in qbin+pack; mark bin boundaries during
gather; write marks in reversed order to remove the reverse kernel; write the `M` sentinel
on device. Existing fill/mark/reverse kernels together cost only ~0.125 ms before added
work. Relevant source: `A/v11av/exp_a_cartesian.cu:1013–1092,1269–1314`.
Relevant radix bit ranges are already restricted (18 uniform bits, 15 sparse bits);
“sort only necessary bits” is not a new optimization (`:1213–1215,1282–1284`).

### 2. Replace warp-visits cube enumeration with exact x-run endpoint sums

This is the cleanest small, low-risk setup experiment. `kernel_visits` currently visits
all bins, even though x-adjacent bin counts telescope:

```
sum(count[x0:x1+1]) = binstart[base+x1+1] - binstart[base+x0]
```

Handle periodic x ranges as at most two segments. Preserve one warp per region and its
warp reduction. For uniform qb=1, this changes **14³ bin iterations to 14² y/z rows**,
without constructing a new summed-area table. `kernel_block_visits` already implements
the needed endpoint logic. Sources: `A/v11av/exp_a_cartesian.cu:152–201,963–1010`.

The target is the **~0.315 ms uniform visits kernel**, not the entire metadata stage.
A Python check of endpoint sums against explicit periodic-bin sums passed **21,784**
1D cases for 1/2/4/7/16/32/64 x bins and all ranges up to one full period. This establishes
only the telescoping/wrapping algebra, not GPU speed or complete kernel correctness.
For ranges longer than one period, explicitly choose whether to preserve the old
multiplicity or match the spread's capped x enumeration; canonical workloads are shorter.

This differs from the rejected SAT and threads-per-region experiments: it reduces work
without adding another prefix structure or changing launch granularity.

Metadata currently also performs four blocking D2H copies. A device summary and one small
host copy could remove some overhead; the residual unassigned metadata budget is only
~0.07 ms uniform/clustered and ~0.115 ms sparse. Sparse single-thread tile compaction
itself costs ~0.07 ms and is another small target. Sources: `A/v11av/exp_a_cartesian.cu:
1332–1419`, `A/v11av/bench.txt`. Residual budgets are not proven synchronization costs.

### 3. Reuse geometry if coordinates repeat

If only strengths change, retain the permutation, sorted coordinates, bin starts, active
tile list, dispatch decision, split count, and allocated workspace. Refresh only
`sorted_strength[s] = strength[permutation[s]]` per execute. Coordinates also determine
support starts and Horner arguments; cache those only after measuring memory tradeoffs.

Avoidable work is bounded by current setup **3.53 / 1.60 / 0.56 ms** before subtracting
strength-refresh time. This is conditional on application reuse, not a one-shot speedup.
Compare against method 3 with one `makeplan/setpts` and repeated `execute` calls, not the
current fresh-setpts baseline. Library separation: `src/cuda/setpts.cu:28–74`,
`src/cuda/execute.cu:106–150`. A full 18-coefficient cache trades computation for much
more global traffic; do not assume it helps.

### 4. Small scan hypothesis after v11ay: simplify nonwrapping overlap

Contiguous coefficient rows make

```
coefficient_offset - local_offset = group_start - support_start
```

for overlapping nonwrapping intervals. Define `d = group_start - support_start`;
overlap is simply `1-G <= d && d < 6`, or the unsigned range test
`unsigned(d + G - 1) < unsigned(6 + G - 1)`. y/z coefficient bases then use `d` directly.
This could remove some min/max/count work from the existing fast template without adding
a per-point dispatch branch or changing the periodic slow path. x still needs its
unpadded-side handling. The compiler may already cancel parts of the offset algebra;
inspect SASS and benchmark rather than infer a saving from source size.

Source: `A/v11ay/exp_a_cartesian.cu:366–374,379–413`; the captured SASS still computes
min/max interval overlap. Exhaustive Python checks passed **38,684** 1D cases over
sizes 6/7/8/16/64/128, group sizes 2/4, full groups and nonwrapping width-6 supports.
This validates the identity and overlap predicate only. No GPU implementation or speed
measurement was made; expected scope is support-placement work, not all scan time.

A larger scanner/ownership redesign remains possible, but increasing occupancy or active
lanes alone is not enough: the goal is fewer warp-issued instructions per useful cell
contribution. The rejected 8-cell geometry increased visits 43–50%; 32-cell ownership
would increase accumulator registers. A larger block tile reducing staging rereads need
not reduce scan rereads. These require an explicit work/register model before coding.

## Measurement and validation before accepting another win

- Use canonical configs and interleaved warm process runs; clocks were unfixed in saved
  ncu captures. Keep ncu timing separate from normal execution timing.
- The harness's reported `total` is **preprocessing median + one metadata measurement +
  spread median**, not one contiguous end-to-end sample. Resource allocation is charged
  inconsistently: main preprocessing and split-partial allocations are excluded, whereas
  metadata allocations are included. Add a repeated contiguous one-shot/steady-state wall
  measurement before drawing production end-to-end conclusions.
- The existing strengths are all `(0.1,0.2)`. Use varying signed complex strengths before
  accepting permutation, strength-refresh, or arithmetic-reassociation changes.
- Validate periodic boundary inputs, partial tiles, inactive output cells and odd batches.
  The block fast-path test checks at most 32 points although batches exceed 32, and the
  unsplit compacted launch has no general inactive-grid clear. These are validation gaps,
  not demonstrated errors on the mandatory datasets.

Sources for the last points: `A/v11av/exp_a_cartesian.cu:1102–1155,1201,1263–1323,
813–829,1332–1464,1500–1508`; v11ay retains the corresponding harness structure.
Do not optimize against stale source-correlated percentages or remove required clearing,
conversion, reduction, or metadata cost from accounting.
