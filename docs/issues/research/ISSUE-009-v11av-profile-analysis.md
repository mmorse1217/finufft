# ISSUE-009: interpreting the v11av profile

**Converter update (2026-10-23):** The duplicate-PC counting, stale source ranges/call
mapping, and dropped samples described below are fixed in
`devel/morton_spread/ncu_samples_to_chrome.py`. The v11av/v11ay uniform and clustered
flamegraphs/call trees were regenerated offline using their matching source snapshots
and actual ncu launch durations. Both roots now account for 100% of deduplicated CSV
samples; 14 CPU-only regression tests cover the fix. Bug descriptions and converter
line references below describe the pre-fix tool and graphs.

**Scope:** offline analysis of existing artifacts; no GPU execution, build, or kernel changes.
**Conclusion:** the shared-load-dominated reading is misleading. There are also two concrete
flamegraph-converter errors: repeated PCs are counted repeatedly, and the clustered tree
omits the new coefficient function. The strongest remaining signal is **many scan warp
instructions doing arithmetic with very few participating lanes**, alongside real but
smaller shared-result dependencies. There is no evidence that shared-memory bandwidth is
the dominant limit; localized/shared-port pressure is not ruled out.

## Evidence and counting method

Paths below use `V = profile-artifacts/expA-output-owned/opt-2026-10-22/v11av/`:

- **U:** [`V/uniform_qb1_source.csv`](../../../profile-artifacts/expA-output-owned/opt-2026-10-22/v11av/uniform_qb1_source.csv)
- **C:** [`V/clustered_qb4_source.csv`](../../../profile-artifacts/expA-output-owned/opt-2026-10-22/v11av/clustered_qb4_source.csv)
- **K:** [`V/exp_a_cartesian.cu`](../../../profile-artifacts/expA-output-owned/opt-2026-10-22/v11av/exp_a_cartesian.cu), the captured source, not its stale geometry comments.
- **P:** [`devel/morton_spread/ncu_samples_to_chrome.py`](../../../devel/morton_spread/ncu_samples_to_chrome.py)
- **I:** [`docs/issues/done/ISSUE-009-output-cell-ownership-cartesian-prototype.md`](../done/ISSUE-009-output-cell-ownership-cartesian-prototype.md)

References such as `C:555` are physical, one-based file lines. Calculations below select
actual `0x…` SASS-address rows, exclude CUDA aggregate rows and `...` placeholders, and
**deduplicate by address within this one captured kernel**. Repeated PCs have identical
numeric metrics in these exports. For exclusive source groups, retain the first occurrence
(the helper/body precedes its caller); group boundaries remain source attribution, not
execution phases. Sum thread counts before dividing by warp counts; do not average or sum
the CSV's source-level “Avg. Threads” aggregates.

### Measurement: the converter double-counts inline correlations

| Count | Uniform | Clustered |
|---|---:|---:|
| SASS-address rows / distinct PCs | 1,662 / 1,592 | 1,732 / 1,640 |
| Naively summed warp instructions | 716,843,518 | 921,479,579 |
| **Distinct-PC warp instructions** | **671,898,962** | **868,613,915** |
| Naively summed all-samples | 189,316 | 221,103 |
| **Distinct-PC all-samples** | **171,630** | **201,895** |

The distinct-PC instruction totals **exactly match** `V/{uniform_qb1,clustered_qb4}_details.txt:59`;
shared-wavefront totals also match the reports' totals at lines 89–90. For example:

- `U:487,577`: the same `0x7d553d2b80c0 IADD3 R88, R32, 0x6, RZ`, attributed both to
  helper line 366 and caller line 384, repeats **2,342 short-scoreboard samples**.
- `C:555,645`: the equivalent `0x763bcf29b6f0` repeats **4,139** such samples.
- `C:625,2459`: the same metadata `LDS.U.128` appears in both scan and caller rows.

`P:173–218` aggregates by `(file,line)`, not PC; `P:294–297` normalizes these overlapping
samples. This inflates support-placement heat particularly strongly. NVIDIA documents
inline source metrics as correlations aggregated across call sites, not an exclusive
partition of SASS [N1]. The duplicate-PC evidence above is independent of that description.

**Separate tree error:** `P:91–98` still puts `prepare_point_coefficients` under
`stage_block_quarter`, but the captured kernel calls **`prepare_two_coefficients`**
(`K:651–655`). Its **6,095 samples** are excluded by the reachability filter (`P:249–252`):
`V/clustered_qb4_calltree.txt:1` consequently accounts for only **97.2%** of its own
non-deduplicated denominator. Including the actual staging callees and deduplicating gives
**15,420 / 201,895 = 7.64%** clustered staging attribution, not 4.7%.

The graphs are also **synthetic sample layouts**, not device call stacks or timelines:
`P:294–353` scales source-ordered samples by a supplied duration. The JSON `otherData`
uses 12.580/13.710 ms, whereas the captured ncu launch durations are 12.60/13.53 ms.
Thus “software pipelining can hide at most 4.7%” (`V/README.md:43–45`, `I:1307–1314`)
is **not a valid wall-time/Amdahl bound**, even apart from the missing function.

## What the metrics actually mean

- **Executed instructions are warp instructions**, independent of participating lane count.
  Thread instructions count active lanes; predicated-on thread instructions additionally
  require a true predicate [N1]. A million FFMAs with four lanes are not 32 million useful
  lane FFMAs, but still consume warp instruction/execution resources.
- **All Samples is not all stalled time.** The statistical sampler selects a random active
  warp; another warp may issue simultaneously. `selected` issued an instruction;
  `not_selected` was eligible but another warp won. Neither is a dependency stall.
  `wait` denotes a fixed-latency dependency. Samples have no time resolution [N2].
- **Short scoreboard is a waiting consumer**, usually dependent on a shared/MIO operation,
  not a sample charged to the triggering load. MIO also includes special math/other
  operations; inspect producer/consumer registers. The separately named *Attributed Stalls*
  metric moves attribution to producers, but that column is absent here [N1,N2].
- **Not-issued Samples** specifically sample cycles when the scheduler issued nothing;
  they are not simply “this warp was not selected.” All/Not-issued can be collected in
  different passes, so do not subtract them or interpret their ratio as elapsed idle time [N1].

### Measurement: corrected scan breakdown

Scan = `K:343–421`, including inlined support helpers. Percentages below are **within scan
all-samples**, not kernel milliseconds; coefficient-source lines include address/control
instructions as well as LDS.

| Exclusive source group | Uniform | Clustered |
|---|---:|---:|
| Support placement | 24.7% | 16.8% |
| Metadata/strength source lines | 4.9% | 3.6% |
| Early-out | 4.4% | 2.9% |
| Coefficient source lines | 12.1% | 14.2% |
| Weight assembly/selects | 16.3% | 14.4% |
| 16-cell accumulation | **37.6%** | **48.2%** |

Source groups: support `K:343–369,384–390`; metadata `378–381`; early-out `392`;
coefficients `395–398`; assembly `399–407`; accumulation `415–417`.
Their SASS spans are `U:253–1545`, `C:319–1582`.

Corrected scan totals are **106,213 / 155,197 samples**, or **61.9% / 76.9%** of the kernel.
Short-scoreboard is **6.4% / 6.9% of scan samples**; actual scan LDS PCs themselves have
zero short-scoreboard samples. In clustered, the two metadata-consuming `IADD3` PCs
(`C:555,567`, 4,139 + 4,031) account for 8,170 samples; coefficient-consuming `FSEL`s
account for another 2,480 (`C:914,918`). This is evidence of load-result latency, **not evidence that the load
instruction's own source line consumed that time**.

The qualitative warning in
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11au/scan_breakdown.txt:18–32` was
right, but its **11.5% / 11.7%** scan short-scoreboard shares also count duplicate inline
PCs. Deduplicating the adjacent v11au source CSVs gives **6.4% / 7.2%**, respectively.

Across the entire v11av kernel, distinct-PC short-scoreboard represents **7.1% / 6.2%**
of all-samples. In the separate not-issued population it is **11.3% / 9.3%**, versus
**41.7% / 34.0% wait** and **18.4% / 33.0% math-pipe throttle**. These are statistical
scheduler-state distributions, not independently recoverable speedups.

### Measurement: loads versus thin-lane arithmetic

| Distinct-PC dynamic count | Uniform | Clustered |
|---|---:|---:|
| All LDS warp instructions | 50.262M (**7.48%**) | 76.581M (**8.82%**) |
| Scan LDS warp instructions | 43.950M | 74.437M |
| Accumulation FFMA warp instructions | 137.217M | 236.222M |
| Accumulation FMUL warp instructions | 68.609M | 118.111M |
| Active **and predicated-on** lanes per accumulation FFMA/FMUL | **6.426 / 32** | **4.122 / 32** |

LDS rows: `U:549–575,645–824`, `C:617–643,710–886`; accumulation arithmetic:
`U:1065–1544`, `C:1131–1582`. These last lane averages correspond to **20.1% / 12.9%**
of full-warp lane participation. This is principally divergent early-out/geometry, not
false instruction predicates: the arithmetic's active and predicated-on counts agree.
The entire-kernel report averages, 16.06/15.74 and 10.44/10.31 active/predicated-on lanes,
conceal this worse scan-math utilization (`V/*_details.txt:41–42`).

All exported **LDS excessive shared wavefront counts are zero**. The reported excess
**3.443M / 1.319M** wavefronts belongs to **STS**, principally staging stores—not scan
loads (`U:180–251`, `C:178–317`; full distinct-PC opcode aggregation). Broadcast reads
of the same shared word do not constitute bank conflicts [N5]. Zero excess does not make
loads free, but rules against a bank-conflict explanation for the scan's hot consumers.

**Further arithmetic-count inference:** the kernel executes **881.689M / 973.792M
predicated-on accumulation lane FFMAs**. Updating all `1e6 × 6³` real/imaginary
contributions needs 432M such FFMAs: **2.04× / 2.25×** as many are executed because the
fixed 16-cell body also updates zero-weight cells (`K:392–417`). Lane occupancy alone
therefore overstates *useful* mathematical work too.

## Compute utilization: high pipeline activity is not high useful FLOP/s

Measured (`V/*_details.txt:8–42`):

| Metric | Uniform | Clustered |
|---|---:|---:|
| Compute (SM) throughput | 62.51% | 80.68% |
| DRAM throughput | 7.41% | 3.59% |
| L1/TEX throughput | 51.33% | 62.47% |
| Issue-active / no-eligible cycles | 68.98% / 31.02% | 75.03% / 24.97% |
| Active / eligible warps per scheduler | 4.15 / 1.47 | 4.87 / 1.86 |

NVIDIA throughput is the **maximum normalized constituent counter**, not average utilization
of every compute pipeline, nor achieved FP32 FLOP/s divided by theoretical FMA peak [N3].
The saved sections do not identify that maximum's constituent; do not assert a specific
pipeline is saturated. Turing has four schedulers/SM, 64 FP32 cores/SM, two-cycle warp
arithmetic execution and four-cycle dependent FMA latency; INT32 and FP32 can execute
concurrently [N4]. Low lane participation and math-pipe contention can consequently coexist.

For scale only, a **derived scalar-arithmetic estimate**, not a measured pipeline metric:
`(2 × predicated-on FFMA + FMUL + FADD thread counts) / (14 × 64 × 2 × elapsed_cycles)`
is approximately **12.3% / 8.8% of all-FMA FP32 peak**. Numerators are 4.243G/3.276G
operations, including zero-weight work; elapsed cycles are 19,275,651/20,698,549. Fourteen
SMs follow from reported total/per-scheduler instructions ≈56 schedulers and [N4]. This
excludes conversions/special math and is approximate across replay passes. It illustrates
why **80.68% compute does not mean 80.68% useful FP32 arithmetic**.

**Inference:** a mixed issue/latency problem with particularly inefficient scan math is
better supported than “mostly shared-load stalls” or “DRAM bound.” SoA/packing failures
are consistent: reducing bytes or LDS instructions can lose vector metadata loads,
zero-padding simplifications, register headroom, or useful scheduling. They do not prove
memory is irrelevant (`I:726–768,811–827,1157–1165`). Three-point packing *succeeded*
for staging; it did not change the scan's lane mapping (`I:1261–1314`). Indeed, v11au and
v11av have identical **789.518M clustered scan instructions**, while whole-kernel
instructions fall 890.248M→868.614M and benchmark spread improves only 14.029→13.714 ms.

### Later sibling artifacts: arithmetic/select reductions are already tried

During this analysis, `profile-artifacts/expA-output-owned/opt-2026-10-22/v11ay/` acquired
source and A/B results. Its `exp_a_cartesian.cu:400–429` already combines strength–z
factoring (nominally eight rather than sixteen inner FMULs) with pre-padded contiguous
y/z coefficient loads that remove selects. Its `ab_v11av_v11ay.txt:1–72` records uniform
spread **12.49–12.59→10.91–10.92 ms** and clustered **13.71–13.72→11.23 ms**. These
are measurements from separate work, **not runs performed here**, and the combined A/B
does not isolate either change. They corroborate scan instruction reduction as a lever;
do not propose those changes again as untried. This is not ncu's generic “fuse non-fused
FP32” advice—the v11av accumulator updates already use FFMAs.

## Remaining ideas, ranked (unmeasured hypotheses, not promised speedups)

1. **A genuinely different scanner mapping**, including the untried 32-cells/lane option.
   Target fewer warp instructions/useful contribution, not fewer shared bytes alone.
   More cells/lane can amortize support placement but doubles accumulator state and can
   increase zero math; predict visits and register/occupancy cost first. Merely adding
   per-cell predicates does not reclaim SIMD issue slots. This is *not* the already-failed
   8-cell/lane geometry, relevance masks, or row compaction (`I:639–711,770–827`).
2. **True producer/scanner warp specialization**, only as a larger redesign. Unlike the
   rejected same-warp v11aw schedule (`I:1316–1323`), independent warps could overlap
   staging and address barrier skew. Clustered barrier-state samples are 5.9% (mostly on
   the post-barrier consumer at `C:2511`), but neither those nor staging's 7.6% are wall-time
   bounds. Extra producers reduce scanner residency; Turing lacks the sm_80 `cp.async`
   path (`I:762–764`). Lower priority than reducing scan work.
3. **A larger block tile alone is lower leverage than its name suggests.** A
   `16×16×16`, eight-warp block retaining `8³` warp regions may reduce staging rereads,
   but **does not automatically reduce scanner visits**: current clustered work is
   `4 × 2M = 8M`; even a 1M union with eight warps is still `8 × 1M = 8M`. Check that
   distinction before accepting it as a scan optimization. Current measured visit counts
   are `V/bench.txt:37–48`; source ownership is `K:660–663,687–703`.

Do not repeat SoA/stride/pair-prefetch, support bit packing, forced six-block occupancy,
load-to-shuffle replacement, per-point masks, or same-warp staging pipelining: these are
already measured rejections (`I:99–115,625–827,1316–1323`). Fix/deduplicate attribution
before using another flamegraph to choose a target; a later measurement should include the
compute-throughput breakdown and not-issued consumer/producer dependencies.

**Limits:** clocks were not fixed and collection used 16 replay passes
(`V/uniform_ncu_run.txt:1–5`, both details files:3). The ncu captures cover one spread
kernel; benchmark spread additionally includes split reduction (`K:1422–1458`). Use
`V/bench.txt:13–24,37–48` for warm performance, not profiled method-3 timings or synthetic
trace widths. No experiments were run in this investigation; later v11ay timings above
are explicitly attributed to its separately produced artifacts.

## Official NVIDIA sources

- **[N1]** [Nsight Compute source metrics and inline correlations](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#pre-defined-source-metrics),
  [Inline Functions Table](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#inline-functions-table).
- **[N2]** [Statistical sampler](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#statistical-sampler),
  [Warp stall reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons).
- **[N3]** [Throughput metric semantics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-overview).
- **[N4]** [Turing instruction scheduling](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#instruction-scheduling),
  [concurrent integer arithmetic](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#integer-arithmetic).
- **[N5]** [CUDA shared-memory banks and broadcast](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks).

## Update — idea 1 was implemented and measured (v11ax/v11ay, 2026-10-23)

The "highest-value small new test" in this analysis was implemented, together with a
second scan change that removes the z/y lane-group shift selects by storing each
coefficient row with enough zero padding to load a lane group's values from one
contiguous base.

- `v11ax_z`: contiguous z row (three zero pre-pads, three post-pads, base
  `21 + support_coefficient_offset_z - local_cell_offset_z`). 11.928 / 12.726 ms uniform /
  clustered vs 12.513 / 13.720 v11av.
- `v11ax`: v11ax_z + four real and four imaginary `strength*z` factors formed once per
  lane group, then one FMA per cell (8 rather than 16 inner FMULs). 11.043 / 11.608 ms.
  The predicted 34.3M / 59.1M instruction saving was realized: measured ncu warp
  instructions fell 671.9M -> 560.7M uniform and 868.6M -> 686.1M clustered.
- `v11ay`: the existing x post-pad doubles as the y pre-pad, so y uses the same contiguous
  base at no extra shared-memory cost. Final 4-run medians: **10.912 / 11.232 / 1.393 ms**
  spread and **14.446 / 12.834 / 1.956 ms** totals for uniform / clustered / sparse.
  Both mandatory `1e6` spread gates now pass (uniform 31.6%, clustered 29.4% faster than
  method 3). Registers/shared: 96/95, 12,696/12,936 B, 5 blocks/SM, no spills.

The "fuse non-fused FP32" warning was not the route: the accumulation already used FFMAs.
The win came from removing the per-cell weight FMULs and the `FSEL` chain, exactly as the
lane-participation and useful-FFMA analysis suggested.

Two variants were rejected for the reasons the analysis anticipated:

- `VALUES_PER_POINT=31` (all rows contiguous) breaks the 8-byte `int2` metadata load
  (`misaligned address`); `VALUES_PER_POINT=32` is aligned but aliases every row to the
  same shared bank and is slower (11.56 / 11.46 ms).
- The new profile is less instruction-bound: clustered now runs at ~79% compute and ~78%
  L1/TEX, so the next step must reduce coefficient/shared loads or cell updates, not just
  arithmetic. The fixed 16-cell body still does ~2.0-2.25x the useful cell updates.
  Idea 2 (different scanner mapping) is the remaining structural option; idea 3
  (producer/scanner warp specialization) is still a larger redesign; idea 4's caution
  about block tiles still holds.

Full evidence: [`ISSUE-009-exp-a-results.md`](ISSUE-009-exp-a-results.md) and
`profile-artifacts/expA-output-owned/opt-2026-10-22/v11ay/`.
