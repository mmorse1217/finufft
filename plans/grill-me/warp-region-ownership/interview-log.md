# Interview log: warp-owned 8×8×8 regions with 16 cells per lane

## Checkpoints

### Turn 0 — setup

- **State:** Interview in progress
- **Summary:** Initial brief created from the chat design discussion. No code
  changes made. The working recommendation is Option B (independent warp per
  region, per-warp staging, `__syncwarp`), but it is still provisional.

### Turn 1 — terminology and glossary grounding

- **State:** Interview in progress
- **Question asked by the user:** Threads vs lanes? Tile vs region? What is a
  batch? Was the design grounded in `docs/GLOSSARY.md`?
- **Answer / action:** Read `docs/GLOSSARY.md` in full. It exposes three conflicts
  in the earlier chat design:
  - **batch** is reserved for a group of stacked transforms, not a group of staged
    sources;
  - **tile** is the output region a CUDA threadblock works on, not the warp-owned
    `8³` box;
  - **halo** is explicitly avoided; **kernel support** and **padding** are the
    canonical terms.
- **Classification:** Fact (glossary definitions), Decision (terminology
  correction), Open (whether the threadblock tile may grow).
- **Implications:** Preserve the glossary's **tile** definition. Introduce local
  terms **warp**, **lane**, **warp region**, **lane cell group**, **staging
  group**, **candidate bins**, and **support overhang**. Use **thread** for
  generic CUDA execution and **lane** only for warp-relative identity. Rename
  code identifiers that use the old vocabulary (`STAGE_POINTS`, `accumulate_batch`,
  `HALO`, `NearBins`) when implementation starts.
- **New question:** With 16 cells per lane, four warps need 2048 cells, so a
  CUDA threadblock tile can no longer be `8³`. Does the threadblock tile grow to
  four `8³` warp regions (e.g. `32×8×8`), or must it stay `8³`, which forces one
  warp per threadblock?

### Turn 2 — does tile growth solve anything?

- **State:** Interview in progress
- **Question asked by the user:** "Sure we can grow it. But is that going to solve
  any problems?"
- **Answer:** Growing the threadblock tile is not itself a performance mechanism.
  It solves only the packaging problem: with 16 cells per lane, four warps need
  2048 cells, so a four-warp-region tile lets a CUDA threadblock keep four warps
  and reach ~20 warps/SM instead of the 16-warps/SM ceiling of one warp per `8³`
  threadblock. It does **not** reduce LDS/cell, remove redundant per-tile source
  scans, or remove block barriers. Those come from the `2×2×4` lane cell group and
  from per-warp candidate-bin enumeration and staging.
- **Condition:** Tile growth is safe only if the tile remains a launch container.
  If candidate-bin enumeration or staging becomes tile-wide, the larger tile
  increases the point reread factor and scan work. That is Option A's failure
  mode.
- **Residual cost:** A four-warp-region tile is coarser for future active-tile
  lists/splits; one active warp region can keep three idle warps resident. Two
  warp regions per threadblock (`16×8×8`) is a middle ground.
- **Classification:** Decision (tile may grow), Analysis/Fact (tile growth is not
  a performance mechanism), Open (number of warp regions per threadblock).
- **Implications:** Add the invariant that candidate bins and staging are per warp
  region. Make warp-regions-per-threadblock a launch-policy parameter. Resource
  math at ~90 registers favors 2 warp regions per threadblock (22 warps/SM) over
  4 (20 warps/SM); 4 matches exp_a.
- **New question:** How many warp regions should one CUDA threadblock own: 1, 2,
  or 4?

### Turn 3 — prototype readiness

- **State:** Needs confirmation
- **Question asked by the user:** "Are we able to implement a prototype or are we
  still stuck on unknowns?"
- **Answer:** Executable as a prototype with defaults. No unknown blocks the
  kernel structure. The open items are tuning knobs and acceptance thresholds:
  final warp-regions-per-threadblock, single vs double staging buffer, numeric
  success threshold, and acceptable regression on the other workloads.
- **Defaults proposed:** Option B; per-warp candidate-bin cursor, staging group,
  and `__syncwarp`; `WARP_REGIONS_PER_THREADBLOCK` as a compile-time parameter
  starting at 2; single buffer, `STAGING_GROUP_SIZE=16`; zero-filled staged kernel
  weights; 3D launch grid with warp regions packed along x; host geometry/cursor
  tests plus the existing bench reference.
- **Classification:** Assessment; defaults are a Decision pending confirmation.
- **New question:** Confirm the defaults and mark the brief `Ready for execution`,
  or name the default to change first.

### Turn 4 — implementation sketch

- **State:** Needs confirmation
- **Question asked by the user:** Sketch the design and implementation API/pseudocode
  and what will change.
- **Answer:** Added a concrete sketch to `brief.md`: header constants, device types,
  function signatures, kernel and launch pseudocode, helper pseudocode, and a
  file-by-file change list. The sketch defaults to `WARP_REGIONS_PER_THREADBLOCK=2`,
  one lane per source in a staging group, `STAGING_GROUP_SIZE=16`, zero-filled
  staged kernel weights, and per-bin `SourceRangeCursor` ranges.
- **Classification:** Decision pending confirmation; the sketch makes the defaults
  concrete without adding a new architectural unknown.
- **Implications:** The implementation can start once the defaults are confirmed.
  The only remaining shape-level knob is 2 versus 4 warp regions per threadblock;
  it is a compile-time parameter.
- **New question:** Does the sketch match the intended shape, and can the brief move
  to `Ready for execution`?

### Turn 5 — implementation handoff

- **State:** Ready for execution
- **User instruction:** Implement the design in a subagent, then profile it with
  `devel/morton_spread/profile_atomic_free_spread.py` and report the full profile
  output.
- **Mode switch:** The interview is complete. The user explicitly switched to
  implementation.
- **Handoff defaults:** Option B; two warp regions per CUDA threadblock
  (`16×8×8` tile, 64 threads); per-warp candidate-bin cursor and staging group;
  single staging buffer; `STAGING_GROUP_SIZE=16`; zero-filled staged kernel
  weights; `__syncwarp` only. The existing bench correctness matrix is the hard
  gate.
- **Next action:** Delegate the kernel implementation to the `worker` subagent,
  review its diff, build, then run the profile script.

### Turn 6 — implementation and profile result

- **State:** Needs confirmation
- **User instruction:** Implement in a subagent, profile with
  `devel/morton_spread/profile_atomic_free_spread.py`, and report the full profile
  output to support the performance assessment.
- **Implementation:** Worker subagent implemented the two-warp-region,
  64-thread warp-owned kernel. 89 registers, 4.61 KB static shared, no spills.
  The full bench correctness matrix passed 8/8 with relL2 between 8e−08 and
  8e−06; all coverage, finiteness, reference, and total checks passed.
- **Profile:** `build/gpu/traces/af_profile_20261110_170201/`.
  - uniform: 18.451 ms (old 19.471, −5.2%; exp_a 10.912, 1.69×)
  - clustered: 61.145 ms (old 26.782, +128%; exp_a 11.232, 5.44×)
  - sparse: 4.635 ms (old 5.968, −22.3%; exp_a 1.393, 3.33×)
  - dense: 20.676 ms (old 8.326, +148%)
- **Microarchitecture:** uniform LDS 128.5M → 64.0M, MIO 42.34% → 0.33%,
  barrier 24.62% → 0%, L1/TEX 96.32% → 46.19%. The LDS/barrier hypothesis is
  confirmed.
- **New bottleneck:** uniform instructions rose 734.5M → 854.6M because the
  one-lane-per-source, three-axis staging doubled the warp-level staging cost.
  Support-index math rose 169.2M → 313.8M; IMAD 43.0M → 133.8M; ISETP 60.4M →
  100.9M. The kernel is now FMA/math-pipe/issue-bound (FMA 76%, math-pipe throttle
  21.8%).
- **Structural regression:** one warp per hot region removes the old four-warp
  parallelism. Clustered occupancy 44.8% → 14.9%; dense 16.5% → 4.3%. Splits are
  required; a staging tweak cannot fix it.
- **Classification:** Fact (profile), Decision (do not adopt as-is), Open (next
  feature).
- **New question:** Keep the prototype as an experimental branch or revert, and
  which follow-up comes first: coefficient staging or splits?

### Turn 7 — both follow-ups approved; active lists are not optional

- **State:** Ready for execution (next iteration)
- **User instruction:** "Ok fine add the coefficient representation. We add active
  region lists and you made me revert it."
- **Interpretation:** Both coefficient staging and active-region lists/splits are
  in scope. The user is correcting the earlier decision to defer/revert active
  regions. A list alone is not enough; the split is the part that restores
  hot-region parallelism.
- **Classification:** Decision.
- **Plan:** Phase 1 coefficient staging (exp_a's 30-float record,
  `place_support`, `scan_one_staged_point`), validate/profile uniform and sparse.
  Phase 2 active warp-region list plus splits and partial-grid reduction,
  validate/profile clustered, dense, and the full matrix.
- **New action:** Implement Phase 1 in a worker subagent, then Phase 2; profile
  after each phase so the effects are attributable.

### Turn 8 — Phase 1 coefficient staging implemented and profiled

- **State:** Phase 1 done; Phase 2 in progress
- **Implementation:** Worker ported exp_a's 30-float coefficient record,
  `prepare_two_coefficients`, `place_support` / `place_support_nowrap`, and
  `scan_one_staged_point` into the warp-region kernel. 96 registers, 3.3 KB
  shared, no spills. Validation: 9/9 matrix, 25/25 sweep, 144 edge runs.
- **Profile:** `build/gpu/traces/af_profile_20261110_173855/`.
  - uniform: 17.004 ms (Phase 0 18.451, old 19.471, exp_a 10.912)
  - clustered: 62.939 ms (Phase 0 61.145, old 26.782, exp_a 11.232)
  - sparse: 2.183 ms (Phase 0 4.635, old 5.968, exp_a 1.393)
  - dense: 27.961 ms (Phase 0 20.676, old 8.326)
- **Instruction counts:** uniform 854.6M → 755.1M; sparse 258.9M → 109.2M;
  clustered 819.2M; dense 61.9M.
- **What changed:** coefficient staging removed most of the zero-filled staging
  cost, but `place_support` moved into the scan and support-index math is still
  33.7% of uniform instructions. Uniform is now 1.56× exp_a and sparse 1.57×.
  Clustered and dense are unchanged-to-worse because hot-region parallelism is
  still missing.
- **Classification:** Decision (Phase 1 accepted as partial), Open (Phase 2
  outcome).
- **Next action:** Implement Phase 2 active-region list plus splits.

### Turn 9 — Phase 2 implemented and profiled; combined kernel accepted

- **State:** Needs confirmation (next optimization open)
- **Implementation:** Worker added active warp-region detection, deterministic
  compaction, split-count selection, a unified 1D work-item `spread` kernel, a
  `spread_region` scan with split-stride batching, partial grids, and a
  fixed-order active-region reduction. 96 registers, 3.3 KB shared, no spills.
  Validation: 9/9 matrix, 25/25 sweep, forced split counts 1..32, clipped/edge
  grids.
- **Profile:** `build/gpu/traces/af_profile_20261110_175243/`.
  - uniform: 16.699 ms (Phase 1 17.004, old 19.471, exp_a 10.912)
  - clustered: 16.433 ms (Phase 1 62.939, old 26.782, exp_a 11.232)
  - sparse: 2.239 ms (Phase 1 2.183, old 5.968, exp_a 1.393)
  - dense: 1.938 ms (Phase 1 27.961, old 8.326)
- **Win:** first configuration that beats the old baseline on all four cases
  (uniform 1.17×, clustered 1.63×, sparse 2.67×, dense 4.30×). The split path
  restores hot-region parallelism; clustered occupancy 15.2% → 55.0%, dense
  4.4% → 38.5%.
- **Remaining gaps:** 1.53× exp_a uniform, 1.46× clustered, 1.61× sparse.
  Uniform support-index math is still 33.7% of instructions; the `place_support`
  general path is hot. Sparse pays ~0.2 ms metadata (≈9%). Partial-grid
  `cudaMalloc`/`cudaFree` churn produced a 4.17 ms dense outlier against a
  1.94 ms median.
- **Classification:** Decision (combined kernel accepted as new baseline), Open
  (next optimization).
- **New question:** Persistent workspace/compact partial grids first, or x-run
  batching first?

### Turn 10 — readability and modularity become the goal

- **State:** Needs confirmation (refactor scope open)
- **User feedback:** The performance result is acceptable, but readability went out
  the window; the goal of this refactor is code quality, readability, and
  modularity.
- **Diagnosis:** The kernel is a ~1005-line translation unit with mixed geometry,
  30-float staging layout, scan dispatch, active-region policy, reduction, and
  host allocation. `spread_region` has 12 parameters; `launch_spread` has four
  inline path branches; candidate-bin enumeration is duplicated; host and device
  concerns share one file; names still mix glossary and old terms.
- **Proposal:** Target modules `spread_geometry.cuh`, `spread_staging.cuh`,
  `spread_scan.cuh`, `spread_active_regions.{cuh,cu}`, and
  `spread_kernel_coefficients.cuh`; concrete refactors are `SupportPlacement`,
  `StagedSource` accessors, `SupportPath`, `RegionWorkItem`, `CandidateBinCursor`,
  `ActiveRegionPlan`/`SpreadWorkspace`, compact partial grids, glossary renames,
  and host tests. Sequence R0–R6 with validation after each.
- **Classification:** Decision (readability is the next goal), Open (refactor
  scope).
- **New question:** Strictly behavior-preserving first, or include the compact
  partial-grid change in the same series?

### Turn 11 — the copied exp_a design is rejected as the starting point

- **State:** Needs confirmation (concise redesign scope and tolerance)
- **User critique:** The code looks copied from `exp_a_cartesian.cu` to recover
  performance without thinking about design or modularity; function names and loop
  patterns are similar; a more concise design should exist.
- **Acknowledgment:** Correct. The current kernel is a direct port of exp_a's
  9-lane staging, 30-float padded record, window placement, fast-path vote,
  degenerate fallback, and active-tile machinery. Those were exp_a's solutions to
  its own constraints, not requirements of our one-warp-per-region design.
- **Proposal:** A concise redesign:
  - stage the six kernel values from `eval_kernel_vec_horner`, not a 9x6 table and
    a 30-float padded record;
  - one lane per source, 32 sources per staging group, all lanes active;
  - scan with a per-cell `tap_value`, removing `place_support`, window reads,
    padding, the fast-path vote, and the degenerate fallback;
  - move the active-region plan to the host, where the bin offsets already exist,
    removing the detection/compaction kernels and blocking host copy;
  - compact partial grids `[split][active_slot][local_cell]`.
- **Classification:** Decision (prototype the concise redesign first), Open
  (performance tolerance).
- **New question:** Confirm the prototype scope and tolerance: staging+scan first,
  then host plan/compact partials; no worse than 5% on uniform/clustered/sparse
  and dense at most 2.0 ms.

### Turn 12 — host active planning and the method-3 comparison

- **State:** Needs confirmation (measurement contract and prototype scope)
- **User question:** Why plan active regions on the host? Is that a dishonest
  comparison with method 3?
- **Answer:** The motivation is the host already has the bin offsets after the CPU
  sort; it is not to move work outside the measured region. But the concern is
  valid: excluding the plan from `spread` while comparing to method 3 `execute`
  would be misleading. The plan will be computed on the host **inside**
  `launch_spread`, so the `spread` interval includes the plan, upload, kernel,
  memset, and reduction. The bench will print a plan/kernel/reduce breakdown and
  the end-to-end `prep + spread` total; the kernel-only NCU number is labelled as
  microarchitecture, not the headline comparison.
- **Classification:** Decision (measurement contract), Open (whether the host plan
  is faster than the current GPU detection/compaction).
- **New question:** Confirm the measurement contract and the prototype scope.

### Turn 13 — three implementation steps, then a full reprofile

- **State:** Ready for execution
- **User instruction:** Implement each concise-redesign step in its own worker
  subagent, then reprofile and present the full profile results for review.
- **Plan:** 5a kernel-value staging + tap scan (worker 1); 5b host active plan
  measured inside `launch_spread` (worker 2); 5c compact partial grids (worker
  3). Validate and bench after each; full profile after 5c.
- **Classification:** Decision.
- **Next action:** Delegate 5a to worker 1.

### Turn 14 — step 5a tap scan rejected

- **State:** Needs confirmation (5a' revision)
- **Step 5a result:** Correct (9/9, 25/25) but outside tolerance. Uniform
  16.70 → 19.47 ms (+16.6%), clustered 16.43 → 17.72 ms (+7.8%), sparse
  2.24 → 3.13 ms (+39.7%), dense 1.94 → 2.31 ms (+19.1%).
- **Profile:** `build/gpu/traces/af_profile_20261110_185628/`. Uniform warp
  instructions 754.6M → 1,130.7M. Support-index group 45.35%; top lines are the
  tap return (27.59%), `tap +=` (17.69%), the accumulator FMAs (22.64%), and the
  start normalization (11.27%). FFMA 192.7M → 295.9M because the tap scan has no
  coverage early-out.
- **Diagnosis:** The staging simplification worked (Horner/FMA 3.8%), but the
  per-cell tap repeats support-wrap arithmetic 8× and always executes the 32-cell
  accumulation even when a lane group is outside the support. The kernel is
  issue/compute-bound (80% issue slots, 75.6% FMA, 64% ALU).
- **Decision:** Reject 5a. Keep the simple `StagedSource`/one-lane staging, but
  restore three `SupportPlacement` calls, the coverage early-out, and 2/2/4 window
  loads (5a'). Do not proceed to 5b/5c on the tap kernel.
- **Classification:** Decision (5a rejected, 5a' next), Fact (profile).
- **New question:** Approve 5a' as the next worker step?

### Turn 15 — Change A approved; reprofile after it

- **State:** Ready for execution (Change A)
- **User instruction:** Do Change A in a subagent and reprofile.
- **Scope:** Change A only: keep the simple `StagedSource`/one-lane staging,
  replace the per-cell tap scan with three `SupportPlacement` calls, a coverage
  early-out, and 2/2/4 window loads. Leave active-region detection, splits,
  full-grid partials, reduction, public API, and the bench untouched. Change B
  (host plan) and Change C (compact partials) wait until Change A is reviewed.
- **Classification:** Decision.
- **Next action:** Worker implements Change A; validate 9/9 + 25/25; full profile.

### Turn 16 — Change A profiled: partial, not accepted

- **State:** Needs confirmation (A.1 staging-group experiment or sparse trade-off)
- **Change A result:** 9/9 + 25/25. Uniform 16.70 → 18.15 ms (+8.7%),
  clustered 16.43 → 14.50 ms (−11.8%), sparse 2.24 → 2.82 ms (+25.9%),
  dense 1.94 → 1.77 ms (−8.5%). Translation unit 783 lines, down from 1005.
- **Profile:** `build/gpu/traces/af_profile_20261110_192025/`.
  - Uniform: 751.9M instructions (similar to Phase 2), but IPC 2.40 vs 2.55,
    occupancy 43.35% vs 47.73%, active threads/warp 18.45 vs 20.34. The general
    `place_support` path is hot (lines 160–187, especially 173), because the
    32-source staging group makes the non-wrapping vote fail often.
  - Sparse: 141.6M instructions vs 106.8M. Horner/FMA is 28.86% of instructions
    vs 5.65% in Phase 2. Small bins leave most lanes idle with one-lane staging;
    Phase 2's nine-lane staging used 27 lanes for three sources.
- **Assessment:** Change A recovered most of the tap regression and improved
  clustered/dense, but uniform is still outside tolerance and sparse is much
  worse. The uniform issue is the staging-group size / fast-path hit rate; the
  sparse issue is the one-lane staging parallelism for small bins.
- **Decision:** Try A.1 (`STAGING_GROUP_SIZE = 13` or 16) first. If uniform
  returns to tolerance and sparse remains the only outlier, decide explicitly
  whether to accept the sparse regression or add a parallel small-bin staging
  path. Do not proceed to B/C yet.
- **Classification:** Decision (A.1), Open (sparse trade-off).
- **New question:** Approve A.1, or accept the sparse regression and proceed?

## Decision history

### D1 — Preserve the glossary's `tile` definition; add local ownership terms

- **Status:** Provisional
- **Decision:** Keep **tile** as the CUDA threadblock's output region. Use **warp
  region** for the warp-owned `8³` subdivision, **lane cell group** for the
  `2×2×4` lane-owned box, **staging group** for the staged source group,
  **candidate bins** for the expanded bin ranges, and **support overhang** for the
  `ceil(ns/2)` per-side expansion.
- **Rationale:** `docs/GLOSSARY.md` and ISSUE-007/008/009 already use **tile** for
  the threadblock's output region. Redefining it would ripple through those
  documents. The new terms make the three ownership levels explicit.
- **Alternatives rejected:** Redefine **tile** as the warp-owned box and invent a
  new term for the threadblock's output region; use **subtile** (less descriptive
  about ownership); use **region** unqualified (undefined and overloaded).
- **Confidence:** High.

### D2 — Replace point-staging "batch" with "staging group"

- **Status:** Provisional
- **Decision:** Use **staging group** for a contiguous range of sorted sources
  whose kernel weights are written to shared memory together. Rename
  `STAGE_POINTS` / `accumulate_batch` accordingly.
- **Rationale:** `docs/GLOSSARY.md` reserves **batch** for a group of stacked
  transforms processed together in one FFT. Using it for sources is a direct
  conflict.
- **Alternatives rejected:** **point batch** (still conflicts); **chunk** (vague);
  **window** (implies a sliding window, which this is not).
- **Confidence:** High.

### D4 — Tile may grow; tile is a launch container only

- **Status:** Decided on principle; size still Open
- **Decision:** The CUDA threadblock tile may grow beyond `8³`. Candidate-bin
  enumeration and staging remain per warp region. Tile size must not leak into
  either.
- **Rationale:** Tile growth is required to keep more than one warp per
  threadblock, which avoids the 16-warps/SM ceiling of one warp per `8³`
  threadblock. It is not a source of the LDS or barrier win.
- **Alternatives rejected:** Keep the tile at `8³` and use one warp per threadblock
  (Option C; occupancy ceiling); grow the tile and use tile-wide staging (Option
  A; higher point reread factor and scan work).
- **Confidence:** High on the principle; Medium on four versus two warp regions
  per threadblock.

### D3 — Use thread for generic CUDA, lane for warp-relative identity

- **Status:** Provisional
- **Decision:** Use **thread** for launch geometry and generic CUDA execution;
  use **lane** for a thread's identity within a warp and for the `2×2×4`
  ownership box. Define **warp** and **lane** once in the plan.
- **Rationale:** The lane mapping is warp-relative; "lane cell group" is more
  precise than "thread cell group" when the warp region is the context. Standard
  CUDA/SIMT usage supports this.
- **Alternatives rejected:** Use **thread** everywhere (less precise for the
  warp-relative mapping); use **lane** everywhere (confusing for launch geometry).
- **Confidence:** Medium.

### D5 — Do not adopt the warp-region prototype as-is

- **Status:** Decided
- **Decision:** The two-warp-region prototype is a valid microarchitecture
  experiment but not an acceptable replacement for the current kernel.
- **Rationale:** It improves uniform by only 5.2% and sparse by 22.3%, while
  regressing clustered by 2.3× and dense by 2.5×. The LDS/MIO/barrier win is real,
  but staging instruction count increased enough to consume it, and hot-region
  parallelism collapsed.
- **Next alternatives:** (a) coefficient staging (6 Horner coefficients + support
  placement) to remove the 36.7% support-index and 25.4% Horner staging cost;
  (b) splits/active-region lists for clustered/dense; (c) a cheaper staging
  parallelism experiment (`STAGING_GROUP_SIZE=32` or four warp regions per
  threadblock).
- **Confidence:** High on the negative result; Medium on which follow-up should
  come first.

### D6 — Add coefficient staging and active-region lists/splits; neither is optional

- **Status:** Decided
- **Decision:** The next iteration includes both exp_a-style coefficient staging
  and active warp-region lists plus splits. Active regions are not deferred again.
- **Rationale:** The profile shows two independent failures: staging instruction
  cost (support-index 36.7% of uniform instructions) and lost hot-region
  parallelism (clustered occupancy 14.9%, dense 4.3%). Coefficient staging attacks
  the first; splits attack the second. An active list without splits does not
  restore hot-region parallelism.
- **Alternatives rejected:** Active list only (does not fix clustered/dense);
  coefficient staging only (does not fix clustered/dense); keeping the failed
  prototype as-is.
- **Confidence:** High on both being required; Medium on how much of the uniform
  gap coefficient staging alone will close.

### D7 — Accept Phase 1 as partial; proceed with Phase 2

- **Status:** Decided
- **Decision:** Keep the coefficient-staging kernel and add active-region lists
  plus splits. Do not stop at Phase 1.
- **Rationale:** Phase 1 improves uniform (18.45 → 17.00 ms) and sparse
  (4.64 → 2.18 ms), but clustered and dense are still far behind because one warp
  owns a hot region with no split. Active lists alone would not fix that; the
  split is the required mechanism.
- **Alternatives rejected:** Stop after Phase 1 (clustered/dense remain broken);
  revert active-region work (already rejected by the user).
- **Confidence:** High.

### D8 — Keep the combined coefficient-staging + active-region/split kernel

- **Status:** Decided
- **Decision:** The combined kernel is the new best atomic-free baseline. Keep it.
- **Rationale:** It beats the old baseline on every canonical case and cuts the
  worst exp_a gap from 5.44× to 1.61×. The lane-ownership geometry, coefficient
  staging, and active-region splits are all required; none is optional.
- **Alternatives rejected:** Revert to the old block-owned kernel (worse on all
  four cases); stop after Phase 1 (clustered 62.9 ms, dense 28.0 ms).
- **Confidence:** High.

### D9 — The next goal is readability and modularity

- **Status:** Decided on the goal; scope Open
- **Decision:** Refactor the combined kernel for readability and modularity.
  Performance and behavior are held constant, except for the deliberate compact
  partial-grid change if it is included.
- **Rationale:** The kernel works but is a 1005-line performance-driven accretion
  with mixed concerns. The user explicitly set readability and modularity as the
  goal of this refactor.
- **Alternatives rejected:** Continue optimizing performance first; leave the
  structure as-is because it passes the bench.
- **Confidence:** High on the goal; Medium on how much to change in the first
  series.

### D10 — Prototype the concise redesign before modularizing the copied code

- **Status:** Decided; performance tolerance Open
- **Decision:** Build the concise staging+scan design, then move the active-region
  policy to the host and compact the partial grids. The header-extraction plan is
  the fallback if the concise scan loses too much performance.
- **Rationale:** The current 1005-line kernel is exp_a's incidental complexity,
  not essential complexity. Staging kernel values directly, one lane per source
  with 32 sources, and per-cell tap placement removes the 9-lane mapping, the
  30-float padding, the window/fast-path/degenerate scan variants, and the GPU
  active-region metadata. The result should be both smaller and easier to reason
  about.
- **Alternatives rejected:** Extract the copied code into headers first (treats the
  symptom, keeps the accidental complexity); stop at the current working kernel.
- **Confidence:** High on simplicity; Medium on whether the tap scan matches the
  window scan's performance.

### D11 — Host active plan is measured inside `launch_spread`

- **Status:** Decided
- **Decision:** Compute the active-region plan on the host from the host bin
  offsets, but inside the timed `launch_spread`, so `spread` includes plan,
  upload, kernel, memset, and reduction. Report a plan/kernel/reduce breakdown.
  The headline is `prep + spread` versus method 3 `setpts + execute`; kernel-only
  NCU is microarchitecture only.
- **Rationale:** The host already owns the sorted bin table. Hiding the plan in
  prep would make the kernel-only comparison misleading. Measuring it in place
  keeps the implementation simple and the comparison honest.
- **Alternatives rejected:** Compute the plan in prep and compare kernel-only to
  method 3 execute (dishonest); keep the GPU detection/compaction purely to keep
  the timing unchanged (more code and a blocking copy).
- **Confidence:** High on the contract; Medium on whether the host plan beats the
  current GPU plan in wall time.

### D12 — Implement the concise redesign in three worker steps

- **Status:** Decided
- **Decision:** Step 5a staging+tap scan, 5b host active plan, 5c compact partial
  grids; each in a separate worker subagent; validate and bench after each; full
  profile at the end.
- **Rationale:** Keeps the changes attributable, catches performance regressions
  early, and gives a clean handoff for review.
- **Alternatives rejected:** One large worker change (hard to attribute);
  modularize the copied exp_a code first (the user rejected that direction).
- **Confidence:** High.

### D13 — Reject the tap scan; revise to window/coverage (5a')

- **Status:** Decided
- **Decision:** Keep the concise staging (`StagedSource`, one lane per source,
  32 sources, `eval_kernel_vec_horner`) but replace per-cell `tap_value` with
  three `SupportPlacement` calls, an early-out when any axis is uncovered, and
  2/2/4 window loads.
- **Rationale:** The tap scan adds ~376M integer instructions and loses the FFMA
  early-out; it is 16.6% slower on uniform and 39.7% on sparse. The window scan
  amortizes support placement over the lane group and skips empty lane groups.
- **Alternatives rejected:** Continue to 5b/5c on the tap kernel (host plan and
  compact partials cannot recover a 16–40% scan regression); revert entirely to
  the exp_a port (keeps the 30-float/9-lane/coefficient-table complexity).
- **Confidence:** High on rejecting 5a; Medium on whether 5a' hits the 5%
  tolerance without the fast-path vote.

### D14 — Change A only, then reprofile

- **Status:** Decided
- **Decision:** Implement the window/coverage scan (Change A) and profile it
  before deciding on Changes B and C.
- **Rationale:** Change A is the performance gate. If it does not restore the
  scan, the host plan and compact partials cannot recover the regression. The user
  wants to review the profile before approving the remaining changes.
- **Alternatives rejected:** Implement A+B+C in one pass (harder to attribute);
  keep the tap scan (known slower).
- **Confidence:** High on the order; Medium on whether Change A alone hits the 5%
  tolerance without the non-wrapping fast path.

### D15 — Change A is partial; try a smaller staging group before B/C

- **Status:** Decided
- **Decision:** Run A.1 with `STAGING_GROUP_SIZE = 13` or 16 and re-bench uniform
  and sparse. Do not proceed to the host plan/compact partials until the
  scan/staging trade-off is settled.
- **Rationale:** Uniform's regression is consistent with the 32-source
  non-wrapping vote failing often. Sparse's regression is consistent with
  one-lane staging being under-parallel for small bins. A smaller group may fix
  uniform but will not fix sparse; the sparse trade-off must be decided
  explicitly.
- **Alternatives rejected:** Proceed to B/C on a version that is still 8.7%
  slower on uniform and 25.9% slower on sparse.
- **Confidence:** High on the diagnosis; Medium on whether 13/16 recovers uniform.

## Unresolved questions

1. Does A.1 with `STAGING_GROUP_SIZE = 13` or 16 recover uniform to within 5%?
2. If sparse remains 25% slower, accept the one-lane staging regression for
   simplicity, or add a parallel small-bin staging path?
3. After A.1, approve Change B (host plan) and Change C (compact partials), or
   stop at the concise scan.
4. Are host tests for the pure geometry/cursor modules required in the concise
   prototype?
5. Is the F0 prototype in scope for the `CandidateBins` move?
