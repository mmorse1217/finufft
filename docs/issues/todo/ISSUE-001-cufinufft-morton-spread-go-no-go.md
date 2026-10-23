# ISSUE-001: cuFINUFFT Morton output-owned spread go/no-go (wayfinder map)

**Created:** 2026-10-21
**Priority:** P1
**Status:** Todo
**Labels:** wayfinder:map
**Started:** _not started_
**Completed:** _not completed_

**Related commits:**
- _None yet_

**Dependencies:**
- _None_

## Description

Wayfinder map for deciding whether Morton-sorted, output-cell-owned spreading should
become a cuFINUFFT GPU method for 3D type 1, and for settling the integration design if it
wins. The user's design sketch is in
[destination_owned_nufft_design.md](../assets/ISSUE-001-destination-owned-nufft-design.md).

## Destination

A measured go/no-go, on this machine's Quadro T1000, on whether Morton-sorted
output-cell-owned spreading beats cuFINUFFT `gpu_method=3` on 3D type-1 spreading at the
agreed success threshold — plus a settled integration design (including the type-2 inverse
and FP64 scope) if the answer is go.

## Notes

**Domain.** Use `docs/GLOSSARY.md`; `docs/agents/domain.md` points there. The canonical term
is **output-cell ownership** (short form **output-owned**), settled in
[Settle the ownership and Morton vocabulary](../done/ISSUE-007-ownership-and-morton-vocabulary.md);
the design sketch's "destination-owned" is a historical synonym, and "destination" remains
reserved for wayfinder maps.

**Skills.** `research` for the fact tickets; `prototype` for the two experiments;
`grilling` plus `domain-modeling` for the decision tickets.

**Environment.** Quadro T1000, sm_75, 4 GB; driver 580.178.04. CUDA 12.8 toolkit at
`~/cuda-12.8` (`nvcc` 12.8.61); Nsight Systems 2024.6.2 and Nsight Compute 2025.1.0 under
`~/cuda-12.8`. `nvidia-smi` sees the GPU even though `NVIDIA_VISIBLE_DEVICES=void`;
profiling permission is available. The CMake `gpu` preset builds `cufinufft` with
`CMAKE_CUDA_ARCHITECTURES=native`.

**Baseline.** Pin `gpu_method=3` explicitly — auto is method 2 for type 1 — and compare in
`gpu_spreadinterponly=1` mode with matching `gpu_kerevalmeth`; methods 1 and 4 are context,
and method 4 is the closest existing approximation to output-cell ownership. Initial
prototype scope is 3D type 1, FP32, fixed `ns=6`, no FFT/deconvolution, compared on the fine
grid. The success protocol — gate, workloads, measurement rule, and no-go conditions — is
settled in
[Settle the prototype success protocol and scope](../done/ISSUE-006-prototype-success-protocol.md).

**Standing preferences.** Plan, don't do: tickets resolve decisions; the baseline task and
the two prototype tickets are the only places that build something, and they build the
smallest artifact that answers their question. Do not modify library code before the
go/no-go. Keep the tracker and research notes local (the user asked for `docs/issues/` to
stay untracked).

**Research storage.** Findings live under `docs/issues/research/` and are linked from each
research ticket; they are not committed or pushed, matching the local-only tracker.

### Log

- 2026-10-21 20:14 — Map charted from the user's goal and design sketch; 11 child tickets
  created, 3 research tickets dispatched.
- 2026-10-21 20:20 — The three research tickets (atomic inventory, correctness contract,
  radix sort) resolved; their gists are in Decisions so far.
- 2026-10-21 21:18 — *Settle the prototype success protocol and scope* resolved with the
  human; gate, workloads, measurement rule, and no-go conditions recorded in the ticket.
- 2026-10-21 21:35 — *Settle the ownership and Morton vocabulary* resolved; glossary updated
  and the term dispute closed.
- 2026-10-21 22:43 — Vocabulary corrected: bin = point grouping, tile = output region,
  local subgrid = shared-memory buffer, CUDA threadblock; removed source bin, output tile,
  and occupied-bin run.
- 2026-10-22 15:04 — *Choose the bin size, tile size, and sorted-point lookup* resolved;
  geometry and lookup choices recorded in the ticket.
- 2026-10-22 15:45 — *Establish the cuFINUFFT CUDA baseline and profiling protocol*
  resolved; measured method-3 baseline recorded.
- 2026-10-22 15:50 — *Prototype output-cell ownership with Cartesian bins* resolved;
  Experiment A is correct but 2.2-13x slower than method 3 (instruction-bound gather,
  clustered load imbalance). Recommendation: do not run Experiment B as scoped.
- 2026-10-22 17:30 — Experiment A revised after an nsys/ncu-driven rewrite (register Horner
  coefficients, batched coalesced raw loads, branchless `dim_support`, single early-out):
  73.0 -> 24.4 ms uniform (1.53x slower), 209 -> 69.7 ms clustered (4.38x, imbalance),
  6.8 -> 3.0 ms sparse (1.06x). Still fails the gate; the uniform gap is now per-point
  candidate work, the clustered gap is load balancing. See the v3 revision in the results
  note and `profile-artifacts/expA-output-owned/v3/`.
- 2026-10-22 17:34 — Experiment A v8: two-point ILP scan plus split warp regions with partial
  grids and a reduction for clustered inputs. Uniform 1.36x slower than method 3, clustered
  1.66x slower (was 4.38x), sparse 1.11x faster. The clustered load-balance regression is
  fixed; the gate is still missed on both mandatory `1e6` cases. Artifacts:
  `profile-artifacts/expA-output-owned/v8/`.
- 2026-10-22 18:05 — Experiment A v10: double-buffered staging, contiguous x-run batching
  (enables `qb=1`, reread 5.36) and a radix-sort preprocessing option. Uniform spread 16.7 ms
  (method 3: 15.95, i.e. parity), sparse 1.18x faster total, clustered 1.63x slower (split-8).
  Gate still fails: uniform total 1.37x because preprocessing is 6.2 ms; even free
  preprocessing would only reach parity. Fold-broadcast measured +31% and was discarded.
- 2026-10-22 18:12 — Experiment A reopened at the user's direction: the working session was
  closed by accident, not the issue. The ticket is back in `in-progress/` and its earlier
  no-go Answer sections remain as history. Fresh measurements confirm two of three
  distributions are competitive on the spread stage (uniform at parity, sparse 1.37x faster);
  clustered is 1.62x slower and the uniform total is still 1.37x because preprocessing is
  6.2 ms. The no-go is withdrawn pending further A work.

- 2026-10-23 — Experiment A completed successfully with **v11av** (`ee278ffa`):
  uniform `1e6` spread/total 12.58/16.10 ms (both gates pass), clustered 13.71/15.31 ms
  (total 8.8% faster than method 3; spread 13.7% faster, short of the 20% stretch), sparse
  1.53/2.09 ms. No mandatory workload regressed. Experiment B (Morton ordering) is now
  unblocked; clustered-spread headroom is a follow-up performance target.

## Decisions so far

<!-- one line per closed ticket: [name](link) — gist -->

- [Inventory cuFINUFFT's GPU atomic and bin-sort costs](../done/ISSUE-003-gpu-atomics-inventory.md):
  Method 3 already avoids shared atomics; the remaining atomic costs are the per-point bin
  histogram and the per-tile global merge. Method 4 already divides the output into exact outer bins
  with ghost index copies.
- [Pin down the spread correctness contract for an output-owned kernel](../done/ISSUE-004-spread-correctness-contract.md):
  Only pinned cuFINUFFT GPU spread-only output is a valid oracle (auto is method 2); match
  `fold_rescale`, kernel support, kernel evaluation, padding, and wrap, then compare
  relative L2 against a reordering-calibrated noise floor.
- [Verify CCCL radix-sort support for Morton keys on sm_75](../done/ISSUE-005-cccl-radix-sort-sm75.md):
  CCCL 3.4.2 provides stable `cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>`; use
  `DoubleBuffer`, set `end_bit` to the used Morton bits, and query temporary storage
  host-side.
- [Settle the prototype success protocol and scope](../done/ISSUE-006-prototype-success-protocol.md):
  Method 3 spread-stage is the gate (≥20% faster on both mandatory `N=1e6` workloads, no
  >5% regression, preprocessing+spread within 5%); A then B; type-1 FP32; T1000 evidence
  with a recorded generalization caveat.
- [Settle the ownership and Morton vocabulary](../done/ISSUE-007-ownership-and-morton-vocabulary.md):
  Canonical terms are bin = point grouping, tile = output region, local subgrid =
  shared-memory buffer holding the tile, and CUDA threadblock; Morton sort is distinct from
  bin sort. The glossary was revised after the first vocabulary pass.
- [Choose the bin size, tile size, and sorted-point lookup](../done/ISSUE-008-bin-tile-and-sorted-point-lookup.md):
  Method 4's exact outer-bin ownership; `binsize=8`, tile `16×16×8`, one CUDA threadblock
  per tile, 256 threads; 64-bit Morton key; dense lookup for the prototype; support
  expansion `ceil(ns/2)=3`.
- [Establish the cuFINUFFT CUDA baseline and profiling protocol](../done/ISSUE-002-cufinufft-cuda-baseline.md):
  Method 3 spread-only on the T1000: 18.67 ms uniform / 15.75 ms clustered at `M=1e6`,
  2.79 ms sparse at `M=1e5`; 25% occupancy from the 65.15 KB local subgrid; zero shared
  atomics; 90–180 MB global `RED` merge traffic.

- [Prototype output-cell ownership with Cartesian bins](../done/ISSUE-009-output-cell-ownership-cartesian-prototype.md):
  Experiment A succeeded with v11av. Uniform spread/total pass both gates; clustered total
  is 8.8% faster than method 3 with spread 13.7% faster; sparse is substantially ahead; no
  mandatory regression. Experiment B is unblocked.

## Not yet specified

- How the winning design extends to the type-2/interpolation direction, and what changes
  in the ownership/staging scheme.
- How and when FP64 follows FP32; atomic throughput and register pressure differ.
- How to handle sparse tiles, overloaded tiles, compaction of tiles that receive no
  contributions, and Morton-prefix subdivision. Experiment A now supplies the load-balance
  data: on clustered `1e6` only 64 of 512 warp regions were active; v8 addressed this by
  splitting each tile into `nsplit` partial grids (4.38x -> 1.66x regression). Whether
  active-tile compaction/work stealing and a gather redesign (16+ cells per thread, lane-level
  candidate filtering) are prerequisites for any further gather experiment is still to be
  settled.
- Small-N policy and method-selection heuristics: when the new path beats methods 1/3/4 and
  when it should fall back.
- Register/shared-memory tuning, point staging, double-buffering, and whether `cp.async`
  helps on sm_75.
- Whether the preprocessing (Morton sort + the sorted points for each bin) should be
  shared with type 2 and other methods.
- Design a sparse bin lookup backend for large or sparse grids; the first prototype uses
  the dense table, and binary search is not the chosen fallback.

## Out of scope

- CPU FINUFFT spreading and CPU performance.
- cuFFT/FFT and deconvolution optimization, except as needed to validate the spread output.
- Type 3 transforms, 1D/2D prototypes, and any non-CUDA backend (Vulkan, OpenCL, HIP,
  SYCL).
- Multi-GPU or distributed execution.
- Public API/ABI changes beyond what the integration design requires.
- Modifying or retiring methods 1-4 before the go/no-go; they remain baselines and
  fallbacks.
- Production implementation of the new method before the go decision.
