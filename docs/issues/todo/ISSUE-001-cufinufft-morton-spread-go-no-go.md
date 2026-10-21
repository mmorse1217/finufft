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

**Domain.** Use `docs/GLOSSARY.md`; `docs/agents/domain.md` points there. The design sketch
calls the ownership rule "destination-owned", but the glossary reserves "destination" for
wayfinder maps (under `target`), so this map uses "output-owned" until
[Settle the ownership and Morton vocabulary](ISSUE-007-ownership-and-morton-vocabulary.md)
resolves the term.

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
grid. The sketch's success bar is a consistent >20% spread-stage speedup with no major
pathological regressions.

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

## Decisions so far

<!-- one line per closed ticket: [name](link) — gist -->

- [Inventory cuFINUFFT's GPU atomic and bin-sort costs](../done/ISSUE-003-gpu-atomics-inventory.md):
  Method 3 already avoids shared atomics; the remaining atomic costs are the per-point bin
  histogram and the per-tile global merge. Method 4 already partitions output bins exactly
  with ghost index copies.
- [Pin down the spread correctness contract for an output-owned kernel](../done/ISSUE-004-spread-correctness-contract.md):
  Only pinned cuFINUFFT GPU spread-only output is a valid oracle (auto is method 2); match
  `fold_rescale`, support, kernel, halo, and wrap, then compare relative L2 against a
  reordering-calibrated noise floor.
- [Verify CCCL radix-sort support for Morton keys on sm_75](../done/ISSUE-005-cccl-radix-sort-sm75.md):
  CCCL 3.4.2 provides stable `cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>`; use
  `DoubleBuffer`, set `end_bit` to the used Morton bits, and query temporary storage
  host-side.

## Not yet specified

- How the winning design extends to the type-2/interpolation direction, and what changes
  in the ownership/staging scheme.
- How and when FP64 follows FP32; atomic throughput and register pressure differ.
- How to handle sparse output tiles, overloaded tiles, active-tile compaction, and
  Morton-prefix subdivision; wait for prototype occupancy/load-balance data.
- Small-N policy and method-selection heuristics: when the new path beats methods 1/3/4 and
  when it should fall back.
- Register/shared-memory tuning, point staging, double-buffering, and whether `cp.async`
  helps on sm_75.
- Whether the preprocessing (Morton sort + occupied-bin runs) should be shared with type 2
  and other methods.

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
