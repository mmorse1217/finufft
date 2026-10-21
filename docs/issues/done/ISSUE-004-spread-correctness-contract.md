# ISSUE-004: Pin down the spread correctness contract for an output-owned kernel

**Created:** 2026-10-21
**Priority:** P1
**Status:** Done
**Labels:** wayfinder:research
**Started:** 2026-10-21
**Completed:** 2026-10-21

**Related commits:**
- _None yet_

**Dependencies:**
- _None_

## Description

Document the exact semantics a new output-owned 3D type-1 spread kernel must reproduce so
its fine-grid output matches the existing cuFINUFFT path within floating-point summation
order. Cover:

- coordinate mapping from user coordinates to fine-grid indices (`fold_rescale`, periodic
  wrap), including negative, edge, and out-of-domain coordinates;
- the kernel support interval for even and odd `ns`, the `ceil(rescaled - ns/2)` start
  rule, Horner versus direct kernel evaluation, and the `sigma`/`beta` parameters;
- the fine-grid layout, padding, and exactly which cells the existing subgrid write covers,
  including the halo and the periodic edge cases;
- how strengths, `idxnupts`, the `setpts` permutation, and `ntr` batching are applied;
- how `gpu_spreadinterponly` and `modeord` affect the spread stage;
- the numerical comparison oracle: relative L2 on the fine grid, where to sample, and what
  error is expected from a different summation order.

Include file and line references in this repo. This is a fact ticket: do not propose a
final design or modify code. Write findings to
`docs/issues/research/ISSUE-004-spread-correctness-contract.md` and append a short
`## Answer` to this ticket.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
- 2026-10-21 20:20 — Resolved by a research subagent; findings in `docs/issues/research/ISSUE-004-spread-correctness-contract.md`; ticket closed.

## Answer

The detailed contract, derivations, and source/line citations are in
[`../research/ISSUE-004-spread-correctness-contract.md`](../research/ISSUE-004-spread-correctness-contract.md).
Key points a new output-owned 3D type-1 kernel must match:

- **Reference is cuFINUFFT GPU, not CPU.** Pin `gpu_method=3` (auto is method 2 for type 1),
  `gpu_spreadinterponly=1`, and the same `gpu_kerevalmeth` (`0` direct ES or `1` Horner).
  Current CPU FINUFFT uses the PSWF kernel and a different coefficient table; its output is
  not a same-kernel summation-order reference.
- **Coordinate map.** On device, `r = ((x/(2*pi)+0.5) - floor(...)) * nf` using cuFINUFFT's
  FMA plus directed subtract/multiply (`include/cufinufft/spreadinterp.hpp:57-89`). It folds
  negative and multi-period coordinates into `[0,nf)`; large magnitudes lose precision, and
  roundoff near `+pi` can produce `r=nf`.
- **Start rule and support.** `s=ceil(r-ns/2)`; the `ns` samples are `u+j`, `j=0..ns-1`, with
  `u=s-r in [-ns/2,-ns/2+1)`, and cell `s+j` is wrapped mod `nf`. For even `ns`, an
  integer-aligned `r` hits the left support endpoint; for odd `ns`, a half-integer `r` can.
  Endpoint values are evaluator-specific: CPU scalar excludes the left endpoint, CUDA direct
  zeroes both endpoints via `z^2>=1`, CUDA Horner has no endpoint test. Do not substitute an
  "equivalent" analytic kernel.
- **Kernel parameters.** cuFINUFFT chooses `ns` from `ceil(-log10(eps/10))` at sigma 2 (or the
  nonstandard-sigma formula), clips to `MAX_NSPREAD<T>`, and sets
  `beta/ns=2.30` (2.20/2.26/2.38 for ns=2/3/4; `0.97*pi*(1-1/(2*sigma))` otherwise). Derived
  `es_c=(2/ns)^2`. For ns=6, sigma=2: `beta=13.8`, `es_c=1/9`. Horner tables exist only for
  exact sigma 2.0/1.25; direct ES supports other sigma.
- **Fine-grid layout.** Interleaved complex, x fastest, then y, then z:
  `fw[ix + nf1*(iy + nf2*iz)]`. Normal type-1 uses `nf=next235(max(ceil(sigma*N),2*ns),2)`;
  spread-only uses `nf=N` exactly and skips FFT/deconvolution.
- **OD tile and halo.** Method 3 pads each bin by `h=ceil(ns/2)` on each side; local tile is
  `binsize+2h` per dimension. Allocated global footprint is
  `[offset-h, offset+binsize-1+h]`. The used range for points in the bin is
  `[offset-floor(ns/2), offset+binsize-1+ceil(ns/2)]`. For even `ns` this equals the footprint;
  for odd `ns` the leftmost allocated cell is never written (local index 0 stays zero) and the
  right used edge is the far edge. OD zeroes the local tile but relies on `execute_type1` to
  zero global `fw` before atomically merging the whole tile (`src/cuda/execute.cu:135`).
- **Strengths, permutation, batches.** `setpts` stores raw point pointers and builds
  `idxnupts` as the bin-sort inverse permutation; the kernel reads `c[idxnupts[q]]` with the
  point coordinates at that same original index. `ntransf` is the slowest dimension:
  transform `t` uses `c+t*M` and `fw+t*nf`; `idxnupts` is reused for all transforms and
  batches. Sorting changes only summation order.
- **`gpu_spreadinterponly` / `modeord`.** Spread-only sets `nf=N`, zeroes the output, spreads,
  and skips FFT/deconvolution; raw grid order is natural x-fastest. `modeord` only affects
  the later deconvolution mapping (`execute.cu:19-63`), so it has no effect on the spread
  stage or spread-only output. If comparing full transforms, use the same `modeord` or apply
  the documented cyclic shift.
- **Oracle.** Use a pinned cuFINUFFT type-1 spread-only run and compare all
  `nf1*nf2*nf3` cells with
  `E_rel = ||X_new - R_ref||_2 / ||R_ref||_2` (`test/utils/norms.hpp:21-37`). Calibrate the
  threshold from the reference's own noise: repeated method-3 runs and method-3 vs method-1
  spread-only runs (same kernel/fold, different atomic order). Pure reordering should be at
  roundoff scale: roughly `1e-7..1e-5` relative L2 for FP32 and `1e-16..1e-14` for FP64,
  growing like `eps_mach*sqrt(K)` with `K` contributions per cell. A difference near the
  requested NUFFT tolerance is a semantic bug, not summation order.
- **Edge data to test.** Negative/out-of-domain points, exact `+/-pi` and the seam, exact
  grid-aligned points, repeated coordinates, clustered points, empty/sparse regions, and at
  least one odd-`ns` case to expose the unused left halo cell.
