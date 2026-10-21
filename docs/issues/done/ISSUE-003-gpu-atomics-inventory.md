# ISSUE-003: Inventory cuFINUFFT's GPU atomic and bin-sort costs

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

Produce a code-level inventory of the current cuFINUFFT GPU spread paths (`gpu_method`
1, 2, 3, and 4) and their preprocessing. For each path, identify:

- every atomic operation and its scope (global or shared), with the unit it is charged to
  (per nonuniform point, per point-cell interaction, per subgrid cell, per bin);
- the binning and sorting scheme, including the global per-bin atomic histogram in
  `calc_bin_size_noghost` / `calc_inverse_of_global_sort_idx` and the ghost-bin variant;
- the subproblem construction and the local-to-global merge, including exactly which cells
  the merge covers and why it needs atomics;
- where the design sketch's three changes (Morton key, sort-based binning, output-cell
  ownership) would replace existing work;
- any place where the current code already approximates output-cell ownership (for example
  method 3's per-bin shared subgrid or method 4's ghost bins) and what prevents it from
  being atomic-free;
- which costs are per-point versus per-interaction, and which are preprocessing versus
  spread.

Include file and line references in this repo. This is a fact ticket: do not propose a
final design or modify code. Write findings to
`docs/issues/research/ISSUE-003-gpu-atomics-inventory.md` and append a short `## Answer` to
this ticket.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
- 2026-10-21 20:20 — Resolved by a research subagent; findings in `docs/issues/research/ISSUE-003-gpu-atomics-inventory.md`; ticket closed.

## Answer

Full inventory with file:line citations:
[research/ISSUE-003-gpu-atomics-inventory.md](../research/ISSUE-003-gpu-atomics-inventory.md)
(source reading at commit `ee272845`; no build or profile was run for this ticket).

**Every atomic site (8).**
- Preprocessing: 1 global `atomicAdd(int)` per NU point per `setpts` — `calc_bin_size_noghost`,
  `include/cufinufft/spreadinterp.hpp:308` (method 1 when `gpu_sort=1`, always for methods 2/3:
  `src/cuda/spread_nupts_driven_inst.cu:111-114`, `src/cuda/spread_subprob_inst.cu:144-147`); method 4
  ghost variant `locate_nupts_to_bins_ghost`, `src/cuda/spread_blockgather_inst.cu:140`.
- Spread, per point–cell interaction: method 1 `ns^dim` global complex atomics per point
  (`spread_nupts_driven_inst.cu:35/43/56`); method 2 `ns^dim` shared (`spread_subprob_inst.cu:57/67/83`);
  method 3 none (non-atomic shared RMW at `spread_output_driven_inst.cu:92`); method 4 up to `ns^3`
  shared per point entry (`spread_blockgather_inst.cu:255`).
- Merge, per tile cell: methods 2/3 write the whole padded tile
  `V_bin = Π(binsize_i + 2·ceil(ns/2))` (`spread_subprob_inst.cu:95`, `spread_output_driven_inst.cu:101`);
  method 4 writes exactly `Π obin_i` (`spread_blockgather_inst.cu:273`).
- A complex shared atomic is always 2 `atomicAdd_block`; a complex global atomic is 1
  `atomicAdd(float2)` only for FP32 on sm≥90, else 2 scalar atomics (`include/cufinufft/utils.hpp:78-102`),
  so on the sm_75 T1000 every complex atomic is 2 hardware atomics. No atomic is charged per bin;
  per-bin work (memset, scan, `calc_subprob`, `map_b_into_subprob`) is non-atomic.

**Binning/sorting.** All paths use a row-major (method 4: padded) bin index → global per-bin atomic
histogram → `thrust` scan → non-atomic inverse-permutation scatter. `calc_inverse_of_global_sort_idx`
contains no atomic: it consumes the rank the histogram returned, and recomputes the bin index
(`spreadinterp.hpp:215-229, 307, 321`). Method 4 copies ghost-bin counts (`blockgather:45-75`) and then
copies ghost *index* entries with a box code (`:77-118`), so its sorted array holds up to `2^dim·M`
entries (default `obin/bin=2`). Method 1 with `gpu_sort=0` is the only path with zero preprocessing
atomics (identity permutation, `spread_nupts_driven_inst.cu:126-128`).

**Why the merge needs atomics.** Methods 2/3 write a padded tile whose `ceil(ns/2)` collar belongs to
neighbouring bins, so a cell is written by up to `2^dim` blocks; and when a bin has more than
`maxsubprobsize` (default 1024) points, several blocks of that bin write the same tile. Method 4's
outer bins already partition the fine grid exactly (`nf % obinsize == 0` enforced,
`blockgather:332-341`), taps are clipped to the tile (`:243-244`), and the write-back covers exactly
`Π obin_i` with no halo (`:264-273`), so its global atomics exist only because multiple subproblem
blocks of one outer bin each write the whole tile; its shared atomics exist because the point loop is
point-parallel (`:215`).

**Existing approximations of output-cell ownership.** Method 4 is the closest: each output cell already
has exactly one owning outer bin, and halo contributions arrive as duplicated ghost index entries (the
sketch's "duplicate boundary-bin descriptors" option) rather than as writes outside the tile. Method 3
already removes shared atomics: local accumulation is a non-atomic RMW, paid for by serialising points
with one block barrier per point (`spread_output_driven_inst.cu:71, 92, 94`). Neither owns the global
write: `fw` is zeroed (`src/cuda/execute.cu:135`) and every path adds into it; method 3 writes a padded
tile and method 4 writes each outer-bin tile once per subproblem block.

**Sketch replacement map.** Morton key replaces the twice-computed bin key; sort-based binning replaces
the histogram + `nbins` memset + scan + `sortidx`/inverse permutation (and, for method 4, the
materialised ghost index lists); output-cell ownership replaces the global merges (`spread_subprob_inst.cu:95`,
`spread_output_driven_inst.cu:101`, `spread_blockgather_inst.cu:273`, ultimately method 1's per-interaction
scatter). `fold_rescale`, kernel evaluation, FFT/deconvolution, and a dense-bin work split are not replaced.

**Per-point vs per-interaction; preprocessing vs spread.** Preprocessing is once per `setpts` (reused
across `ntransf` and repeated `execute`): 1 global int atomic per point, 2 bin-key computations per point,
O(nbins) scan/memset, and one host sync per `setpts` in methods 2/3/4 (`spread_subprob_inst.cu:168-174`,
`spread_blockgather_inst.cu:432-438`). Spread is per transform (`blksize` launches): method 1 `M·ns^dim`
global complex atomics; methods 2/3 `V_bin` global complex atomics per subproblem; method 4 `Π obin_i`
per subproblem; methods 2/4 also `ns^dim` shared atomics per point interaction; method 3 `ns^dim`
non-atomic shared RMWs plus one barrier per point. Global-merge cost per point is
`V_tile / points_per_block` — minimal for dense bins, up to `V_tile` for a nearly empty bin.
