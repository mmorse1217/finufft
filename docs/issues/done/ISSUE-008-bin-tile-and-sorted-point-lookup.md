# ISSUE-008: Choose the bin size, tile size, and sorted-point lookup

**Created:** 2026-10-21
**Priority:** P1
**Status:** Done
**Labels:** wayfinder:grilling
**Started:** 2026-10-21
**Completed:** 2026-10-22

**Related commits:**
- _None yet_

**Dependencies:**
- ISSUE-003
- ISSUE-004
- ISSUE-005
- ISSUE-006
- ISSUE-007

## Description

Using the atomic inventory, the correctness contract, and the radix-sort findings, settle
the prototype's spatial parameters with the human:

- bin size per dimension, and whether bins are cubic;
- tile size, cells per thread, and CUDA threadblock geometry;
- Morton key width and bit interleaving, including how many bits per dimension are needed
  and how periodic wrapping maps to keys;
- lookup of the sorted points for a bin: the first prototype uses the dense table mapping
  each bin to the start offset and count of its points, behind a small interface so a
  sparse backend can replace it later. A sparse backend is deferred; do not commit to
  binary search. Design a better sparse data structure after the prototype;
- how a tile enumerates the bins whose points' kernel support can reach it (the tile
  expanded by `ceil(ns/2)` in each dimension), including periodic wrap, and how the
  tile-to-bin enumeration and support overlap can avoid data-dependent branches (compare a
  fixed expanded-bin box with masks against a bounded overlap loop);
- whether the first prototype uses one CUDA threadblock per tile (the tile is the unit of
  output-cell ownership; threads within the CUDA threadblock partition its cells) with no
  compaction of tiles that receive no contributions, as the sketch recommends, or compacts
  from the start;
- benchmark fine-grid sizes for the mandatory cases (recommend 64³ dense and 128³ sparse,
  matching the existing method-3 profile), plus optional 256³/512³ for lookup scaling;
  1024³ was illustrative, not a test case;
- whether to base the output-owned kernel on method 3's output-driven structure or on
  method 4's ghost-bin division — method 4 already gives each outer bin exactly one owner
  and duplicates ghost index entries, so it is the closest existing starting point;
- how the bin width relates to the tile width and the kernel support, and what average
  number of tiles that read a point the choice implies.

Record the choices, rejected alternatives, and the resulting parameters in `## Answer`.
Graduate any load-balancing question that becomes sharp into a new ticket or the map's Not
yet specified section.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
- 2026-10-21 21:37 — Claimed by the wayfinder session to choose the source-bin, output-tile, and run-lookup geometry.
- 2026-10-21 22:43 — Title and body terms revised to bin/tile/local subgrid/CUDA threadblock; removed source bin, output tile, and occupied-bin run.
- 2026-10-21 22:52 — Clarified one CUDA threadblock per tile as output-cell ownership; added modular lookup, benchmark fine-grid, and branch-light requirements.
- 2026-10-22 14:35 — Lookup decision refined: dense backend for the first prototype; sparse backend deferred; binary search is not the chosen fallback.
- 2026-10-22 15:04 — Resolved with the human; geometry recorded in Answer; ticket closed.

## Answer

Settled with the human.

**Ownership model.** Use method 4's exact division of the fine grid into outer bins (its
output regions) as the ownership model: each output cell belongs to exactly one tile, and
one CUDA threadblock owns each tile. Do not copy method 4's ghost index entries; read the
sorted points for the needed bins directly. Method 3 remains the benchmark.

**Sizes.** `binsize = 8` per dimension. Tile `16×16×8`; one CUDA threadblock per tile; 256
threads; 8 cells per thread. Clamp partial edge tiles instead of padding the fine grid.
Optional sweep: `8³` and `16³` tiles.

**Morton key.** 64-bit key, x-lowest bit interleaving. `bits = ceil(log2(max nbins_i))`,
`end_bit = 3·bits`. Compute bin coordinates modulo `nbins_i` before keying, so periodic
wrapping maps to the existing bin on the other side. No ghost copies.

**Lookup.** The first prototype uses the dense table mapping each bin to the start offset
and count of its points, behind a small interface so a sparse backend can replace it later.
The sparse backend is deferred; binary search is not the chosen fallback. Design a better
sparse data structure after the prototype.

**Kernel support boundary.** Expand each tile by `ceil(ns/2) = 3` cells; enumerate the bins
intersecting that region. Because `binsize = 8 > 3`, one adjacent bin layer is enough. The
support test rejects points in those bins that do not actually reach the tile. Keep the
tile-to-bin enumeration and support overlap branch-light where practical: compare a fixed
expanded-bin box with masks against a bounded overlap loop; use the Horner kernel path to
avoid the direct ES endpoint branch.

**Benchmark fine grids.** Mandatory: `64³` dense (`N=1e6`) and `128³` sparse (`N=1e5`),
matching the existing method-3 profile. Optional: `256³` and `512³` for lookup scaling.
`1024³` was illustrative, not a test case.

**Rejected alternatives.** Method 3's padded local subgrid as the ownership model (it
writes overlapping tiles and needs a global merge); ghost index copies; committing to
binary search for the sparse lookup; treating `1024³` as a benchmark case.
