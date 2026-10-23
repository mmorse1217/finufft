# v11av: three-point packing in block-shared staging

Current best. Extends v11au's per-warp three-point packing to the block-shared clustered
staging path. The scan and per-warp kernels are otherwise unchanged.

## What changed

- `stage_block_quarter` now uses 9 lanes per point / 3 points per warp, with the same
  two-coefficient split as the per-warp path:
  - index `k` from the register-resident KER6 set,
  - index `k+3` from the shared KER6 copy.
- The block kernel gets a shared `second_kernel_coefficients[9][6]` table (216 B).
- Block-shared kernel: **94 registers, 12,760 B shared, 5 blocks/SM, no spills**.
- Each warp stages a 14-point quarter of the 56-point block batch in five 3-point
  iterations instead of fourteen one-point iterations.

## Results

Medians of 3 process runs, 5 timed runs each:

| workload | v11au pre | v11av pre | v11au spread | v11av spread | v11au total | v11av total |
|---|---:|---:|---:|---:|---:|---:|
| uniform `1e6` `64^3` qb=1 radix | 3.086 | 3.082 | 12.591 | 12.575 | 16.121 | 16.101 |
| clustered `1e6` `64^3` qb=4 counting | 1.462 | 1.467 | 14.029 | **13.714** | 15.624 | **15.311** |
| sparse `1e5` `128^3` qb=4 radix | 0.284 | 0.284 | 1.534 | 1.534 | 2.090 | 2.092 |

- Uniform spread/total gates still pass.
- Clustered total is faster than method 3; clustered spread is now **13.714 vs 12.72 ms**
  (gap ~0.99 ms).
- `relL2`: 5.49e-7 uniform, 2.0-2.3e-7 clustered, 6.76e-8 sparse.

## Flamegraphs

- `uniform_qb1_flamegraph_chrome.json`
- `clustered_qb4_flamegraph_chrome.json`

The uniform/clustered flamegraphs and call trees were regenerated from the existing
CSV captures after fixing duplicate inline-PC counting and the missing coefficient
callee. Each root now accounts for 100% of its deduplicated samples:

| capture | unique samples | duplicate PC rows removed | supplied ncu duration |
|---|---:|---:|---:|
| uniform | 171,630 | 70 | 12,600.576 µs |
| clustered | 201,895 | 92 | 13,530.144 µs |

Corrected clustered source shares:

- `scan_one_staged_point`, including support helpers: **76.9%**
- `stage_block_quarter`, including raw/coefficient staging: **7.6%**

These are sample-attribution shares, not wall-clock time or Amdahl bounds. The old
4.7% staging share omitted `prepare_two_coefficients`; the claim that software
pipelining could hide at most 4.7% is withdrawn. The converter uses this directory's
source snapshot, with explicit synthetic-layout labels and separate all/not-issued
stall tooltips. No kernel or benchmark measurements changed.
