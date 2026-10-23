# v11ay: contiguous coefficient rows + strength-scaled accumulation

Current best. This is the v11av kernel with the scan's unused cell-update arithmetic and
z-shift select chain removed. Uniform, clustered, and sparse all improve; both mandatory
`1e6` spread gates now pass.

## What changed

The profile analysis in
[`docs/issues/research/ISSUE-009-v11av-profile-analysis.md`](../../../../docs/issues/research/ISSUE-009-v11av-profile-analysis.md)
showed that the scan, not shared-memory bandwidth, was the remaining cost: only about
6.4 / 4.1 lanes per accumulation instruction were active, and the 16-cell body performed
one `xy*z` multiply plus two FFMAs per cell. Three source changes target that directly.

1. **Contiguous z coefficient load (no z shift selects).** The z row is now bracketed by
   three zero pre-pads (`18,19,20`) and the existing three post-pads (`27,28,29`). The four
   z kernel values for a lane group are four contiguous slots at

   `21 + support_coefficient_offset_z - local_cell_offset_z`.

   This replaces the nested `local_cell_offset_z` ternary chain (the old `FSEL`/`ISETP`
   block) with four scalar `LDS` and one base add. `VALUES_PER_POINT` grows from 28 to 30.

2. **Free y coefficient load.** The single x post-pad at slot 11 is also the y pre-pad.
   y therefore uses the same contiguous-base trick at
   `12 + support_coefficient_offset_y - local_cell_offset_y` with no extra shared memory.
   This removes the two y `FSEL`s.

3. **Strength-scaled z accumulation.** The scan now forms four real and four imaginary
   `strength*z` factors once per lane group and performs one FMA per cell:

   ```
   for z: scaled_z[z] = (strength_real*z[z], strength_imag*z[z])
   for z,y,x: acc[cell].x += xy[xy] * scaled_z[z].x
              acc[cell].y += xy[xy] * scaled_z[z].y
   ```

   That is 8 inner `FMUL`s instead of 16, and 32 FFMAs as before: the 16-cell weight /
   accumulate sequence drops from 48 to 40 warp instructions per scan. The FP32
   multiplication association changes (`(s*z)*xy` instead of `s*(xy*z)`); it is not the
   polynomial and stays inside the FP32 reordering noise floor.

Because `VALUES_PER_POINT` is now 30, the maximum 5-block batches shrink from 14/56 to
**13/53**. Shared memory is 12,696 B (per-warp) and 12,936 B (block-shared), still **5
blocks/SM**; registers are **96** per-warp and **95** block-shared, with no spills.

Layout (`VALUES_PER_POINT = 30`):

| slots | contents |
|---|---|
| 0-2 | support starts x,y,z (ints after staging) |
| 3-4 | strength real, imag |
| 5-10 | x coefficients, 11 = zero (also y pre-pad) |
| 12-17 | y coefficients |
| 18-20 | zero (z pre-pads; 18 is also the y post-pad) |
| 21-26 | z coefficients |
| 27-29 | zero (z post-pads) |

## Results

Medians of 4 process runs, 5 timed runs each, best configuration per workload
(`uniform qb=1 radix`, `clustered qb=4 counting`, `sparse qb=4 radix`). Raw logs:
`ab_v11av_v11ay.txt`.

| workload | v11av pre | v11ay pre | v11av spread | v11ay spread | v11av total | v11ay total |
|---|---:|---:|---:|---:|---:|---:|
| uniform `1e6` `64^3` qb=1 radix | 3.088 | 3.086 | 12.528 | **10.912** | 16.058 | **14.446** |
| clustered `1e6` `64^3` qb=4 counting | 1.464 | 1.464 | 13.714 | **11.232** | 15.314 | **12.834** |
| sparse `1e5` `128^3` qb=4 radix | 0.284 | 0.284 | 1.536 | **1.393** | 2.089 | **1.956** |

Relative to the pinned method-3 reference (spread / total 15.95 / 16.78, 15.90 / 16.79,
2.82 / 2.95 ms):

| workload | spread speedup | total speedup |
|---|---:|---:|
| uniform | **1.46x** (31.6% faster) | **1.16x** (13.9% faster) |
| clustered | **1.42x** (29.4% faster) | **1.31x** (23.6% faster) |
| sparse | **2.02x** (50.6% faster) | **1.51x** (33.7% faster) |

Both mandatory `1e6` spread gates (>=20% faster than method 3) now pass. `relL2` is
5.64e-7 uniform, 2.26e-7 clustered, 7.34e-8 sparse — the same FP32 noise floor.

### Isolation of the two ideas

Two-process-run interleaved A/B, same binaries:

| variant | uniform spread | clustered spread | sparse spread |
|---|---:|---:|---:|
| v11av baseline | 12.513 | 13.720 | 1.542 |
| v11ax_z: contiguous z only | 11.928 (-4.7%) | 12.726 (-7.2%) | — |
| v11ax: contiguous z + strength scaling | 11.043 (-11.8%) | 11.608 (-15.4%) | 1.402 |
| v11ay: + contiguous y (final) | **10.912 (-12.8%)** | **11.232 (-18.1%)** | **1.393 (-9.7%)** |

So the select-chain removal is roughly a third of the win; removing half the inner
multiplies is the larger part. This matches the instruction counts: 671.9M -> 560.7M
uniform (-16.6%) and 868.6M -> 686.1M clustered (-21.0%).

### Profile

ncu captures: `uniform_qb1_ncu.ncu-rep`, `clustered_qb4_ncu.ncu-rep` (details and source
CSVs alongside). The captures were taken with unfixed clocks; use the benchmark medians
for wall time.

| metric | v11av U / C | v11ay U / C |
|---|---:|---:|
| ncu kernel duration | 12.60 / 13.53 ms | **11.12 / 11.05 ms** |
| warp instructions | 671.9M / 868.6M | **560.7M / 686.1M** |
| Compute (SM) throughput | 62.5 / 80.7 % | 60.6 / 79.3 % |
| Memory throughput | 43.9 / 55.2 % | **57.3 / 74.9 %** |
| L1/TEX throughput | 51.3 / 62.5 % | **63.5 / 77.9 %** |
| Active warps / scheduler | 4.15 / 4.87 | 4.12 / 4.87 |
| No eligible | 31.0 / 25.0 % | 34.4 / 27.4 % |
| Avg. active threads / warp | 16.06 / 10.44 | 17.26 / 12.62 |
| registers / shared / blocks | 96·94 / 12,760 B / 5 | 96·95 / 12,696·12,936 B / 5 |

The kernel is now less purely instruction-bound: clustered is at ~79% compute and ~78%
L1/TEX, and uniform memory throughput rose from 44% to 57% while compute fell. The
remaining scan cost is the irreducible two-FFMA-per-cell accumulation plus the
coefficient loads; the old z-select `FSEL` chain is gone. The regenerated,
PC-deduplicated trees attribute **25.5% / 10.5%** to staging (uniform raw + coefficient
frames / clustered block staging) and **55.6% / 70.9%** to the scan including its support
helpers. Within the scan, support-helper lines account for **15.8% / 11.9%** of the whole
kernel, strength×z source lines for **4.7% / 13.8%**, and cell-update source lines 428–429
for **17.6% / 26.3%**. These are source-attribution samples, not wall time, and the v11av
analysis document's caveats still apply.

Flamegraphs and call trees:
`uniform_qb1_flamegraph_chrome.json`, `uniform_qb1_calltree.txt`,
`clustered_qb4_flamegraph_chrome.json`, `clustered_qb4_calltree.txt`. Regenerated offline
from the existing CSVs using the matching source snapshot, not hard-coded line ranges:

| capture | unique samples | duplicate PC rows removed | supplied ncu duration |
|---|---:|---:|---:|
| uniform | 151,259 | 64 | 11,118.688 µs |
| clustered | 164,912 | 92 | 11,049.024 µs |

Both roots account for 100% of the deduplicated CSV samples. Widths/time coordinates are
explicitly labeled as synthetic sample attribution, not a GPU timeline. The clustered
CSV's distinct-PC instruction sum is 685,949,114 versus the aggregate report's
686,088,043; the converter preserves exported PC counters without forcing agreement.
No kernel or benchmark measurements changed.

## Rejected variants

| variant | change | result |
|---|---|---|
| v11ax_z | contiguous z only, no strength scaling | -4.7% / -7.2%; kept as the intermediate measurement |
| v11az | all three rows contiguous (STAGE=31, BATCH 12/51) | invalid: row stride 31 floats breaks the 8-byte `int2` metadata load (`misaligned address`) |
| v11ba | scale xy by strength instead of z | uniform 11.206, clustered 11.356, sparse 1.423; worse |
| v11bb | strength scaling scoped inside the z loop | 10.817 / 11.237 / 1.393; tied with v11ay, no register change; not kept |
| v11bc | `EXP_A_NSPLIT` sweep | uniform nsplit 2/4: 11.117 / 11.149 (nsplit 1 is 10.918); clustered 16/32/56/64: 11.888 / 11.326 / **11.231** / 11.363; 56 remains the knee |
| v11bd | three-point scan unroll | uniform 10.916, clustered 11.418, sparse 1.396; clustered worse |
| v11be | STAGE=32, all rows contiguous | 11.561 / 11.464; BATCH loss plus same-bank row aliasing; worse |
| v11bf | two `float2` staging stores | 10.937 / 11.235 / 1.393; neutral, ptxas store scheduling already adequate |

The non-obvious negative was v11be: a power-of-two row stride gives 16-byte alignment but
makes every point row start on the same shared-memory bank, so the apparent select
savings are lost to shared conflicts. STAGE=30 (16 distinct banks, 2-way conflict for 32
lanes) is the best even stride available; an odd stride would break the aligned metadata
load.

## Remaining headroom

- **Preprocessing**: the uniform total is now 14.45 ms, of which preprocess + metadata is
  3.54 ms (24%). `kernel_gather_sorted_packed` alone is ~1.35 ms (nsys, uniform): sorting
  gives a random permutation, so the 32-byte payload gather is scattered. A packed sorted
  payload (write one `float4` + one scalar instead of x/y/z/strength separately, and read
  it in staging) is the cheapest next experiment; it does not change the scan.
- **Scan**: the accumulation is now 32 FFMAs per point/region for 16 owned cells, of which
  only ~40 cells of 512 are useful on average. Reducing that needs a different ownership /
  geometry, not another packing tweak. The previously rejected 8-cell geometry, masks,
  compaction, SoA, and block-wide naive sharing all remain closed.
- **Clustered** is now at 11.23 ms, below the 12.72 ms stretch target; the block-shared
  path is still the right dispatch for it. Uniform/sparse remain on the per-warp path.

## Artifacts

Source: `exp_a_cartesian.cu` (also copied to `devel/morton_spread/exp_a_cartesian.cu`).
Binaries: `build/gpu/exp_a_cartesian[_lineinfo]` are v11ay; variant binaries are
`build/gpu/exp_a_cartesian_v11ax`, `_v11ax_z`, `_v11ay`, `_v11az`, `_v11ba`, `_v11bb`,
`_v11bc`, `_v11bd`, `_v11be`, `_v11bf`.

- `ab_v11av_v11ay.txt` — 4-process-run interleaved benchmark log.
- `bench.txt` — same run, extracted one-line summaries.
- `ptxas.txt` — ptxas resource log for v11ay.
- `uniform_ncu_run.txt`, `clustered_ncu_run.txt` — ncu command output.
- `*_ncu.ncu-rep`, `*_details.txt`, `*_source.csv` — ncu reports and source/SASS pages.
- `*_flamegraph_chrome.json`, `*_calltree.txt` — nested source-attribution graphs.


## Committed-source verification (2026-10-23 17:32 UTC)

Rebuilt `devel/morton_spread/exp_a_cartesian.cu` from the committed worktree
(`git hash-object` and `HEAD:` both `446b0345f28dd7d41b37f8392f725344de9df46d`) with the
`run_exp_a.sh` `nvcc` command into `build/gpu/exp_a_cartesian_verify`, then ran the same
4-process-run / 5-timed-run protocol. Raw log: `verify_committed_bench.txt`.

| workload | reported prep / spread / total | verify prep / spread / total | spread delta |
|---|---:|---:|---:|
| uniform | 3.086 / 10.912 / 14.446 | 3.089 / **10.917** / 14.476 | +0.05% |
| clustered | 1.464 / 11.232 / 12.834 | 1.461 / **11.234** / 12.827 | +0.02% |
| sparse | 0.284 / 1.393 / 1.956 | 0.284 / **1.396** / 1.972 | +0.2% |

The spread medians reproduce within 0.2%. Totals are within 0.8%; the sparse total
difference is metadata-timing variance (verify meta 0.293 ms vs the implied 0.279 ms,
individual runs 0.245-0.348 ms). `relL2` and resources reproduce as well: uniform
5.634e-7 / 96 regs / 12,696 B / 5 blocks; clustered 2.1-2.6e-7 / 95 regs / 12,936 B /
5 blocks; sparse 7.34e-8 / 96 regs / 12,696 B / 5 blocks.
