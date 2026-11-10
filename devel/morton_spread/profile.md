# NCU bottleneck report: `atomic_free_spread.cu::spread`

## Scope and data used

This report is based on the existing Nsight Compute captures only; no profile was re-run for
it.

- `build/gpu/traces/af_uniform_ncu.ncu-rep`
- `build/gpu/traces/af_clustered_ncu.ncu-rep`
- `build/gpu/traces/af_sparse_ncu.ncu-rep`
- `build/gpu/traces/af_dense_ncu.ncu-rep`
- Source CSVs: `/tmp/af_{uniform,clustered,sparse}_source.csv`

The kernel is the restored HEAD baseline `spread`, with 256 threads per block and one thread
per output cell.

The NCU source CSV repeats inlined SASS rows under multiple source lines. To avoid double
counting, every SASS address was counted once, keeping its first source correlation. The
resulting totals match the NCU Instruction Statistics totals exactly:

| case | total warp instructions | total PC samples |
|---|---:|---:|
| uniform `1e6/64^3` | 16,394,605,800 | 3,412,028 |
| clustered `1e6/64^3` | 18,332,269,284 | 3,670,014 |
| sparse `1e5/128^3` | 1,753,940,400 | 360,336 |

Durations in the NCU reports differ from benchmark medians because clocks are unlocked and
the kernel is replayed. The instruction, utilization, and stall data are the reliable part.

Source line numbers below come from the NCU source view. Lines such as `47`, `68`, `70`,
`72`, and `124` are inlined from `include/cufinufft/spreadinterp.hpp`; lines `90`, `133`,
`134`, `144`, and `150` are in `atomic_free_spread.cu`.

## 1. Global utilization and resources

| metric | uniform | clustered | sparse | dense cluster |
|---|---:|---:|---:|---:|
| NCU duration | 261.5 ms | 492.9 ms | 27.36 ms | 110.9 ms |
| Compute (SM) throughput | 82.9% | 56.3% | 84.8% | 20.5% |
| Memory throughput | 8.7% | 5.5% | 10.9% | 2.2% |
| DRAM throughput | 0.9% | 2.0% | 0.7% | 0.1% |
| IPC | 3.35 | 2.58 | 3.39 | n/a |
| Issue slots busy | 83.8% | 64.5% | 84.9% | n/a |
| FMA pipe | 79.5% | 64.1% | 76.4% | n/a |
| ALU pipe | 63.4% | 47.7% | 63.8% | n/a |
| XU pipe | 49.8% | 34.9% | 50.7% | n/a |
| Registers/thread | 54 | 54 | 54 | 54 |
| Static shared memory/block | 36 B | 36 B | 36 B | 36 B |
| Theoretical occupancy | 100% | 100% | 100% | 100% |
| Achieved occupancy | 95.5% | 63.3% | 96.3% | 27.8% |
| Active warps/scheduler | 7.64 | 5.57 | 7.69 | 2.23 |
| Eligible warps/scheduler | 3.56 | 2.39 | 3.59 | 0.75 |
| No eligible warps | 16.2% | 28.0% | 15.1% | 51.8% |
| Warp cycles/issued instruction | 9.11 | 7.74 | 9.06 | 4.63 |
| Active threads/warp | 23.9 | 21.3 | 24.4 | 23.1 |
| Branch efficiency | 88.6% | 85.0% | 92.0% | n/a |

Resources are not the limiting factor: 54 registers, 36 B shared memory, 100% theoretical
occupancy, zero local-memory traffic, zero shared-memory bank conflicts. Uniform and sparse
achieve 95-96% occupancy. Clustered and dense lose occupancy because only a small number of
tiles contain work, not because of register or shared-memory pressure.

## 2. Instruction budget by source region

This is the central justification for the ranking. The groups are non-overlapping and cover
essentially the whole kernel.

| region | uniform inst | % | clustered inst | % | sparse inst | % |
|---|---:|---:|---:|---:|---:|---:|
| support index: `coefficient_index()` lines 90-92 | 7.301B | **44.53%** | 7.296B | **39.80%** | 0.765B | **43.64%** |
| Horner/FMA: inlined `cudaFMA`/Horner lines 47, 124, 147-149 | 3.541B | 21.60% | 4.655B | 25.39% | 0.354B | 20.17% |
| weight/select/accum: lines 150-152 | 2.641B | 16.11% | 3.464B | 18.90% | 0.264B | 15.05% |
| `fold_rescale`: lines 70, 72, 135-137 | 1.152B | 7.03% | 1.152B | 6.28% | 0.115B | 6.58% |
| `ceilf`: lines 138-140 | 0.576B | 3.51% | 0.576B | 3.14% | 0.058B | 3.28% |
| bin loop: line 133 | 0.675B | 4.12% | 0.674B | 3.68% | 0.089B | 5.08% |
| point load: line 134 | 0.214B | 1.30% | 0.221B | 1.21% | 0.021B | 1.22% |
| support branch: line 144 | 0.289B | 1.76% | 0.288B | 1.57% | 0.038B | 2.15% |
| **sum** | **16.388B** | **99.96%** | **18.326B** | **99.97%** | **1.704B** | **97.17%** |

The sparse remainder is mostly `near_bins()` setup, loop control, and the final output store.

The same grouping in terms of PC-sample share:

| region | uniform samples | % | clustered samples | % | sparse samples | % |
|---|---:|---:|---:|---:|---:|---:|
| support index | 1.587M | **46.50%** | 1.424M | **38.79%** | 0.161M | **44.66%** |
| Horner/FMA | 0.655M | 19.20% | 0.993M | 27.05% | 0.048M | 13.33% |
| weight/select/accum | 0.233M | 6.84% | 0.264M | 7.19% | 0.022M | 6.10% |
| fold_rescale | 0.489M | 14.33% | 0.552M | 15.04% | 0.043M | 11.91% |
| ceilf | 0.195M | 5.70% | 0.168M | 4.59% | 0.018M | 5.07% |
| bin loop | 0.102M | 2.99% | 0.109M | 2.97% | 0.017M | 4.72% |
| point load | 0.032M | 0.93% | 0.037M | 1.01% | 0.003M | 0.85% |
| support branch | 0.105M | 3.09% | 0.104M | 2.82% | 0.014M | 3.78% |

## 3. Hottest individual source sites

| source site | uniform inst | uniform % | clustered % | sparse % |
|---|---:|---:|---:|---:|
| `int offset = (cell - support_start) % grid_size;` | 5.861B | **35.75%** | **31.94%** | **35.43%** |
| inlined `return __fmaf_rn(a, b, c);` | 3.277B | **19.99%** | **23.49%** | **18.67%** |
| `float weight = kx[ix] * ky[iy] * kz[iz];` | 2.479B | **15.12%** | **17.90%** | **14.13%** |
| `if (offset < 0) offset += grid_size;` | 0.864B | 5.27% | 4.71% | 4.93% |
| `for (int i = bin_offsets[bin]; ...)` | 0.675B | 4.12% | 3.68% | 5.08% |
| `result = __fsub_rd(result, floorf(result));` | 0.576B | 3.51% | 3.14% | 3.29% |
| `return offset < WIDTH ? offset : -1;` | 0.576B | 3.51% | 3.14% | 3.28% |
| `if (ix < 0 \|\| iy < 0 \|\| iz < 0) continue;` | 0.289B | 1.76% | 1.57% | 2.15% |
| `return __fmul_rd(result, static_cast<T>(N));` | 0.288B | 1.76% | 1.57% | 1.64% |
| `Point p = points[i];` | 0.214B | 1.30% | 1.21% | 1.22% |
| `int sx/sy/sz = int(ceilf(...))` (3 lines) | 0.576B | 3.51% | 3.14% | 3.28% |

The single modulo line is 31.9-35.8% of all kernel instructions in every case.

The weight/select line is also the source of the dynamic coefficient-index select chain.
`ISETP.EQ.AND` alone is:

- uniform: 1.185B instructions, 7.23% of the kernel
- clustered: 1.568B instructions, 8.55%
- sparse: 0.118B instructions, 6.75%

`SEL` adds another 0.289B / 0.289B / 0.037B. This is consistent with compiling `kx[ix]`,
`ky[iy]`, and `kz[iz]` as predicated selection trees rather than direct indexed loads.

## 4. SASS opcode mix

Family-level totals show that integer/address and predicate/control instructions dominate even
more than FP32 FMA:

| opcode family | uniform | clustered | sparse |
|---|---:|---:|---:|
| `FFMA` | 21.70% | 25.17% | 20.27% |
| `IMAD*` (all multiply-add variants) | 19.78% | 19.01% | 19.19% |
| `ISETP*` (all predicate comparisons) | 17.19% | 17.47% | 17.03% |
| `IADD3` | 5.86% | 5.24% | 5.92% |
| `LOP3` | 4.10% | 3.67% | 3.84% |
| `FADD` | 3.91% | 3.62% | 3.66% |
| `MOV` | 3.62% | 4.28% | 3.56% |
| `IABS` | 3.52% | 3.14% | 3.46% |
| `F2I` | 2.93% | 2.62% | 2.91% |
| `FMUL` | 2.02% | 1.89% | 1.89% |
| `LDG` | 1.31% | 1.21% | 1.59% |

Full `IMAD`-family and `ISETP`-family instructions together are about 37% of uniform
instructions and 36.5% of clustered instructions. That is the integer/modulo/select
machinery, not FP32 math.

The modulo sequence at line 90 is not one instruction. Its SASS includes, among others:

- `IMAD.HI.U32` 480M
- `IMAD` 480M
- `IABS` 576M
- `IADD3` 576M
- `ISETP.GT.U32.AND` 384M
- `MUFU.RCP` 192M
- `I2F.RP` 192M
- `F2I.FTZ.U32.TRUNC.NTZ` 192M
- many `IMAD.MOV`, `ISETP`, and `UMOV` variants

That is why a single source line accounts for more than a third of the instruction stream.

## 5. What the redesign would actually remove

The kernel repeats point-only work once per output cell. The amount of repetition can be
measured from the fold instruction count.

`FRND.FLOOR` executes 288,000,000 warp instructions in each case. There are three fold calls
per point-cell visit, so the number of point-cell visits is:

```text
288M warp instructions × active threads per warp / 3 dimensions
```

Using the measured average active threads/warp:

- uniform: about 2.29B point-cell visits for 1e6 points, roughly **2,300 visits per point**
- using 32 threads as an upper bound: about 3.07B visits, roughly **3,100 per point**

Each visit recomputes:

- 3 `fold_rescale` calls
- 3 `ceilf` conversions
- 3 `coefficient_index` calls, each with the runtime modulo
- 18 Horner coefficients (6 taps × 3 dimensions), but only 3 taps are used

This is the root cause of the instruction volume. The fold/Horner coefficients depend only on
the point, not on the output cell, so most of this work is redundant by construction.

It also explains why the active-tile experiment did not produce a large win: empty blocks are
cheap. Clustered has only about 96/1024 tiles with work, yet it executes 12% more instructions
than fully active uniform, because the active tiles scan dense clusters heavily. The cost is
in the active tiles, not the empty ones.

## 6. Scheduler and stall breakdown

Stall shares are percentages of all warp-state stall cycles for that kernel.

| stall reason | uniform | clustered | sparse | dense cluster |
|---|---:|---:|---:|---:|
| Not selected | 40.0% | 33.5% | 40.0% | 17.0% |
| Math pipe throttle | 30.1% | 26.2% | 28.1% | 12.1% |
| Wait | 14.2% | 15.7% | 15.7% | 30.5% |
| Long scoreboard | 6.3% | 14.9% | 1.5% | 26.7% |
| MIO throttle | 3.8% | 4.7% | 4.0% | 6.2% |
| Dispatch stall | 2.8% | 2.3% | 3.1% | 1.6% |
| Branch resolving | 1.2% | 1.3% | 1.2% | 2.7% |
| No instruction | 1.1% | 0.9% | 1.2% | 1.6% |
| Short scoreboard | 0.4% | 0.4% | 2.7% | 1.3% |
| Barrier | 0.0% | 0.0% | 2.4% | 0.3% |

Interpretation:

- **Uniform and sparse** are compute-pipe limited. `Not selected` is high because there are
  many eligible warps, which is healthy. `Math pipe throttle` at 28-30% means the FMA/ALU
  pipes are saturated. FMA utilization is 79.5% uniform / 76.4% sparse.
- **Clustered** has lower pipe utilization and much higher `Long scoreboard` at 14.9%,
  consistent with its L2 load hit rate dropping to 17.0%. It is partly latency-limited.
- **Dense cluster** is strongly latency/parallelism limited: `Wait` 30.5% + `Long scoreboard`
  26.7% = 57.2% of stalls, `No eligible` 51.8%, achieved occupancy 27.8%, compute throughput
  20.5%.

## 7. Memory behavior

Memory is not the primary bottleneck by bandwidth, but it is not free either.

| metric | uniform | clustered | sparse |
|---|---:|---:|---:|
| Memory throughput | 8.7% | 5.5% | 10.9% |
| DRAM throughput | 0.9% | 2.0% | 0.7% |
| L1/TEX throughput | 13.1% | 8.1% | 15.2% |
| L1 hit rate, global loads | 82.0% | 82.1% | 96.6% |
| L2 hit rate, loads | 78.8% | 17.0% | 90.2% |
| Avg. bytes/sector, global loads | 6.20 | 6.25 | 5.70 |
| Global load instructions | 214.7M | 221.8M | 27.7M |
| Global load share of instructions | 1.31% | 1.21% | 1.59% |
| Long-scoreboard share of stalls | 6.3% | 14.9% | 1.5% |

The low bytes/sector is mostly because all threads in a warp load the same point or offset, so
the load is a broadcast/scalar load rather than a coalesced 128-bit load. It is not primarily a
coalescing bug, and memory bandwidth is not saturated. The meaningful memory effect is latency:
clustered L2 hit is only 17%, and long-scoreboard stalls rise to 14.9%; dense reaches 26.7%.

## 8. Ranked bottlenecks with justification

### 1. Support-index construction in `coefficient_index()` — dominant

- 44.53% of uniform instructions, 39.80% clustered, 43.64% sparse.
- 46.50% of uniform samples, 38.79% clustered, 44.66% sparse.
- The modulo line alone is 35.75% / 31.94% / 35.43% of all instructions.
- The modulo is a long reciprocal/multiply-high/predicate sequence, not a single remainder
  operation.
- It runs three times per point per cell.

This is the largest single instruction and stall site in every case.

### 2. Redundant Horner/FMA plus dynamic coefficient selection and weight accumulation

- Horner/FMA: 21.60% uniform, 25.39% clustered, 20.17% sparse.
- Weight/select/accum: 16.11% uniform, 18.90% clustered, 15.05% sparse.
- Combined: 37.71% / 44.29% / 35.22% of instructions.
- Combined sample share: 26.04% / 34.24% / 19.43%.
- `FFMA` is 21.70% / 25.17% / 20.27% of all instructions.
- `ISETP.EQ.AND` plus `SEL` from dynamic coefficient indexing is 1.47B uniform / 1.86B
  clustered instructions.
- All six taps per dimension are computed, but only one is used.

This is the second-largest instruction cost and the largest source of FP32 work.

### 3. `fold_rescale()` and `ceilf()` work, repeated per cell

- Fold: 7.03% uniform, 6.28% clustered, 6.58% sparse.
- `ceilf`: 3.51% / 3.14% / 3.28%.
- Combined instructions: 10.54% / 9.42% / 9.86%.
- Combined sample share: 20.03% / 19.63% / 16.98%.

Although this is a smaller instruction share than Horner, its sample share is
disproportionately high because the fold chain (`FRND.FLOOR`, `FADD.RM`, `FMUL.RM`) and the
point-load dependency create fixed-latency waits. The clustered
`pz = fold_rescale(...)` line alone has 6.17% of all samples.

### 4. Bin loop, point load, and support branch

- Bin loop: 4.12% / 3.68% / 5.08%.
- Point load: 1.30% / 1.21% / 1.22%.
- Support branch: 1.76% / 1.57% / 2.15%.
- Combined: 7.18% / 6.46% / 8.45% of instructions.
- Combined sample share: 7.01% / 6.80% / 9.35%.

This is not the top instruction cost, but it contributes to divergence: branch efficiency is
88.6% / 85.0% / 92.0%, and average active threads/warp fall to 23.9 / 21.3 / 24.4.

### 5. Memory latency, especially for clustered/dense

- DRAM and memory throughput are low: memory 8.7% / 5.5% / 10.9%, DRAM 0.9% / 2.0% / 0.7%.
- L2 load hit is 78.8% / 17.0% / 90.2%.
- Long-scoreboard stall share is 6.3% / 14.9% / 1.5% / 26.7% in dense.
- Global loads are only 1.2-1.6% of instructions, so load-instruction pressure is small.

Memory is therefore a secondary bottleneck for uniform/sparse, but a real latency
contributor for clustered/dense.

### 6. Clustered/dense load imbalance — separate from instruction count

- Clustered has about 96/1024 active tiles; dense has about 12/1024.
- Clustered executes only 11.8% more instructions than uniform but takes 88.5% longer under
  NCU.
- Clustered achieved occupancy is 63.3%, no-eligible 28.0%, compute 56.3%.
- Dense achieved occupancy is 27.8%, no-eligible 51.8%, compute 20.5%.

This is not an instruction-count problem in the same sense as items 1-3. It is a
work-distribution problem: the few heavy blocks cannot fill the GPU. Skipping empty tiles does
not fix it; splitting hot tiles or using a persistent work queue does.

### 7. Minor overheads

- FP32 fusion: 3.56B fused vs 0.97B non-fused; NCU estimates up to ~8.5% from more fusion, but
  this is small relative to removing redundant work.
- Shared memory and `__syncthreads` are negligible: 36 B/block, zero bank conflicts, 72,704
  shared wavefronts.
- No local-memory spills.
- `assign_bins` is negligible in benchmark timing: 0.30 ms uniform, 0.33 ms clustered, 0.05 ms
  sparse.

## Bottom line

The profile validates this ranking by both instruction count and stall attribution:

1. **Support-index/modulo**: ~40-45% of instructions and stall samples.
2. **Horner/FMA + weight/select**: ~35-44% of instructions and ~19-34% of samples.
3. **Fold/ceil**: ~10% of instructions but ~17-20% of samples because of latency.
4. **Bin loop/point load/support branch**: ~6-8% of instructions, plus divergence.
5. **Memory**: not bandwidth-bound; latency matters mainly for clustered/dense.
6. **Occupancy/load imbalance**: not resource-limited; clustered/dense are underfilled by
   active work.

The dominant design-level issue is that each point's fold, Horner coefficients, support index,
and weight selection are recomputed for roughly 2,300-3,100 output-cell visits per point.
Removing the modulo and sharing point-level work across cells would attack the largest measured
costs directly. Optimizing memory coalescing or register occupancy first would not address the
instruction and stall profile shown here.
