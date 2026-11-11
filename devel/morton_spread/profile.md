# Atomic-free spread vs cuFINUFFT method 3 profiling digest

Generated: 2026-11-11T16:01:06

Repository: `/home/paseo/.paseo/worktrees/3gvdkmv7/loud-dog`  
Commit: `7499155172da89f564b676a0d390e59b8f92a349`  
Git status: M devel/morton_spread/README.md;  M devel/morton_spread/atomic_free_spread.cu;  M devel/morton_spread/profile.md;  M devel/morton_spread/profile_atomic_free_spread.py  
Capture: `build/gpu/traces/af_profile_method3_final`  
GPU: Quadro T1000, 7.5, 580.178.04, 4096 MiB  
CUDA: nvcc: NVIDIA (R) Cuda compiler driver  
Nsight Compute: NVIDIA (R) Nsight Compute Command Line Profiler  
Build: `/home/paseo/cuda-12.8/bin/nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Iinclude -I/tmp/cccl-3.4.2/thrust -I/tmp/cccl-3.4.2/libcudacxx/include -I/tmp/cccl-3.4.2/cub -isystem=/home/paseo/cuda-12.8/include -isystem=/home/paseo/.paseo/worktrees/3gvdkmv7/loud-dog/build/gpu/_deps/poet-src/include devel/morton_spread/atomic_free_spread.cu devel/morton_spread/f0_point_owned_spread.cu devel/morton_spread/atomic_free_spread_bench.cu -o /home/paseo/.paseo/worktrees/3gvdkmv7/loud-dog/build/gpu/atomic_free_spread_bench_lineinfo /home/paseo/.paseo/worktrees/3gvdkmv7/loud-dog/build/gpu/libcufinufft.a /home/paseo/.paseo/worktrees/3gvdkmv7/loud-dog/build/gpu/src/common/libfinufft_common.a -lcudart -lcufft -lpthread -ldl -lm`

## Artifact map

`<case>` is `<workload>__<variant>`, where variant is `atomic_free` or `method3`.

- `reports/<case>.ncu-rep` — original Nsight Compute report
- `ncu/<case>.raw.csv` — every raw metric/unit/value from the report
- `ncu/<case>.details.csv` / `.txt` — section-organized metrics and rules
- `ncu/<case>.source.csv` / `.txt` — CUDA/SASS source correlation
- `analysis/<case>/source_hotspots.csv` — every source line by instructions/samples
- `analysis/<case>/source_stalls.csv` — source line x stall reason
- `analysis/<case>/source_groups.csv` — heuristic semantic grouping
- `analysis/<case>/sass_opcodes.csv` / `sass_opcode_families.csv` — SASS mix
- `analysis/<case>/stalls.csv` — warp-state stall breakdown
- `analysis/<case>/metrics_details.csv` — all detail metrics in long form


## Cases

| workload | implementation | bench args (M N1 N2 N3 cluster bin_size pattern seed) | atomic-free bench median ms | method 3 bench execute ms | ncu kernel ms | ncu / matching bench | relL2 | checks |
|---|---|---|---|---|---|---|---|---|
| uniform | atomic-free spread | 1000000 64 64 64 0 4 0 42 | 20.39 | 19.99 | 20.38 | 1.00x | 5.282e-07 | PASS |
| uniform | cuFINUFFT method 3 (output-driven) | 1000000 64 64 64 0 4 0 42 | 20.39 | 19.99 | 17.90 | 0.90x | 5.282e-07 | PASS |
| clustered | atomic-free spread | 1000000 64 64 64 1 4 0 42 | 26.00 | 20.11 | 26.03 | 1.00x | 6.525e-06 | PASS |
| clustered | cuFINUFFT method 3 (output-driven) | 1000000 64 64 64 1 4 0 42 | 26.00 | 20.11 | 17.88 | 0.89x | 6.525e-06 | PASS |
| sparse | atomic-free spread | 100000 128 128 128 0 4 0 42 | 4.29 | 5.10 | 4.29 | 1.00x | 7.958e-08 | PASS |
| sparse | cuFINUFFT method 3 (output-driven) | 100000 128 128 128 0 4 0 42 | 4.29 | 5.10 | 2.97 | 0.58x | 7.958e-08 | PASS |
| dense | atomic-free spread | 100000 64 64 64 0 4 3 42 | 7.59 | 4.07 | 7.60 | 1.00x | 8.649e-06 | PASS |
| dense | cuFINUFFT method 3 (output-driven) | 100000 64 64 64 0 4 3 42 | 7.59 | 4.07 | 1.78 | 0.44x | 8.649e-06 | PASS |

`ncu kernel ms` is one replayed launch with unlocked clocks, not a benchmark median. For `atomic_free` it is the `spread` kernel; for `method3` it is the `spread_output_driven` kernel. The method-3 benchmark execute also includes the output-grid clear and launch overhead, so its NCU kernel time can be lower than the bench execute time.

The gap between method-3 bench `execute` and its NCU kernel time is about 2.1-2.3 ms on every workload here, consistent with a fixed output-grid clear plus launch/synchronization overhead.

## Quick read

- **`uniform`** — NCU kernel time: method 3 is 0.88x atomic-free (12% lower). Atomic-free: 92.02% occupancy, 97.57% L1/TEX, top stalls MIO Throttle 43.9%, Barrier 30.0%. Method 3: 24.99% occupancy, 65.15 KiB shared/block, 7.93% DRAM, 4.20% L2, top stalls Wait 26.8%, Barrier 23.0%.
- **`clustered`** — NCU kernel time: method 3 is 0.69x atomic-free (31% lower). Atomic-free: 43.21% occupancy, 70.65% L1/TEX, top stalls MIO Throttle 40.5%, Barrier 19.4%. Method 3: 25.05% occupancy, 65.15 KiB shared/block, 6.91% DRAM, 3.64% L2, top stalls Wait 26.8%, Barrier 23.0%.
- **`sparse`** — NCU kernel time: method 3 is 0.69x atomic-free (31% lower). Atomic-free: 98.91% occupancy, 90.63% L1/TEX, top stalls Barrier 22.4%, Wait 16.1%. Method 3: 24.92% occupancy, 65.15 KiB shared/block, 20.85% DRAM, 33.08% L2, top stalls Wait 29.5%, Barrier 22.6%.
- **`dense`** — NCU kernel time: method 3 is 0.23x atomic-free (77% lower). Atomic-free: 18.92% occupancy, 53.81% L1/TEX, top stalls Wait 22.9%, MIO Throttle 18.5%. Method 3: 25.00% occupancy, 65.15 KiB shared/block, 1.36% DRAM, 2.45% L2, top stalls Wait 26.8%, Barrier 22.7%.

## Cross-case key metrics

| metric | uniform__atomic_free | uniform__method3 | clustered__atomic_free | clustered__method3 | sparse__atomic_free | sparse__method3 | dense__atomic_free | dense__method3 |
|---|---|---|---|---|---|---|---|---|
| ncu duration (ms) | 20.38 | 17.90 | 26.03 | 17.88 | 4.29 | 2.97 | 7.60 | 1.78 |
| compute (SM) % | 73.57 | 37.93 | 49.30 | 37.99 | 75.73 | 33.37 | 21.50 | 38.04 |
| memory % | 73.57 | 40.76 | 49.30 | 40.84 | 75.73 | 33.08 | 21.50 | 40.65 |
| memory throughput (GByte/s) | 5.83 | 8.87 | 5.41 | 7.72 | 4.76 | 23.28 | 0.81 | 1.52 |
| DRAM % | 5.22 | 7.93 | 4.74 | 6.91 | 4.26 | 20.85 | 0.56 | 1.36 |
| L1/TEX % | 97.57 | 81.51 | 70.65 | 81.68 | 90.63 | 57.27 | 53.81 | 81.29 |
| L2 % | 2.97 | 4.20 | 2.25 | 3.64 | 3.98 | 33.08 | 0.93 | 2.45 |
| IPC | 1.62 | 1.13 | 1.52 | 1.13 | 2.66 | 1.28 | 1.16 | 1.13 |
| issue slots busy % | 40.47 | 28.21 | 38.00 | 28.19 | 66.57 | 32.08 | 29.11 | 28.26 |
| FMA pipe % | 49.07 | 13.66 | 46.50 | 13.68 | 52.29 | 13.22 | 34.23 | 13.71 |
| ALU pipe % | 8.91 | 21.26 | 7.99 | 21.17 | 44.68 | 33.77 | 7.41 | 21.21 |
| XU pipe % | 1.26 | 0.76 | 1.13 | 0.71 | 5.70 | 7.36 | 0.85 | 0.70 |
| achieved occupancy % | 92.02 | 24.99 | 43.21 | 25.05 | 98.91 | 24.92 | 18.92 | 25.00 |
| active warps/scheduler | 7.36 | 2.00 | 3.41 | 2.00 | 7.91 | 2.00 | 1.52 | 2.00 |
| eligible warps/scheduler | 0.94 | 0.32 | 0.60 | 0.32 | 1.62 | 0.40 | 0.34 | 0.32 |
| no eligible % | 59.53 | 71.79 | 62.33 | 71.85 | 33.43 | 67.88 | 68.04 | 71.74 |
| warp cycles / issued inst | 18.19 | 7.09 | 9.05 | 7.09 | 11.89 | 6.22 | 4.74 | 7.07 |
| active threads / warp | 31.62 | 31.11 | 31.99 | 31.11 | 18.48 | 31.40 | 31.93 | 31.11 |
| predicated threads / warp | 30.94 | 29.78 | 31.34 | 29.79 | 17.36 | 28.85 | 31.27 | 29.79 |
| branch inst ratio % | 0.03 | 0.09 | 0.02 | 0.09 | 0.07 | 0.06 | 0.03 | 0.09 |
| L1 hit global loads % | 83.54 | 1.30 | 83.46 | 1.01 | 89.29 | 13.21 | 84.27 | 62.34 |
| L2 hit loads % | 39.51 | 0.71 | 38.35 | 0.17 | 82.00 | 9.60 | 80.62 | 9.78 |
| avg bytes/sector load | 8.76 | 5.82 | 8.85 | 5.78 | 6.65 | 5.52 | 8.56 | 11.84 |
| warp instructions | 616,307,316 | 380,208,924 | 605,368,716 | 378,473,229 | 215,326,744 | 71,222,313 | 63,531,168 | 37,828,041 |
| registers/thread | 63 | 51 | 63 | 51 | 63 | 51 | 63 | 51 |
| static shared bytes | 5 | 0 | 5 | 0 | 5 | 0 | 5 | 0 |
| dynamic shared KiB | 0 | 65.15 | 0 | 65.15 | 0 | 65.15 | 0 | 65.15 |
| grid blocks | 512 | 1,099 | 512 | 992 | 4,096 | 2,197 | 512 | 98 |
| block threads | 128 | 256 | 128 | 256 | 128 | 256 | 128 | 256 |
| source instruction total | 616307316 | 380208924 | 605368716 | 378473229 | 215326744 | 71222313 | 63531168 | 37828041 |
| source sample total | 265706 | 235227 | 267999 | 234595 | 56477 | 38776 | 27693 | 23340 |

## Benchmark phase breakdown

| workload | host generate | upload points | assign bins | copy bins | host sort | upload sorted | prep total | setup total | atomic-free end-to-end | method 3 setpts | method 3 execute | method 3 total | atomic-free spread median | method-3 execute / spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 45.09 | 2.27 | 0.28 | 0.51 | 17.44 | 2.20 | 22.71 | 67.80 | 88.19 | 7.34 | 19.99 | 27.34 | 20.39 | 0.98 |
| clustered | 80.98 | 2.35 | 0.29 | 0.51 | 9.16 | 2.25 | 14.57 | 95.55 | 121.55 | 0.93 | 20.11 | 21.04 | 26.00 | 0.77 |
| sparse | 4.54 | 0.25 | 0.03 | 0.08 | 1.38 | 0.26 | 2.00 | 6.54 | 10.83 | 7.57 | 5.10 | 12.67 | 4.29 | 1.19 |
| dense | 7.69 | 0.25 | 0.03 | 0.08 | 1.06 | 0.26 | 1.67 | 9.36 | 16.96 | 4.10 | 4.07 | 8.17 | 7.59 | 0.54 |

All columns are milliseconds except the last, which is the ratio method-3 execute / atomic-free spread median (above 1 means method 3 is slower). `atomic-free end-to-end` is setup total + spread median; `method 3 total` is setpts + execute. The atomic-free preprocessing (assign bins, copy bins, host counting sort, upload sorted) is not part of the `spread` kernel time. Method-3 `setpts` is its own preprocessing phase.

## Implementation comparison

`ratio` is method 3 / atomic-free (above 1 means method 3 is larger or slower); `delta` is method 3 minus atomic-free in percentage points. Missing values mean the metric was not exported for that kernel.

### `uniform`

| metric | atomic-free | method 3 | method 3 vs atomic-free |
|---|---|---|---|
| ncu kernel duration (ms) | 20.38 | 17.90 | 0.88x |
| compute (SM) % | 73.57 | 37.93 | -35.64 pp |
| memory % | 73.57 | 40.76 | -32.81 pp |
| memory throughput (GByte/s) | 5.83 | 8.87 | 1.52x |
| DRAM % | 5.22 | 7.93 | +2.71 pp |
| L1/TEX % | 97.57 | 81.51 | -16.06 pp |
| L2 % | 2.97 | 4.20 | +1.23 pp |
| IPC | 1.62 | 1.13 | 0.70x |
| issue slots busy % | 40.47 | 28.21 | -12.26 pp |
| FMA pipe % | 49.07 | 13.66 | -35.41 pp |
| ALU pipe % | 8.91 | 21.26 | +12.35 pp |
| XU pipe % | 1.26 | 0.76 | -0.50 pp |
| achieved occupancy % | 92.02 | 24.99 | -67.03 pp |
| active warps/scheduler | 7.36 | 2.00 | 0.27x |
| eligible warps/scheduler | 0.94 | 0.32 | 0.34x |
| no eligible % | 59.53 | 71.79 | +12.26 pp |
| warp cycles / issued inst | 18.19 | 7.09 | 0.39x |
| active threads / warp | 31.62 | 31.11 | 0.98x |
| predicated threads / warp | 30.94 | 29.78 | 0.96x |
| L1 hit global loads % | 83.54 | 1.30 | -82.24 pp |
| L2 hit loads % | 39.51 | 0.71 | -38.80 pp |
| warp instructions | 616,307,316 | 380,208,924 | 0.62x |
| registers/thread | 63 | 51 | 0.81x |
| dynamic shared KiB | 0 | 65.15 | - |
| grid blocks | 512 | 1,099 | 2.15x |
| block threads | 128 | 256 | 2.00x |
| source sample total | 265706 | 235227 | 0.89x |

### `clustered`

| metric | atomic-free | method 3 | method 3 vs atomic-free |
|---|---|---|---|
| ncu kernel duration (ms) | 26.03 | 17.88 | 0.69x |
| compute (SM) % | 49.30 | 37.99 | -11.31 pp |
| memory % | 49.30 | 40.84 | -8.46 pp |
| memory throughput (GByte/s) | 5.41 | 7.72 | 1.43x |
| DRAM % | 4.74 | 6.91 | +2.17 pp |
| L1/TEX % | 70.65 | 81.68 | +11.03 pp |
| L2 % | 2.25 | 3.64 | +1.39 pp |
| IPC | 1.52 | 1.13 | 0.74x |
| issue slots busy % | 38.00 | 28.19 | -9.81 pp |
| FMA pipe % | 46.50 | 13.68 | -32.82 pp |
| ALU pipe % | 7.99 | 21.17 | +13.18 pp |
| XU pipe % | 1.13 | 0.71 | -0.42 pp |
| achieved occupancy % | 43.21 | 25.05 | -18.16 pp |
| active warps/scheduler | 3.41 | 2.00 | 0.59x |
| eligible warps/scheduler | 0.60 | 0.32 | 0.53x |
| no eligible % | 62.33 | 71.85 | +9.52 pp |
| warp cycles / issued inst | 9.05 | 7.09 | 0.78x |
| active threads / warp | 31.99 | 31.11 | 0.97x |
| predicated threads / warp | 31.34 | 29.79 | 0.95x |
| L1 hit global loads % | 83.46 | 1.01 | -82.45 pp |
| L2 hit loads % | 38.35 | 0.17 | -38.18 pp |
| warp instructions | 605,368,716 | 378,473,229 | 0.63x |
| registers/thread | 63 | 51 | 0.81x |
| dynamic shared KiB | 0 | 65.15 | - |
| grid blocks | 512 | 992 | 1.94x |
| block threads | 128 | 256 | 2.00x |
| source sample total | 267999 | 234595 | 0.88x |

### `sparse`

| metric | atomic-free | method 3 | method 3 vs atomic-free |
|---|---|---|---|
| ncu kernel duration (ms) | 4.29 | 2.97 | 0.69x |
| compute (SM) % | 75.73 | 33.37 | -42.36 pp |
| memory % | 75.73 | 33.08 | -42.65 pp |
| memory throughput (GByte/s) | 4.76 | 23.28 | 4.89x |
| DRAM % | 4.26 | 20.85 | +16.59 pp |
| L1/TEX % | 90.63 | 57.27 | -33.36 pp |
| L2 % | 3.98 | 33.08 | +29.10 pp |
| IPC | 2.66 | 1.28 | 0.48x |
| issue slots busy % | 66.57 | 32.08 | -34.49 pp |
| FMA pipe % | 52.29 | 13.22 | -39.07 pp |
| ALU pipe % | 44.68 | 33.77 | -10.91 pp |
| XU pipe % | 5.70 | 7.36 | +1.66 pp |
| achieved occupancy % | 98.91 | 24.92 | -73.99 pp |
| active warps/scheduler | 7.91 | 2.00 | 0.25x |
| eligible warps/scheduler | 1.62 | 0.40 | 0.25x |
| no eligible % | 33.43 | 67.88 | +34.45 pp |
| warp cycles / issued inst | 11.89 | 6.22 | 0.52x |
| active threads / warp | 18.48 | 31.40 | 1.70x |
| predicated threads / warp | 17.36 | 28.85 | 1.66x |
| L1 hit global loads % | 89.29 | 13.21 | -76.08 pp |
| L2 hit loads % | 82.00 | 9.60 | -72.40 pp |
| warp instructions | 215,326,744 | 71,222,313 | 0.33x |
| registers/thread | 63 | 51 | 0.81x |
| dynamic shared KiB | 0 | 65.15 | - |
| grid blocks | 4,096 | 2,197 | 0.54x |
| block threads | 128 | 256 | 2.00x |
| source sample total | 56477 | 38776 | 0.69x |

### `dense`

| metric | atomic-free | method 3 | method 3 vs atomic-free |
|---|---|---|---|
| ncu kernel duration (ms) | 7.60 | 1.78 | 0.23x |
| compute (SM) % | 21.50 | 38.04 | +16.54 pp |
| memory % | 21.50 | 40.65 | +19.15 pp |
| memory throughput (GByte/s) | 0.81 | 1.52 | 1.88x |
| DRAM % | 0.56 | 1.36 | +0.80 pp |
| L1/TEX % | 53.81 | 81.29 | +27.48 pp |
| L2 % | 0.93 | 2.45 | +1.52 pp |
| IPC | 1.16 | 1.13 | 0.97x |
| issue slots busy % | 29.11 | 28.26 | -0.85 pp |
| FMA pipe % | 34.23 | 13.71 | -20.52 pp |
| ALU pipe % | 7.41 | 21.21 | +13.80 pp |
| XU pipe % | 0.85 | 0.70 | -0.15 pp |
| achieved occupancy % | 18.92 | 25.00 | +6.08 pp |
| active warps/scheduler | 1.52 | 2.00 | 1.32x |
| eligible warps/scheduler | 0.34 | 0.32 | 0.94x |
| no eligible % | 68.04 | 71.74 | +3.70 pp |
| warp cycles / issued inst | 4.74 | 7.07 | 1.49x |
| active threads / warp | 31.93 | 31.11 | 0.97x |
| predicated threads / warp | 31.27 | 29.79 | 0.95x |
| L1 hit global loads % | 84.27 | 62.34 | -21.93 pp |
| L2 hit loads % | 80.62 | 9.78 | -70.84 pp |
| warp instructions | 63,531,168 | 37,828,041 | 0.60x |
| registers/thread | 63 | 51 | 0.81x |
| dynamic shared KiB | 0 | 65.15 | - |
| grid blocks | 512 | 98 | 0.19x |
| block threads | 128 | 256 | 2.00x |
| source sample total | 27693 | 23340 | 0.84x |

## Case: `uniform__atomic_free`

Workload: dense uniform input; all output tiles have work.  
Implementation: **atomic-free spread** — Output-cell-owned atomic-free prototype: every fine-grid cell is owned by one thread and stored once, with no spreading atomics.

Benchmark (same process for both implementations): atomic-free spread median **20.39 ms** (min 20.39, max 20.40); method 3 execute **19.99 ms** (setpts 7.34 ms); method-3 execute / spread **0.98x**; relL2 **5.282e-07**; checks **PASS**.

NCU: **20.38 ms**, compute **73.57%**, memory **73.57%**, DRAM **5.22%**, IPC **1.62**, occupancy **92.02%**, no-eligible **59.53%**.

### Resource and memory detail

| metric | value |
|---|---|
| elapsed cycles | 27,516,923 |
| waves per SM | 4.57 |
| memory throughput (GByte/s) | 5.83 |
| L1/TEX hit rate % | 83.41 |
| L2 hit rate % | 40.48 |
| L1 hit global loads % | 83.54 |
| L1 hit global stores % | 49.34 |
| avg bytes/sector load | 8.76 |
| avg bytes/sector store | 16.00 |
| shared-memory bank conflicts | 5,012,045 |
| shared load instructions | 128,460,416 |
| shared store instructions | 9,758,592 |
| global load instructions | 1,856,608 |
| global store instructions | 8,192 |
| global reduction (atomic) instructions | 0 |
| DRAM bytes read (MB) | 116.50 |
| DRAM bytes write (MB) | 2.34 |
| branch efficiency % | 97.49 |
| L1 conflicts shared N-way | 223.00 |
| L1 shared wavefronts excessive | 4,886,028 |
| L2 theoretical sectors global excessive (MB) | 26.39 |
| avg threads executed | 28464.00 |
| avg predicated-on threads executed | 27274.00 |
| block limit: shared memory | 13 |
| block limit: registers | 8 |
| theoretical active warps/SM | 32.00 |
| achieved active warps/SM | 29.45 |

### Stall breakdown

| warp state | cycles / issued inst | share of warp state | share of stalls |
|---|---|---|---|
| Stall MIO Throttle | 7.990 | 43.92% | 46.48% |
| Stall Barrier | 5.450 | 29.96% | 31.70% |
| Stall Not Selected | 1.340 | 7.37% | 7.79% |
| Selected | 1.000 | 5.50% | -% |
| Stall Wait | 0.980 | 5.39% | 5.70% |
| Stall Long Scoreboard | 0.700 | 3.85% | 4.07% |
| Stall Math Pipe Throttle | 0.340 | 1.87% | 1.98% |
| Stall Short Scoreboard | 0.210 | 1.15% | 1.22% |
| Stall No Instruction | 0.110 | 0.60% | 0.64% |
| Stall Branch Resolving | 0.050 | 0.28% | 0.29% |
| Stall Dispatch Stall | 0.020 | 0.11% | 0.12% |
| Stall Drain | 0 | 0% | 0% |

### Source groups (heuristic)

| group | instructions | inst % | samples | sample % | source lines | SASS PCs |
|---|---|---|---|---|---|---|
| weight / select / accumulate | 403168896 | 65.42% | 95893 | 36.09% | 4 | 400 |
| Horner / FMA | 98986688 | 16.06% | 42476 | 15.99% | 3 | 112 |
| other / kernel setup | 71286464 | 11.57% | 31124 | 11.71% | 40 | 377 |
| weight staging | 20540340 | 3.33% | 16708 | 6.29% | 4 | 75 |
| bin iteration / offsets | 11153952 | 1.81% | 59899 | 22.54% | 18 | 190 |
| point load | 4251904 | 0.69% | 14936 | 5.62% | 1 | 6 |
| fold_rescale | 3188928 | 0.52% | 677 | 0.26% | 3 | 4 |
| global write / atomics | 2128000 | 0.34% | 3681 | 1.39% | 3 | 3 |
| ceilf | 1594464 | 0.26% | 312 | 0.12% | 1 | 2 |
| support index / modulo | 7680 | 0.00% | 0 | 0% | 2 | 15 |

### Top source lines by instructions

| site | source | instructions | inst % | samples | sample % | group | top stall |
|---|---|---|---|---|---|---|---|
| `atomic_free_spread.cu:97` | weight[gx][gy] = kx[j][x0 + gx] * ky[j][y0 + gy]; | 146101344 | 23.71% | 76788 | 28.90% | weight / select / accumulate | mio |
| `atomic_free_spread.cu:110` | acc[gx][gy][gz].y += z[gz].y * weight[gx][gy]; | 129051168 | 20.94% | 9624 | 3.62% | weight / select / accumulate | selected |
| `atomic_free_spread.cu:109` | acc[gx][gy][gz].x += z[gz].x * weight[gx][gy]; | 128000000 | 20.77% | 9476 | 3.57% | weight / select / accumulate | selected |
| `atomic_free_spread.cu:101` | for (int gz = 0; gz < GROUP_Z; ++gz) z[gz] = kz[j][… | 54341696 | 8.82% | 38256 | 14.40% | Horner / FMA | mio |
| `spreadinterp.hpp:47` | return __fmaf_rn(a, b, c); | 39861600 | 6.47% | 3334 | 1.25% | Horner / FMA | wait |
| `atomic_free_spread.cu:91` | for (int j = 0; j < staged; ++j) { | 15974944 | 2.59% | 22607 | 8.51% | other / kernel setup | barrier |
| `atomic_free_spread.cu:94` | for (int gx = 0; gx < GROUP_X; ++gx) | 10681472 | 1.73% | 1975 | 0.74% | other / kernel setup | wait |
| `atomic_free_spread.cu:197` | const int staged = min(STAGE_POINTS, bin_end - base… | 10629760 | 1.72% | 1145 | 0.43% | other / kernel setup | wait |
| `atomic_free_spread.cu:55` | cell -= (cell >= fine_size) ? fine_size : 0; | 9566784 | 1.55% | 538 | 0.20% | other / kernel setup | selected |
| `atomic_free_spread.cu:80` | dest[local] = make_float2(ker[k] * strength.x, ker[… | 7720736 | 1.25% | 5076 | 1.91% | weight staging | mio |
| `atomic_free_spread.cu:58` | dest[local] = ker[k]; | 6441748 | 1.05% | 4615 | 1.74% | weight staging | mio |
| `atomic_free_spread.cu:77` | cell -= (cell >= fine_size) ? fine_size : 0; | 4783392 | 0.78% | 289 | 0.11% | other / kernel setup | selected |
| `atomic_free_spread.cu:206` | eval_kernel_vec_horner<float, WIDTH>(ker, float(sta… | 4783392 | 0.78% | 886 | 0.33% | Horner / FMA | short_sb |
| `atomic_free_spread.cu:196` | for (int base = bin_begin; base < bin_end; base += … | 4776192 | 0.78% | 57580 | 21.67% | bin iteration / offsets | barrier |
| `atomic_free_spread.cu:47` | dest[local] = 0.0f; | 4251904 | 0.69% | 4804 | 1.81% | weight staging | mio |

### Top source lines by PC samples

| site | source | samples | sample % | instructions | inst % | top stall |
|---|---|---|---|---|---|---|
| `atomic_free_spread.cu:97` | weight[gx][gy] = kx[j][x0 + gx] * ky[j][y0 + gy]; | 76788 | 28.90% | 146101344 | 23.71% | mio |
| `atomic_free_spread.cu:196` | for (int base = bin_begin; base < bin_end; base += … | 57580 | 21.67% | 4776192 | 0.78% | barrier |
| `atomic_free_spread.cu:101` | for (int gz = 0; gz < GROUP_Z; ++gz) z[gz] = kz[j][… | 38256 | 14.40% | 54341696 | 8.82% | mio |
| `atomic_free_spread.cu:91` | for (int j = 0; j < staged; ++j) { | 22607 | 8.51% | 15974944 | 2.59% | barrier |
| `atomic_free_spread.cu:199` | const Point p = points[base + stage_point]; | 14936 | 5.62% | 4251904 | 0.69% | long_sb |
| `atomic_free_spread.cu:110` | acc[gx][gy][gz].y += z[gz].y * weight[gx][gy]; | 9624 | 3.62% | 129051168 | 20.94% | selected |
| `atomic_free_spread.cu:109` | acc[gx][gy][gz].x += z[gz].x * weight[gx][gy]; | 9476 | 3.57% | 128000000 | 20.77% | selected |
| `atomic_free_spread.cu:80` | dest[local] = make_float2(ker[k] * strength.x, ker[… | 5076 | 1.91% | 7720736 | 1.25% | mio |
| `atomic_free_spread.cu:47` | dest[local] = 0.0f; | 4804 | 1.81% | 4251904 | 0.69% | mio |
| `atomic_free_spread.cu:58` | dest[local] = ker[k]; | 4615 | 1.74% | 6441748 | 1.05% | mio |
| `spreadinterp.hpp:47` | return __fmaf_rn(a, b, c); | 3334 | 1.25% | 39861600 | 6.47% | wait |
| `atomic_free_spread.cu:200` | const float raw = stage_dim == 0 ? p.x : (stage_dim… | 2453 | 0.92% | 797232 | 0.13% | long_sb |

### Top SASS opcode families

| family | instructions | inst % | samples | sample % | SASS PCs |
|---|---|---|---|---|---|
| FFMA | 295861600 | 48.01% | 22010 | 8.28% | 298 |
| LDS | 128460416 | 20.84% | 110452 | 41.57% | 135 |
| FMUL | 67851584 | 11.01% | 6004 | 2.26% | 75 |
| ISETP | 23050784 | 3.74% | 79670 | 29.98% | 116 |
| IADD3 | 19542160 | 3.17% | 2184 | 0.82% | 157 |
| BRA | 14293636 | 2.32% | 3395 | 1.28% | 28 |
| STS | 11029748 | 1.79% | 14243 | 5.36% | 48 |
| IMAD | 7520176 | 1.22% | 864 | 0.33% | 90 |
| MOV | 5230880 | 0.85% | 285 | 0.11% | 25 |
| SEL | 4967200 | 0.81% | 325 | 0.12% | 38 |
| UIMAD | 4338304 | 0.70% | 765 | 0.29% | 8 |
| UMOV | 4285344 | 0.69% | 115 | 0.04% | 7 |
| LEA | 4141908 | 0.67% | 416 | 0.16% | 31 |
| LOP3 | 4000496 | 0.65% | 285 | 0.11% | 25 |
| IMNMX | 3195072 | 0.52% | 271 | 0.10% | 15 |

### Artifacts for this case

- `reports/uniform__atomic_free.ncu-rep` (report)
- `ncu/uniform__atomic_free.raw.csv` (raw_csv)
- `ncu/uniform__atomic_free.details.csv` (details_csv)
- `ncu/uniform__atomic_free.details.txt` (details_txt)
- `ncu/uniform__atomic_free.source.csv` (source_csv)
- `ncu/uniform__atomic_free.source.txt` (source_txt)
- `analysis/uniform__atomic_free/metrics_details.csv` (metrics_details_csv)
- `analysis/uniform__atomic_free/stalls.csv` (stalls_csv)
- `analysis/uniform__atomic_free/source_hotspots.csv` (source_hotspots_csv)
- `analysis/uniform__atomic_free/source_stalls.csv` (source_stalls_csv)
- `analysis/uniform__atomic_free/source_groups.csv` (source_groups_csv)
- `analysis/uniform__atomic_free/sass_opcodes.csv` (sass_opcodes_csv)
- `analysis/uniform__atomic_free/sass_opcode_families.csv` (sass_opcode_families_csv)

## Case: `uniform__method3`

Workload: dense uniform input; all output tiles have work.  
Implementation: **cuFINUFFT method 3 (output-driven)** — Pinned cuFINUFFT reference: gpu_method=3 output-driven spread, spread-only, Horner kernel evaluation (ns=6, sigma=2).

Benchmark (same process for both implementations): atomic-free spread median **20.39 ms** (min 20.39, max 20.40); method 3 execute **19.99 ms** (setpts 7.34 ms); method-3 execute / spread **0.98x**; relL2 **5.282e-07**; checks **PASS**.

NCU: **17.90 ms**, compute **37.93%**, memory **40.76%**, DRAM **7.93%**, IPC **1.13**, occupancy **24.99%**, no-eligible **71.79%**.

### Resource and memory detail

| metric | value |
|---|---|
| elapsed cycles | 24,159,570 |
| waves per SM | 78.50 |
| memory throughput (GByte/s) | 8.87 |
| L1/TEX hit rate % | 0.78 |
| L2 hit rate % | 35.22 |
| L1 hit global loads % | 1.30 |
| L1 hit global stores % | 0 |
| avg bytes/sector load | 5.82 |
| avg bytes/sector store | 0 |
| shared-memory bank conflicts | 54,004,660 |
| shared load instructions | 56,140,672 |
| shared store instructions | 7,549,119 |
| global load instructions | 192,263 |
| global store instructions | 0 |
| global reduction (atomic) instructions | 281,344 |
| DRAM bytes read (MB) | 143.29 |
| DRAM bytes write (MB) | 15.47 |
| branch efficiency % | 95.76 |
| L1 conflicts shared N-way | 34.00 |
| L1 shared wavefronts excessive | 54,000,000 |
| L2 theoretical sectors global excessive (MB) | 5.05 |
| avg threads executed | 18524.00 |
| avg predicated-on threads executed | 17110.00 |
| block limit: shared memory | 1 |
| block limit: registers | 4 |
| theoretical active warps/SM | 8.00 |
| achieved active warps/SM | 8.00 |

### Stall breakdown

| warp state | cycles / issued inst | share of warp state | share of stalls |
|---|---|---|---|
| Stall Wait | 1.900 | 26.76% | 31.15% |
| Stall Barrier | 1.630 | 22.96% | 26.72% |
| Stall Short Scoreboard | 1.130 | 15.91% | 18.52% |
| Selected | 1.000 | 14.09% | -% |
| Stall MIO Throttle | 0.330 | 4.65% | 5.41% |
| Stall Branch Resolving | 0.290 | 4.08% | 4.75% |
| Stall Long Scoreboard | 0.240 | 3.38% | 3.93% |
| Stall No Instruction | 0.190 | 2.68% | 3.12% |
| Stall Dispatch Stall | 0.170 | 2.39% | 2.79% |
| Stall Not Selected | 0.150 | 2.11% | 2.46% |
| Stall Math Pipe Throttle | 0.070 | 0.99% | 1.15% |
| Stall Drain | 0 | 0% | 0% |

### Source groups (heuristic)

| group | instructions | inst % | samples | sample % | source lines | SASS PCs |
|---|---|---|---|---|---|---|
| other / kernel setup | 162339598 | 42.70% | 61969 | 26.34% | 20 | 178 |
| kernel eval / batch staging | 149473081 | 39.31% | 139475 | 59.29% | 13 | 76 |
| local subgrid accumulation | 52325690 | 13.76% | 22587 | 9.60% | 8 | 100 |
| global write / atomics | 8315008 | 2.19% | 785 | 0.33% | 4 | 5 |
| Horner / FMA | 5404068 | 1.42% | 7181 | 3.05% | 4 | 289 |
| support index / modulo | 1591352 | 0.42% | 1169 | 0.50% | 2 | 26 |
| fold_rescale | 408447 | 0.11% | 1747 | 0.74% | 3 | 13 |
| subproblem / subgrid setup | 351680 | 0.09% | 314 | 0.13% | 9 | 25 |

### Top source lines by instructions

| site | source | instructions | inst % | samples | sample % | group | top stall |
|---|---|---|---|---|---|---|---|
| `spread_output_driven_3d.cu:71` | for (auto i = 0; i < batch_size; i++) { | 48074616 | 12.64% | 20972 | 8.92% | kernel eval / batch staging | wait |
| `spread_output_driven_3d.cu:82` | int s = tmp % ns; | 42000000 | 11.05% | 14120 | 6.00% | other / kernel setup | short_sb |
| `spread_output_driven_3d.cu:75` | for (int idx = threadIdx.x; idx < total; idx += blo… | 38000000 | 9.99% | 12975 | 5.52% | other / kernel setup | wait |
| `array:173` | return __elems_[__n]; | 35000000 | 9.21% | 7503 | 3.19% | other / kernel setup | wait |
| `spread_output_driven_3d.cu:73` | const auto start = shift[i]; | 35000000 | 9.21% | 15709 | 6.68% | kernel eval / batch staging | wait |
| `spread_output_driven_3d.cu:92` | local_subgrid[idxout] += cnow * kervalue; | 28000000 | 7.36% | 6994 | 2.97% | local subgrid accumulation | wait |
| `spread_output_driven_3d.cu:83` | kervalue *= kerevals[i][idim][s]; | 21000000 | 5.52% | 18691 | 7.95% | kernel eval / batch staging | short_sb |
| `spread_output_driven_3d.cu:86` | tmp /= ns; | 21000000 | 5.52% | 8870 | 3.77% | other / kernel setup | wait |
| `spread_output_driven_3d.cu:60` | const auto batch_size = min(np, nupts - batch_begin… | 16099488 | 4.23% | 66962 | 28.47% | kernel eval / batch staging | barrier |
| `spread_output_driven_3d.cu:72` | const auto cnow  = nupts_sm[i]; | 14000000 | 3.68% | 10296 | 4.38% | kernel eval / batch staging | mio |
| `spread_output_driven_3d.cu:89` | kervalue *= kerevals[i][ndim - 1][tmp]; | 14000000 | 3.68% | 3294 | 1.40% | kernel eval / batch staging | wait |
| `helper_math.h:187` | a = cuCaddf(a, b); | 14000000 | 3.68% | 15657 | 6.66% | other / kernel setup | short_sb |
| `spreadinterp.hpp:254` | int idx0 = flatidx % padded_size[idim]; | 10075632 | 2.65% | 3419 | 1.45% | local subgrid accumulation | wait |
| `spread_output_driven_3d.cu:94` | __syncthreads(); | 8000000 | 2.10% | 743 | 0.32% | global write / atomics | selected |
| `spread_output_driven_3d.cu:84` | idxout += strideout * (s + start[idim] + ns_2); | 7000000 | 1.84% | 1749 | 0.74% | local subgrid accumulation | wait |

### Top source lines by PC samples

| site | source | samples | sample % | instructions | inst % | top stall |
|---|---|---|---|---|---|---|
| `spread_output_driven_3d.cu:60` | const auto batch_size = min(np, nupts - batch_begin… | 66962 | 28.47% | 16099488 | 4.23% | barrier |
| `spread_output_driven_3d.cu:71` | for (auto i = 0; i < batch_size; i++) { | 20972 | 8.92% | 48074616 | 12.64% | wait |
| `spread_output_driven_3d.cu:83` | kervalue *= kerevals[i][idim][s]; | 18691 | 7.95% | 21000000 | 5.52% | short_sb |
| `spread_output_driven_3d.cu:73` | const auto start = shift[i]; | 15709 | 6.68% | 35000000 | 9.21% | wait |
| `helper_math.h:187` | a = cuCaddf(a, b); | 15657 | 6.66% | 14000000 | 3.68% | short_sb |
| `spread_output_driven_3d.cu:82` | int s = tmp % ns; | 14120 | 6.00% | 42000000 | 11.05% | short_sb |
| `spread_output_driven_3d.cu:75` | for (int idx = threadIdx.x; idx < total; idx += blo… | 12975 | 5.52% | 38000000 | 9.99% | wait |
| `spread_output_driven_3d.cu:90` | idxout += strideout * (tmp + start[ndim - 1] + ns_2… | 10372 | 4.41% | 7000000 | 1.84% | short_sb |
| `spread_output_driven_3d.cu:72` | const auto cnow  = nupts_sm[i]; | 10296 | 4.38% | 14000000 | 3.68% | mio |
| `spread_output_driven_3d.cu:86` | tmp /= ns; | 8870 | 3.77% | 21000000 | 5.52% | wait |
| `array:173` | return __elems_[__n]; | 7503 | 3.19% | 35000000 | 9.21% | wait |
| `spread_output_driven_3d.cu:92` | local_subgrid[idxout] += cnow * kervalue; | 6994 | 2.97% | 28000000 | 7.36% | wait |

### Top SASS opcode families

| family | instructions | inst % | samples | sample % | SASS PCs |
|---|---|---|---|---|---|
| IMAD | 58998205 | 15.52% | 18200 | 7.74% | 61 |
| LDS | 56140672 | 14.77% | 29533 | 12.55% | 9 |
| LEA | 42117976 | 11.08% | 11385 | 4.84% | 10 |
| IADD3 | 40819982 | 10.74% | 25070 | 10.66% | 85 |
| ISETP | 25954643 | 6.83% | 9679 | 4.12% | 46 |
| BRA | 23584685 | 6.20% | 16681 | 7.09% | 19 |
| FFMA | 18712850 | 4.92% | 18732 | 7.96% | 260 |
| MOV | 15934235 | 4.19% | 2063 | 0.88% | 54 |
| FMUL | 14094257 | 3.71% | 18086 | 7.69% | 5 |
| BSYNC | 9065414 | 2.38% | 4025 | 1.71% | 4 |
| BMOV | 8065083 | 2.12% | 2852 | 1.21% | 4 |
| BSSY | 8065083 | 2.12% | 692 | 0.29% | 4 |
| IMNMX | 8058536 | 2.12% | 57244 | 24.34% | 4 |
| BAR | 8033664 | 2.11% | 743 | 0.32% | 3 |
| UIADD3 | 8033664 | 2.11% | 687 | 0.29% | 3 |

### Artifacts for this case

- `reports/uniform__method3.ncu-rep` (report)
- `ncu/uniform__method3.raw.csv` (raw_csv)
- `ncu/uniform__method3.details.csv` (details_csv)
- `ncu/uniform__method3.details.txt` (details_txt)
- `ncu/uniform__method3.source.csv` (source_csv)
- `ncu/uniform__method3.source.txt` (source_txt)
- `analysis/uniform__method3/metrics_details.csv` (metrics_details_csv)
- `analysis/uniform__method3/stalls.csv` (stalls_csv)
- `analysis/uniform__method3/source_hotspots.csv` (source_hotspots_csv)
- `analysis/uniform__method3/source_stalls.csv` (source_stalls_csv)
- `analysis/uniform__method3/source_groups.csv` (source_groups_csv)
- `analysis/uniform__method3/sass_opcodes.csv` (sass_opcodes_csv)
- `analysis/uniform__method3/sass_opcode_families.csv` (sass_opcode_families_csv)

## Case: `clustered__atomic_free`

Workload: eight tight clusters; few hot output tiles.  
Implementation: **atomic-free spread** — Output-cell-owned atomic-free prototype: every fine-grid cell is owned by one thread and stored once, with no spreading atomics.

Benchmark (same process for both implementations): atomic-free spread median **26.00 ms** (min 25.90, max 29.38); method 3 execute **20.11 ms** (setpts 0.93 ms); method-3 execute / spread **0.77x**; relL2 **6.525e-06**; checks **PASS**.

NCU: **26.03 ms**, compute **49.30%**, memory **49.30%**, DRAM **4.74%**, IPC **1.52**, occupancy **43.21%**, no-eligible **62.33%**.

### Resource and memory detail

| metric | value |
|---|---|
| elapsed cycles | 35,123,775 |
| waves per SM | 4.57 |
| memory throughput (GByte/s) | 5.41 |
| L1/TEX hit rate % | 83.36 |
| L2 hit rate % | 31.22 |
| L1 hit global loads % | 83.46 |
| L1 hit global stores % | 54.30 |
| avg bytes/sector load | 8.85 |
| avg bytes/sector store | 16.00 |
| shared-memory bank conflicts | 2,764,833 |
| shared load instructions | 128,397,504 |
| shared store instructions | 8,855,144 |
| global load instructions | 1,762,528 |
| global store instructions | 8,192 |
| global reduction (atomic) instructions | 0 |
| DRAM bytes read (MB) | 138.28 |
| DRAM bytes write (MB) | 2.46 |
| branch efficiency % | 99.99 |
| L1 conflicts shared N-way | 220.00 |
| L1 shared wavefronts excessive | 2,702,948 |
| L2 theoretical sectors global excessive (MB) | 26.07 |
| avg threads executed | 28812.00 |
| avg predicated-on threads executed | 27604.00 |
| block limit: shared memory | 13 |
| block limit: registers | 8 |
| theoretical active warps/SM | 32.00 |
| achieved active warps/SM | 13.83 |

### Stall breakdown

| warp state | cycles / issued inst | share of warp state | share of stalls |
|---|---|---|---|
| Stall MIO Throttle | 3.710 | 40.50% | 45.47% |
| Stall Barrier | 1.780 | 19.43% | 21.81% |
| Selected | 1.000 | 10.92% | -% |
| Stall Wait | 0.950 | 10.37% | 11.64% |
| Stall Long Scoreboard | 0.630 | 6.88% | 7.72% |
| Stall Not Selected | 0.590 | 6.44% | 7.23% |
| Stall Math Pipe Throttle | 0.180 | 1.97% | 2.21% |
| Stall Short Scoreboard | 0.140 | 1.53% | 1.72% |
| Stall No Instruction | 0.120 | 1.31% | 1.47% |
| Stall Branch Resolving | 0.050 | 0.55% | 0.61% |
| Stall Dispatch Stall | 0.010 | 0.11% | 0.12% |
| Stall Drain | 0 | 0% | 0% |

### Source groups (heuristic)

| group | instructions | inst % | samples | sample % | source lines | SASS PCs |
|---|---|---|---|---|---|---|
| weight / select / accumulate | 403016832 | 66.57% | 104547 | 39.01% | 4 | 400 |
| Horner / FMA | 96011200 | 15.86% | 44763 | 16.70% | 3 | 112 |
| other / kernel setup | 67094560 | 11.08% | 43693 | 16.30% | 40 | 377 |
| weight staging | 18505132 | 3.06% | 14901 | 5.56% | 4 | 75 |
| bin iteration / offsets | 10228576 | 1.69% | 26689 | 9.96% | 18 | 190 |
| point load | 4001024 | 0.66% | 27246 | 10.17% | 1 | 6 |
| fold_rescale | 3000768 | 0.50% | 1434 | 0.54% | 3 | 4 |
| global write / atomics | 2002560 | 0.33% | 4186 | 1.56% | 3 | 3 |
| ceilf | 1500384 | 0.25% | 540 | 0.20% | 1 | 2 |
| support index / modulo | 7680 | 0.00% | 0 | 0% | 2 | 15 |

### Top source lines by instructions

| site | source | instructions | inst % | samples | sample % | group | top stall |
|---|---|---|---|---|---|---|---|
| `atomic_free_spread.cu:97` | weight[gx][gy] = kx[j][x0 + gx] * ky[j][y0 + gy]; | 146000256 | 24.12% | 71126 | 26.54% | weight / select / accumulate | mio |
| `atomic_free_spread.cu:110` | acc[gx][gy][gz].y += z[gz].y * weight[gx][gy]; | 129000192 | 21.31% | 17112 | 6.39% | weight / select / accumulate | selected |
| `atomic_free_spread.cu:109` | acc[gx][gy][gz].x += z[gz].x * weight[gx][gy]; | 128000000 | 21.14% | 16301 | 6.08% | weight / select / accumulate | selected |
| `atomic_free_spread.cu:101` | for (int gz = 0; gz < GROUP_Z; ++gz) z[gz] = kz[j][… | 54000448 | 8.92% | 36490 | 13.62% | Horner / FMA | mio |
| `spreadinterp.hpp:47` | return __fmaf_rn(a, b, c); | 37509600 | 6.20% | 6277 | 2.34% | Horner / FMA | wait |
| `atomic_free_spread.cu:91` | for (int j = 0; j < staged; ++j) { | 15002240 | 2.48% | 29984 | 11.19% | other / kernel setup | barrier |
| `atomic_free_spread.cu:197` | const int staged = min(STAGE_POINTS, bin_end - base… | 10002560 | 1.65% | 2351 | 0.88% | other / kernel setup | wait |
| `atomic_free_spread.cu:94` | for (int gx = 0; gx < GROUP_X; ++gx) | 10002432 | 1.65% | 4350 | 1.62% | other / kernel setup | wait |
| `atomic_free_spread.cu:55` | cell -= (cell >= fine_size) ? fine_size : 0; | 9002304 | 1.49% | 1091 | 0.41% | other / kernel setup | selected |
| `atomic_free_spread.cu:80` | dest[local] = make_float2(ker[k] * strength.x, ker[… | 6627032 | 1.09% | 3898 | 1.45% | weight staging | mio |
| `atomic_free_spread.cu:58` | dest[local] = ker[k]; | 5876564 | 0.97% | 4529 | 1.69% | weight staging | mio |
| `atomic_free_spread.cu:77` | cell -= (cell >= fine_size) ? fine_size : 0; | 4501152 | 0.74% | 471 | 0.18% | other / kernel setup | selected |
| `atomic_free_spread.cu:206` | eval_kernel_vec_horner<float, WIDTH>(ker, float(sta… | 4501152 | 0.74% | 1996 | 0.74% | Horner / FMA | short_sb |
| `atomic_free_spread.cu:196` | for (int base = bin_begin; base < bin_end; base += … | 4263808 | 0.70% | 24316 | 9.07% | bin iteration / offsets | barrier |
| `atomic_free_spread.cu:47` | dest[local] = 0.0f; | 4001024 | 0.66% | 4764 | 1.78% | weight staging | mio |

### Top source lines by PC samples

| site | source | samples | sample % | instructions | inst % | top stall |
|---|---|---|---|---|---|---|
| `atomic_free_spread.cu:97` | weight[gx][gy] = kx[j][x0 + gx] * ky[j][y0 + gy]; | 71126 | 26.54% | 146000256 | 24.12% | mio |
| `atomic_free_spread.cu:101` | for (int gz = 0; gz < GROUP_Z; ++gz) z[gz] = kz[j][… | 36490 | 13.62% | 54000448 | 8.92% | mio |
| `atomic_free_spread.cu:91` | for (int j = 0; j < staged; ++j) { | 29984 | 11.19% | 15002240 | 2.48% | barrier |
| `atomic_free_spread.cu:199` | const Point p = points[base + stage_point]; | 27246 | 10.17% | 4001024 | 0.66% | long_sb |
| `atomic_free_spread.cu:196` | for (int base = bin_begin; base < bin_end; base += … | 24316 | 9.07% | 4263808 | 0.70% | barrier |
| `atomic_free_spread.cu:110` | acc[gx][gy][gz].y += z[gz].y * weight[gx][gy]; | 17112 | 6.39% | 129000192 | 21.31% | selected |
| `atomic_free_spread.cu:109` | acc[gx][gy][gz].x += z[gz].x * weight[gx][gy]; | 16301 | 6.08% | 128000000 | 21.14% | selected |
| `spreadinterp.hpp:47` | return __fmaf_rn(a, b, c); | 6277 | 2.34% | 37509600 | 6.20% | wait |
| `atomic_free_spread.cu:47` | dest[local] = 0.0f; | 4764 | 1.78% | 4001024 | 0.66% | mio |
| `atomic_free_spread.cu:58` | dest[local] = ker[k]; | 4529 | 1.69% | 5876564 | 0.97% | mio |
| `atomic_free_spread.cu:94` | for (int gx = 0; gx < GROUP_X; ++gx) | 4350 | 1.62% | 10002432 | 1.65% | wait |
| `atomic_free_spread.cu:80` | dest[local] = make_float2(ker[k] * strength.x, ker[… | 3898 | 1.45% | 6627032 | 1.09% | mio |

### Top SASS opcode families

| family | instructions | inst % | samples | sample % | SASS PCs |
|---|---|---|---|---|---|
| FFMA | 293509600 | 48.48% | 38759 | 14.46% | 298 |
| LDS | 128397504 | 21.21% | 97848 | 36.51% | 135 |
| FMUL | 67500912 | 11.15% | 9998 | 3.73% | 75 |
| ISETP | 21737408 | 3.59% | 52682 | 19.66% | 116 |
| IADD3 | 18495984 | 3.06% | 4251 | 1.59% | 157 |
| BRA | 13350876 | 2.21% | 6882 | 2.57% | 28 |
| STS | 10255724 | 1.69% | 12200 | 4.55% | 48 |
| IMAD | 7096816 | 1.17% | 1369 | 0.51% | 90 |
| MOV | 4687136 | 0.77% | 649 | 0.24% | 25 |
| SEL | 4684960 | 0.77% | 639 | 0.24% | 38 |
| UMOV | 4002752 | 0.66% | 254 | 0.10% | 7 |
| UIMAD | 4000512 | 0.66% | 1776 | 0.66% | 8 |
| LEA | 3775564 | 0.62% | 681 | 0.25% | 31 |
| LOP3 | 3765296 | 0.62% | 544 | 0.20% | 25 |
| IMNMX | 3006912 | 0.50% | 589 | 0.22% | 15 |

### Artifacts for this case

- `reports/clustered__atomic_free.ncu-rep` (report)
- `ncu/clustered__atomic_free.raw.csv` (raw_csv)
- `ncu/clustered__atomic_free.details.csv` (details_csv)
- `ncu/clustered__atomic_free.details.txt` (details_txt)
- `ncu/clustered__atomic_free.source.csv` (source_csv)
- `ncu/clustered__atomic_free.source.txt` (source_txt)
- `analysis/clustered__atomic_free/metrics_details.csv` (metrics_details_csv)
- `analysis/clustered__atomic_free/stalls.csv` (stalls_csv)
- `analysis/clustered__atomic_free/source_hotspots.csv` (source_hotspots_csv)
- `analysis/clustered__atomic_free/source_stalls.csv` (source_stalls_csv)
- `analysis/clustered__atomic_free/source_groups.csv` (source_groups_csv)
- `analysis/clustered__atomic_free/sass_opcodes.csv` (sass_opcodes_csv)
- `analysis/clustered__atomic_free/sass_opcode_families.csv` (sass_opcode_families_csv)

## Case: `clustered__method3`

Workload: eight tight clusters; few hot output tiles.  
Implementation: **cuFINUFFT method 3 (output-driven)** — Pinned cuFINUFFT reference: gpu_method=3 output-driven spread, spread-only, Horner kernel evaluation (ns=6, sigma=2).

Benchmark (same process for both implementations): atomic-free spread median **26.00 ms** (min 25.90, max 29.38); method 3 execute **20.11 ms** (setpts 0.93 ms); method-3 execute / spread **0.77x**; relL2 **6.525e-06**; checks **PASS**.

NCU: **17.88 ms**, compute **37.99%**, memory **40.84%**, DRAM **6.91%**, IPC **1.13**, occupancy **25.05%**, no-eligible **71.85%**.

### Resource and memory detail

| metric | value |
|---|---|
| elapsed cycles | 24,129,501 |
| waves per SM | 70.86 |
| memory throughput (GByte/s) | 7.72 |
| L1/TEX hit rate % | 0.61 |
| L2 hit rate % | 37.11 |
| L1 hit global loads % | 1.01 |
| L1 hit global stores % | 0 |
| avg bytes/sector load | 5.78 |
| avg bytes/sector store | 0 |
| shared-memory bank conflicts | 54,015,646 |
| shared load instructions | 56,126,976 |
| shared store instructions | 7,533,395 |
| global load instructions | 188,059 |
| global store instructions | 0 |
| global reduction (atomic) instructions | 253,952 |
| DRAM bytes read (MB) | 134.69 |
| DRAM bytes write (MB) | 3.31 |
| branch efficiency % | 95.75 |
| L1 conflicts shared N-way | 34.00 |
| L1 shared wavefronts excessive | 54,000,000 |
| L2 theoretical sectors global excessive (MB) | 4.91 |
| avg threads executed | 18524.00 |
| avg predicated-on threads executed | 17130.00 |
| block limit: shared memory | 1 |
| block limit: registers | 4 |
| theoretical active warps/SM | 8.00 |
| achieved active warps/SM | 8.02 |

### Stall breakdown

| warp state | cycles / issued inst | share of warp state | share of stalls |
|---|---|---|---|
| Stall Wait | 1.900 | 26.80% | 31.20% |
| Stall Barrier | 1.630 | 22.99% | 26.77% |
| Stall Short Scoreboard | 1.140 | 16.08% | 18.72% |
| Selected | 1.000 | 14.10% | -% |
| Stall MIO Throttle | 0.340 | 4.79% | 5.58% |
| Stall Branch Resolving | 0.290 | 4.09% | 4.76% |
| Stall Long Scoreboard | 0.220 | 3.10% | 3.61% |
| Stall No Instruction | 0.190 | 2.68% | 3.12% |
| Stall Dispatch Stall | 0.170 | 2.40% | 2.79% |
| Stall Not Selected | 0.140 | 1.98% | 2.30% |
| Stall Math Pipe Throttle | 0.070 | 0.99% | 1.15% |
| Stall Drain | 0 | 0% | 0% |

### Source groups (heuristic)

| group | instructions | inst % | samples | sample % | source lines | SASS PCs |
|---|---|---|---|---|---|---|
| other / kernel setup | 161871982 | 42.77% | 61812 | 26.35% | 20 | 178 |
| kernel eval / batch staging | 149440518 | 39.48% | 139639 | 59.52% | 13 | 76 |
| local subgrid accumulation | 51337786 | 13.56% | 22384 | 9.54% | 8 | 100 |
| global write / atomics | 8285432 | 2.19% | 727 | 0.31% | 4 | 5 |
| Horner / FMA | 5377236 | 1.42% | 6898 | 2.94% | 4 | 289 |
| support index / modulo | 1436416 | 0.38% | 1228 | 0.52% | 2 | 26 |
| fold_rescale | 406419 | 0.11% | 1628 | 0.69% | 3 | 13 |
| subproblem / subgrid setup | 317440 | 0.08% | 279 | 0.12% | 9 | 25 |

### Top source lines by instructions

| site | source | instructions | inst % | samples | sample % | group | top stall |
|---|---|---|---|---|---|---|---|
| `spread_output_driven_3d.cu:71` | for (auto i = 0; i < batch_size; i++) { | 48070632 | 12.70% | 21000 | 8.95% | kernel eval / batch staging | wait |
| `spread_output_driven_3d.cu:82` | int s = tmp % ns; | 42000000 | 11.10% | 14397 | 6.14% | other / kernel setup | short_sb |
| `spread_output_driven_3d.cu:75` | for (int idx = threadIdx.x; idx < total; idx += blo… | 38000000 | 10.04% | 12991 | 5.54% | other / kernel setup | wait |
| `array:173` | return __elems_[__n]; | 35000000 | 9.25% | 7471 | 3.18% | other / kernel setup | wait |
| `spread_output_driven_3d.cu:73` | const auto start = shift[i]; | 35000000 | 9.25% | 15577 | 6.64% | kernel eval / batch staging | wait |
| `spread_output_driven_3d.cu:92` | local_subgrid[idxout] += cnow * kervalue; | 28000000 | 7.40% | 6860 | 2.92% | local subgrid accumulation | wait |
| `spread_output_driven_3d.cu:83` | kervalue *= kerevals[i][idim][s]; | 21000000 | 5.55% | 18914 | 8.06% | kernel eval / batch staging | short_sb |
| `spread_output_driven_3d.cu:86` | tmp /= ns; | 21000000 | 5.55% | 8733 | 3.72% | other / kernel setup | wait |
| `spread_output_driven_3d.cu:60` | const auto batch_size = min(np, nupts - batch_begin… | 16094176 | 4.25% | 66985 | 28.55% | kernel eval / batch staging | barrier |
| `spread_output_driven_3d.cu:72` | const auto cnow  = nupts_sm[i]; | 14000000 | 3.70% | 10378 | 4.42% | kernel eval / batch staging | mio |
| `spread_output_driven_3d.cu:89` | kervalue *= kerevals[i][ndim - 1][tmp]; | 14000000 | 3.70% | 3295 | 1.40% | kernel eval / batch staging | wait |
| `helper_math.h:187` | a = cuCaddf(a, b); | 14000000 | 3.70% | 15539 | 6.62% | other / kernel setup | short_sb |
| `spreadinterp.hpp:254` | int idx0 = flatidx % padded_size[idim]; | 9094656 | 2.40% | 2997 | 1.28% | local subgrid accumulation | wait |
| `spread_output_driven_3d.cu:94` | __syncthreads(); | 8000000 | 2.11% | 708 | 0.30% | global write / atomics | selected |
| `spread_output_driven_3d.cu:84` | idxout += strideout * (s + start[idim] + ns_2); | 7000000 | 1.85% | 1625 | 0.69% | local subgrid accumulation | wait |

### Top source lines by PC samples

| site | source | samples | sample % | instructions | inst % | top stall |
|---|---|---|---|---|---|---|
| `spread_output_driven_3d.cu:60` | const auto batch_size = min(np, nupts - batch_begin… | 66985 | 28.55% | 16094176 | 4.25% | barrier |
| `spread_output_driven_3d.cu:71` | for (auto i = 0; i < batch_size; i++) { | 21000 | 8.95% | 48070632 | 12.70% | wait |
| `spread_output_driven_3d.cu:83` | kervalue *= kerevals[i][idim][s]; | 18914 | 8.06% | 21000000 | 5.55% | short_sb |
| `spread_output_driven_3d.cu:73` | const auto start = shift[i]; | 15577 | 6.64% | 35000000 | 9.25% | wait |
| `helper_math.h:187` | a = cuCaddf(a, b); | 15539 | 6.62% | 14000000 | 3.70% | short_sb |
| `spread_output_driven_3d.cu:82` | int s = tmp % ns; | 14397 | 6.14% | 42000000 | 11.10% | short_sb |
| `spread_output_driven_3d.cu:75` | for (int idx = threadIdx.x; idx < total; idx += blo… | 12991 | 5.54% | 38000000 | 10.04% | wait |
| `spread_output_driven_3d.cu:90` | idxout += strideout * (tmp + start[ndim - 1] + ns_2… | 10846 | 4.62% | 7000000 | 1.85% | short_sb |
| `spread_output_driven_3d.cu:72` | const auto cnow  = nupts_sm[i]; | 10378 | 4.42% | 14000000 | 3.70% | mio |
| `spread_output_driven_3d.cu:86` | tmp /= ns; | 8733 | 3.72% | 21000000 | 5.55% | wait |
| `array:173` | return __elems_[__n]; | 7471 | 3.18% | 35000000 | 9.25% | wait |
| `spread_output_driven_3d.cu:92` | local_subgrid[idxout] += cnow * kervalue; | 6860 | 2.92% | 28000000 | 7.40% | wait |

### Top SASS opcode families

| family | instructions | inst % | samples | sample % | SASS PCs |
|---|---|---|---|---|---|
| IMAD | 58726617 | 15.52% | 18158 | 7.74% | 61 |
| LDS | 56126976 | 14.83% | 29423 | 12.54% | 9 |
| LEA | 42117257 | 11.13% | 11456 | 4.88% | 10 |
| IADD3 | 40369463 | 10.67% | 25410 | 10.83% | 85 |
| ISETP | 25673159 | 6.78% | 9711 | 4.14% | 46 |
| BRA | 23551361 | 6.22% | 16637 | 7.09% | 19 |
| FFMA | 18689450 | 4.94% | 18147 | 7.74% | 260 |
| MOV | 15793577 | 4.17% | 1946 | 0.83% | 54 |
| FMUL | 14093789 | 3.72% | 18280 | 7.79% | 5 |
| BSYNC | 9062769 | 2.40% | 4035 | 1.72% | 4 |
| BMOV | 8062743 | 2.13% | 2799 | 1.19% | 4 |
| BSSY | 8062743 | 2.13% | 735 | 0.31% | 4 |
| IMNMX | 8055024 | 2.13% | 57124 | 24.35% | 4 |
| BAR | 8031480 | 2.12% | 708 | 0.30% | 3 |
| UIADD3 | 8031480 | 2.12% | 713 | 0.30% | 3 |

### Artifacts for this case

- `reports/clustered__method3.ncu-rep` (report)
- `ncu/clustered__method3.raw.csv` (raw_csv)
- `ncu/clustered__method3.details.csv` (details_csv)
- `ncu/clustered__method3.details.txt` (details_txt)
- `ncu/clustered__method3.source.csv` (source_csv)
- `ncu/clustered__method3.source.txt` (source_txt)
- `analysis/clustered__method3/metrics_details.csv` (metrics_details_csv)
- `analysis/clustered__method3/stalls.csv` (stalls_csv)
- `analysis/clustered__method3/source_hotspots.csv` (source_hotspots_csv)
- `analysis/clustered__method3/source_stalls.csv` (source_stalls_csv)
- `analysis/clustered__method3/source_groups.csv` (source_groups_csv)
- `analysis/clustered__method3/sass_opcodes.csv` (sass_opcodes_csv)
- `analysis/clustered__method3/sass_opcode_families.csv` (sass_opcode_families_csv)

## Case: `sparse__atomic_free`

Workload: sparse uniform input on a larger grid.  
Implementation: **atomic-free spread** — Output-cell-owned atomic-free prototype: every fine-grid cell is owned by one thread and stored once, with no spreading atomics.

Benchmark (same process for both implementations): atomic-free spread median **4.29 ms** (min 4.29, max 4.30); method 3 execute **5.10 ms** (setpts 7.57 ms); method-3 execute / spread **1.19x**; relL2 **7.958e-08**; checks **PASS**.

NCU: **4.29 ms**, compute **75.73%**, memory **75.73%**, DRAM **4.26%**, IPC **2.66**, occupancy **98.91%**, no-eligible **33.43%**.

### Resource and memory detail

| metric | value |
|---|---|
| elapsed cycles | 5,794,234 |
| waves per SM | 36.57 |
| memory throughput (GByte/s) | 4.76 |
| L1/TEX hit rate % | 83.39 |
| L2 hit rate % | 92.98 |
| L1 hit global loads % | 89.29 |
| L1 hit global stores % | 49.66 |
| avg bytes/sector load | 6.65 |
| avg bytes/sector store | 16.00 |
| shared-memory bank conflicts | 137,809 |
| shared load instructions | 16,452,448 |
| shared store instructions | 8,768,392 |
| global load instructions | 3,591,920 |
| global store instructions | 65,536 |
| global reduction (atomic) instructions | 0 |
| DRAM bytes read (MB) | 3.83 |
| DRAM bytes write (MB) | 16.61 |
| branch efficiency % | 93.29 |
| L1 conflicts shared N-way | 119.00 |
| L1 shared wavefronts excessive | 92,064 |
| L2 theoretical sectors global excessive (MB) | 2.79 |
| avg threads executed | 14058.00 |
| avg predicated-on threads executed | 13489.00 |
| block limit: shared memory | 13 |
| block limit: registers | 8 |
| theoretical active warps/SM | 32.00 |
| achieved active warps/SM | 31.65 |

### Stall breakdown

| warp state | cycles / issued inst | share of warp state | share of stalls |
|---|---|---|---|
| Stall Barrier | 2.660 | 22.39% | 24.45% |
| Stall Wait | 1.910 | 16.08% | 17.55% |
| Stall Not Selected | 1.440 | 12.12% | 13.23% |
| Stall MIO Throttle | 1.370 | 11.53% | 12.59% |
| Stall Long Scoreboard | 1.340 | 11.28% | 12.32% |
| Selected | 1.000 | 8.42% | -% |
| Stall Short Scoreboard | 0.950 | 8.00% | 8.73% |
| Stall Math Pipe Throttle | 0.670 | 5.64% | 6.16% |
| Stall No Instruction | 0.260 | 2.19% | 2.39% |
| Stall Branch Resolving | 0.170 | 1.43% | 1.56% |
| Stall Dispatch Stall | 0.100 | 0.84% | 0.92% |
| Stall Drain | 0.010 | 0.08% | 0.09% |

### Source groups (heuristic)

| group | instructions | inst % | samples | sample % | source lines | SASS PCs |
|---|---|---|---|---|---|---|
| other / kernel setup | 69807984 | 32.42% | 20258 | 35.87% | 40 | 377 |
| Horner / FMA | 49581888 | 23.03% | 5863 | 10.38% | 3 | 112 |
| weight / select / accumulate | 40650880 | 18.88% | 7503 | 13.29% | 4 | 400 |
| bin iteration / offsets | 25263408 | 11.73% | 12041 | 21.32% | 18 | 190 |
| weight staging | 19481384 | 9.05% | 3932 | 6.96% | 4 | 75 |
| point load | 3986048 | 1.85% | 4821 | 8.54% | 1 | 6 |
| fold_rescale | 2989536 | 1.39% | 688 | 1.22% | 3 | 4 |
| global write / atomics | 2009408 | 0.93% | 1073 | 1.90% | 3 | 3 |
| ceilf | 1494768 | 0.69% | 297 | 0.53% | 1 | 2 |
| support index / modulo | 61440 | 0.03% | 1 | 0.00% | 2 | 15 |

### Top source lines by instructions

| site | source | instructions | inst % | samples | sample % | group | top stall |
|---|---|---|---|---|---|---|---|
| `spreadinterp.hpp:47` | return __fmaf_rn(a, b, c); | 37369200 | 17.35% | 3375 | 5.98% | Horner / FMA | wait |
| `atomic_free_spread.cu:97` | weight[gx][gy] = kx[j][x0 + gx] * ky[j][y0 + gy]; | 14536800 | 6.75% | 4997 | 8.85% | weight / select / accumulate | mio |
| `atomic_free_spread.cu:110` | acc[gx][gy][gz].y += z[gz].y * weight[gx][gy]; | 13183008 | 6.12% | 1112 | 1.97% | weight / select / accumulate | selected |
| `atomic_free_spread.cu:109` | acc[gx][gy][gz].x += z[gz].x * weight[gx][gy]; | 12800000 | 5.94% | 1374 | 2.43% | weight / select / accumulate | wait |
| `atomic_free_spread.cu:197` | const int staged = min(STAGE_POINTS, bin_end - base… | 9965120 | 4.63% | 1023 | 1.81% | other / kernel setup | wait |
| `atomic_free_spread.cu:55` | cell -= (cell >= fine_size) ? fine_size : 0; | 8968608 | 4.17% | 501 | 0.89% | other / kernel setup | selected |
| `atomic_free_spread.cu:91` | for (int j = 0; j < staged; ++j) { | 8494144 | 3.94% | 10765 | 19.06% | other / kernel setup | barrier |
| `atomic_free_spread.cu:94` | for (int gx = 0; gx < GROUP_X; ++gx) | 8318944 | 3.86% | 1479 | 2.62% | other / kernel setup | wait |
| `atomic_free_spread.cu:196` | for (int base = bin_begin; base < bin_end; base += … | 8076224 | 3.75% | 6419 | 11.37% | bin iteration / offsets | barrier |
| `atomic_free_spread.cu:101` | for (int gz = 0; gz < GROUP_Z; ++gz) z[gz] = kz[j][… | 7728384 | 3.59% | 1570 | 2.78% | Horner / FMA | mio |
| `atomic_free_spread.cu:80` | dest[local] = make_float2(ker[k] * strength.x, ker[… | 7553208 | 3.51% | 1642 | 2.91% | weight staging | mio |
| `atomic_free_spread.cuh:91` | if (bin < 0) bin += bins; | 6291456 | 2.92% | 1602 | 2.84% | other / kernel setup | wait |
| `atomic_free_spread.cu:58` | dest[local] = ker[k]; | 5949104 | 2.76% | 1042 | 1.84% | weight staging | mio |
| `atomic_free_spread.cu:189` | for (int x_bin_index = 0; x_bin_index < x_bins.coun… | 4863136 | 2.26% | 2096 | 3.71% | bin iteration / offsets | short_sb |
| `atomic_free_spread.cu:77` | cell -= (cell >= fine_size) ? fine_size : 0; | 4484304 | 2.08% | 228 | 0.40% | other / kernel setup | selected |

### Top source lines by PC samples

| site | source | samples | sample % | instructions | inst % | top stall |
|---|---|---|---|---|---|---|
| `atomic_free_spread.cu:91` | for (int j = 0; j < staged; ++j) { | 10765 | 19.06% | 8494144 | 3.94% | barrier |
| `atomic_free_spread.cu:196` | for (int base = bin_begin; base < bin_end; base += … | 6419 | 11.37% | 8076224 | 3.75% | barrier |
| `atomic_free_spread.cu:97` | weight[gx][gy] = kx[j][x0 + gx] * ky[j][y0 + gy]; | 4997 | 8.85% | 14536800 | 6.75% | mio |
| `atomic_free_spread.cu:199` | const Point p = points[base + stage_point]; | 4821 | 8.54% | 3986048 | 1.85% | long_sb |
| `spreadinterp.hpp:47` | return __fmaf_rn(a, b, c); | 3375 | 5.98% | 37369200 | 17.35% | wait |
| `atomic_free_spread.cu:189` | for (int x_bin_index = 0; x_bin_index < x_bins.coun… | 2096 | 3.71% | 4863136 | 2.26% | short_sb |
| `atomic_free_spread.cu:80` | dest[local] = make_float2(ker[k] * strength.x, ker[… | 1642 | 2.91% | 7553208 | 3.51% | mio |
| `atomic_free_spread.cuh:91` | if (bin < 0) bin += bins; | 1602 | 2.84% | 6291456 | 2.92% | wait |
| `atomic_free_spread.cu:101` | for (int gz = 0; gz < GROUP_Z; ++gz) z[gz] = kz[j][… | 1570 | 2.78% | 7728384 | 3.59% | mio |
| `atomic_free_spread.cuh:90` | int bin = first + i; | 1513 | 2.68% | 2424832 | 1.13% | short_sb |
| `atomic_free_spread.cu:94` | for (int gx = 0; gx < GROUP_X; ++gx) | 1479 | 2.62% | 8318944 | 3.86% | wait |
| `atomic_free_spread.cu:109` | acc[gx][gy][gz].x += z[gz].x * weight[gx][gy]; | 1374 | 2.43% | 12800000 | 5.94% | wait |

### Top SASS opcode families

| family | instructions | inst % | samples | sample % | SASS PCs |
|---|---|---|---|---|---|
| FFMA | 62969200 | 29.24% | 5686 | 10.07% | 298 |
| ISETP | 24059552 | 11.17% | 18325 | 32.45% | 116 |
| IADD3 | 21399736 | 9.94% | 3359 | 5.95% | 157 |
| LDS | 16452448 | 7.64% | 6476 | 11.47% | 135 |
| BRA | 13074380 | 6.07% | 3137 | 5.55% | 28 |
| STS | 10301844 | 4.78% | 2479 | 4.39% | 48 |
| FMUL | 9950768 | 4.62% | 2733 | 4.84% | 75 |
| IMAD | 9487160 | 4.41% | 1784 | 3.16% | 90 |
| MOV | 7960080 | 3.70% | 583 | 1.03% | 25 |
| SEL | 5954768 | 2.77% | 716 | 1.27% | 38 |
| LEA | 3996548 | 1.86% | 372 | 0.66% | 31 |
| LOP3 | 3851608 | 1.79% | 287 | 0.51% | 25 |
| LDG | 3841048 | 1.78% | 5366 | 9.50% | 5 |
| IMNMX | 3038688 | 1.41% | 282 | 0.50% | 15 |
| UMOV | 2586816 | 1.20% | 58 | 0.10% | 7 |

### Artifacts for this case

- `reports/sparse__atomic_free.ncu-rep` (report)
- `ncu/sparse__atomic_free.raw.csv` (raw_csv)
- `ncu/sparse__atomic_free.details.csv` (details_csv)
- `ncu/sparse__atomic_free.details.txt` (details_txt)
- `ncu/sparse__atomic_free.source.csv` (source_csv)
- `ncu/sparse__atomic_free.source.txt` (source_txt)
- `analysis/sparse__atomic_free/metrics_details.csv` (metrics_details_csv)
- `analysis/sparse__atomic_free/stalls.csv` (stalls_csv)
- `analysis/sparse__atomic_free/source_hotspots.csv` (source_hotspots_csv)
- `analysis/sparse__atomic_free/source_stalls.csv` (source_stalls_csv)
- `analysis/sparse__atomic_free/source_groups.csv` (source_groups_csv)
- `analysis/sparse__atomic_free/sass_opcodes.csv` (sass_opcodes_csv)
- `analysis/sparse__atomic_free/sass_opcode_families.csv` (sass_opcode_families_csv)

## Case: `sparse__method3`

Workload: sparse uniform input on a larger grid.  
Implementation: **cuFINUFFT method 3 (output-driven)** — Pinned cuFINUFFT reference: gpu_method=3 output-driven spread, spread-only, Horner kernel evaluation (ns=6, sigma=2).

Benchmark (same process for both implementations): atomic-free spread median **4.29 ms** (min 4.29, max 4.30); method 3 execute **5.10 ms** (setpts 7.57 ms); method-3 execute / spread **1.19x**; relL2 **7.958e-08**; checks **PASS**.

NCU: **2.97 ms**, compute **33.37%**, memory **33.08%**, DRAM **20.85%**, IPC **1.28**, occupancy **24.92%**, no-eligible **67.88%**.

### Resource and memory detail

| metric | value |
|---|---|
| elapsed cycles | 4,003,196 |
| waves per SM | 156.93 |
| memory throughput (GByte/s) | 23.28 |
| L1/TEX hit rate % | 1.05 |
| L2 hit rate % | 79.00 |
| L1 hit global loads % | 13.21 |
| L1 hit global stores % | 0 |
| avg bytes/sector load | 5.52 |
| avg bytes/sector store | 0 |
| shared-memory bank conflicts | 5,465,012 |
| shared load instructions | 5,881,216 |
| shared store instructions | 1,036,999 |
| global load instructions | 91,759 |
| global store instructions | 0 |
| global reduction (atomic) instructions | 562,432 |
| DRAM bytes read (MB) | 40.69 |
| DRAM bytes write (MB) | 28.37 |
| branch efficiency % | 96.55 |
| L1 conflicts shared N-way | 34.00 |
| L1 shared wavefronts excessive | 5,400,000 |
| L2 theoretical sectors global excessive (MB) | 3.71 |
| avg threads executed | 16577.00 |
| avg predicated-on threads executed | 15185.00 |
| block limit: shared memory | 1 |
| block limit: registers | 4 |
| theoretical active warps/SM | 8.00 |
| achieved active warps/SM | 7.98 |

### Stall breakdown

| warp state | cycles / issued inst | share of warp state | share of stalls |
|---|---|---|---|
| Stall Wait | 1.830 | 29.52% | 35.19% |
| Stall Barrier | 1.400 | 22.58% | 26.92% |
| Selected | 1.000 | 16.13% | -% |
| Stall Short Scoreboard | 0.590 | 9.52% | 11.35% |
| Stall Long Scoreboard | 0.260 | 4.19% | 5.00% |
| Stall Not Selected | 0.250 | 4.03% | 4.81% |
| Stall Math Pipe Throttle | 0.210 | 3.39% | 4.04% |
| Stall Dispatch Stall | 0.200 | 3.23% | 3.85% |
| Stall Branch Resolving | 0.180 | 2.90% | 3.46% |
| Stall MIO Throttle | 0.130 | 2.10% | 2.50% |
| Stall No Instruction | 0.120 | 1.94% | 2.31% |
| Stall Drain | 0.030 | 0.48% | 0.58% |

### Source groups (heuristic)

| group | instructions | inst % | samples | sample % | source lines | SASS PCs |
|---|---|---|---|---|---|---|
| other / kernel setup | 25321006 | 35.55% | 8773 | 22.62% | 20 | 178 |
| local subgrid accumulation | 24490874 | 34.39% | 8373 | 21.59% | 8 | 100 |
| kernel eval / batch staging | 15334718 | 21.53% | 17106 | 44.12% | 13 | 76 |
| support index / modulo | 3181256 | 4.47% | 2586 | 6.67% | 2 | 26 |
| global write / atomics | 1397584 | 1.96% | 158 | 0.41% | 4 | 5 |
| Horner / FMA | 738052 | 1.04% | 902 | 2.33% | 4 | 289 |
| subproblem / subgrid setup | 703040 | 0.99% | 667 | 1.72% | 9 | 25 |
| fold_rescale | 55783 | 0.08% | 211 | 0.54% | 3 | 13 |

### Top source lines by instructions

| site | source | instructions | inst % | samples | sample % | group | top stall |
|---|---|---|---|---|---|---|---|
| `spreadinterp.hpp:254` | int idx0 = flatidx % padded_size[idim]; | 20142096 | 28.28% | 6679 | 17.22% | local subgrid accumulation | wait |
| `spread_output_driven_3d.cu:71` | for (auto i = 0; i < batch_size; i++) { | 4852728 | 6.81% | 2058 | 5.31% | kernel eval / batch staging | wait |
| `spreadinterp.hpp:256` | idx      = idx < 0 ? idx + nf[idim] : (idx >= nf[id… | 4798248 | 6.74% | 1142 | 2.95% | other / kernel setup | selected |
| `spread_output_driven_3d.cu:82` | int s = tmp % ns; | 4200000 | 5.90% | 1258 | 3.24% | other / kernel setup | short_sb |
| `spread_output_driven_3d.cu:75` | for (int idx = threadIdx.x; idx < total; idx += blo… | 3800000 | 5.34% | 1341 | 3.46% | other / kernel setup | wait |
| `array:173` | return __elems_[__n]; | 3500000 | 4.91% | 733 | 1.89% | other / kernel setup | wait |
| `spread_output_driven_3d.cu:73` | const auto start = shift[i]; | 3500000 | 4.91% | 1559 | 4.02% | kernel eval / batch staging | wait |
| `spread_output_driven_3d.cu:92` | local_subgrid[idxout] += cnow * kervalue; | 2800000 | 3.93% | 689 | 1.78% | local subgrid accumulation | wait |
| `spread_output_driven_3d.cu:98` | for (int n = threadIdx.x; n < local_subgrid_size; n… | 2232152 | 3.13% | 1884 | 4.86% | support index / modulo | wait |
| `spread_output_driven_3d.cu:83` | kervalue *= kerevals[i][idim][s]; | 2100000 | 2.95% | 1768 | 4.56% | kernel eval / batch staging | short_sb |
| `spread_output_driven_3d.cu:86` | tmp /= ns; | 2100000 | 2.95% | 1005 | 2.59% | other / kernel setup | wait |
| `spreadinterp.hpp:255` | int idx  = idx0 + offset[idim] - ns_2; | 1687296 | 2.37% | 239 | 0.62% | other / kernel setup | selected |
| `spread_output_driven_3d.cu:60` | const auto batch_size = min(np, nupts - batch_begin… | 1670304 | 2.35% | 10037 | 25.88% | kernel eval / batch staging | barrier |
| `spread_output_driven_3d.cu:72` | const auto cnow  = nupts_sm[i]; | 1400000 | 1.97% | 823 | 2.12% | kernel eval / batch staging | mio |
| `spread_output_driven_3d.cu:89` | kervalue *= kerevals[i][ndim - 1][tmp]; | 1400000 | 1.97% | 347 | 0.89% | kernel eval / batch staging | wait |

### Top source lines by PC samples

| site | source | samples | sample % | instructions | inst % | top stall |
|---|---|---|---|---|---|---|
| `spread_output_driven_3d.cu:60` | const auto batch_size = min(np, nupts - batch_begin… | 10037 | 25.88% | 1670304 | 2.35% | barrier |
| `spreadinterp.hpp:254` | int idx0 = flatidx % padded_size[idim]; | 6679 | 17.22% | 20142096 | 28.28% | wait |
| `spread_output_driven_3d.cu:71` | for (auto i = 0; i < batch_size; i++) { | 2058 | 5.31% | 4852728 | 6.81% | wait |
| `spread_output_driven_3d.cu:98` | for (int n = threadIdx.x; n < local_subgrid_size; n… | 1884 | 4.86% | 2232152 | 3.13% | wait |
| `spread_output_driven_3d.cu:83` | kervalue *= kerevals[i][idim][s]; | 1768 | 4.56% | 2100000 | 2.95% | short_sb |
| `helper_math.h:187` | a = cuCaddf(a, b); | 1669 | 4.30% | 1400000 | 1.97% | short_sb |
| `spread_output_driven_3d.cu:73` | const auto start = shift[i]; | 1559 | 4.02% | 3500000 | 4.91% | wait |
| `spread_output_driven_3d.cu:75` | for (int idx = threadIdx.x; idx < total; idx += blo… | 1341 | 3.46% | 3800000 | 5.34% | wait |
| `spread_output_driven_3d.cu:82` | int s = tmp % ns; | 1258 | 3.24% | 4200000 | 5.90% | short_sb |
| `spreadinterp.hpp:256` | idx      = idx < 0 ? idx + nf[idim] : (idx >= nf[id… | 1142 | 2.95% | 4798248 | 6.74% | selected |
| `spread_output_driven_3d.cu:86` | tmp /= ns; | 1005 | 2.59% | 2100000 | 2.95% | wait |
| `spread_output_driven_3d.cu:72` | const auto cnow  = nupts_sm[i]; | 823 | 2.12% | 1400000 | 1.97% | mio |

### Top SASS opcode families

| family | instructions | inst % | samples | sample % | SASS PCs |
|---|---|---|---|---|---|
| IADD3 | 12835570 | 18.02% | 5940 | 15.32% | 85 |
| IMAD | 11184053 | 15.70% | 3317 | 8.55% | 61 |
| ISETP | 8051643 | 11.30% | 2528 | 6.52% | 46 |
| LDS | 5881216 | 8.26% | 2706 | 6.98% | 9 |
| MOV | 4302683 | 6.04% | 1004 | 2.59% | 54 |
| LEA | 4217164 | 5.92% | 1213 | 3.13% | 10 |
| BRA | 2962773 | 4.16% | 2341 | 6.04% | 19 |
| IABS | 2724280 | 3.83% | 808 | 2.08% | 20 |
| FFMA | 2043650 | 2.87% | 2164 | 5.58% | 260 |
| SEL | 1968512 | 2.76% | 322 | 0.83% | 7 |
| LOP3 | 1546688 | 2.17% | 635 | 1.64% | 14 |
| FMUL | 1412873 | 1.98% | 1728 | 4.46% | 5 |
| STS | 1036999 | 1.46% | 642 | 1.66% | 15 |
| BSYNC | 941595 | 1.32% | 562 | 1.45% | 4 |
| IMNMX | 852728 | 1.20% | 9022 | 23.27% | 4 |

### Artifacts for this case

- `reports/sparse__method3.ncu-rep` (report)
- `ncu/sparse__method3.raw.csv` (raw_csv)
- `ncu/sparse__method3.details.csv` (details_csv)
- `ncu/sparse__method3.details.txt` (details_txt)
- `ncu/sparse__method3.source.csv` (source_csv)
- `ncu/sparse__method3.source.txt` (source_txt)
- `analysis/sparse__method3/metrics_details.csv` (metrics_details_csv)
- `analysis/sparse__method3/stalls.csv` (stalls_csv)
- `analysis/sparse__method3/source_hotspots.csv` (source_hotspots_csv)
- `analysis/sparse__method3/source_stalls.csv` (source_stalls_csv)
- `analysis/sparse__method3/source_groups.csv` (source_groups_csv)
- `analysis/sparse__method3/sass_opcodes.csv` (sass_opcodes_csv)
- `analysis/sparse__method3/sass_opcode_families.csv` (sass_opcode_families_csv)

## Case: `dense__atomic_free`

Workload: one dense cluster; long same-tile accumulation.  
Implementation: **atomic-free spread** — Output-cell-owned atomic-free prototype: every fine-grid cell is owned by one thread and stored once, with no spreading atomics.

Benchmark (same process for both implementations): atomic-free spread median **7.59 ms** (min 7.19, max 8.03); method 3 execute **4.07 ms** (setpts 4.10 ms); method-3 execute / spread **0.54x**; relL2 **8.649e-06**; checks **PASS**.

NCU: **7.60 ms**, compute **21.50%**, memory **21.50%**, DRAM **0.56%**, IPC **1.16**, occupancy **18.92%**, no-eligible **68.04%**.

### Resource and memory detail

| metric | value |
|---|---|
| elapsed cycles | 10,250,224 |
| waves per SM | 4.57 |
| memory throughput (GByte/s) | 0.81 |
| L1/TEX hit rate % | 83.30 |
| L2 hit rate % | 85.73 |
| L1 hit global loads % | 84.27 |
| L1 hit global stores % | 55.04 |
| avg bytes/sector load | 8.56 |
| avg bytes/sector store | 16.00 |
| shared-memory bank conflicts | 4,582 |
| shared load instructions | 13,197,344 |
| shared store instructions | 828,072 |
| global load instructions | 412,144 |
| global store instructions | 8,192 |
| global reduction (atomic) instructions | 0 |
| DRAM bytes read (MB) | 3.77 |
| DRAM bytes write (MB) | 2.37 |
| branch efficiency % | 99.97 |
| L1 conflicts shared N-way | 138.00 |
| L1 shared wavefronts excessive | 0 |
| L2 theoretical sectors global excessive (MB) | 2.67 |
| avg threads executed | 21714.00 |
| avg predicated-on threads executed | 20587.00 |
| block limit: shared memory | 13 |
| block limit: registers | 8 |
| theoretical active warps/SM | 32.00 |
| achieved active warps/SM | 6.05 |

### Stall breakdown

| warp state | cycles / issued inst | share of warp state | share of stalls |
|---|---|---|---|
| Stall Wait | 1.080 | 22.93% | 29.11% |
| Selected | 1.000 | 21.23% | -% |
| Stall MIO Throttle | 0.870 | 18.47% | 23.45% |
| Stall Long Scoreboard | 0.630 | 13.38% | 16.98% |
| Stall Barrier | 0.480 | 10.19% | 12.94% |
| Stall Short Scoreboard | 0.230 | 4.88% | 6.20% |
| Stall Not Selected | 0.170 | 3.61% | 4.58% |
| Stall No Instruction | 0.110 | 2.33% | 2.96% |
| Stall Branch Resolving | 0.060 | 1.27% | 1.62% |
| Stall Math Pipe Throttle | 0.040 | 0.85% | 1.08% |
| Stall Dispatch Stall | 0.020 | 0.42% | 0.54% |
| Stall Drain | 0.010 | 0.21% | 0.27% |

### Source groups (heuristic)

| group | instructions | inst % | samples | sample % | source lines | SASS PCs |
|---|---|---|---|---|---|---|
| weight / select / accumulate | 40316384 | 63.46% | 10771 | 38.89% | 4 | 400 |
| Horner / FMA | 9600000 | 15.11% | 4319 | 15.60% | 3 | 112 |
| other / kernel setup | 8129520 | 12.80% | 6224 | 22.48% | 40 | 377 |
| bin iteration / offsets | 2575536 | 4.05% | 2276 | 8.22% | 18 | 190 |
| weight staging | 1850000 | 2.91% | 635 | 2.29% | 4 | 75 |
| point load | 400000 | 0.63% | 2890 | 10.44% | 1 | 6 |
| fold_rescale | 300000 | 0.47% | 198 | 0.71% | 3 | 4 |
| global write / atomics | 202048 | 0.32% | 301 | 1.09% | 3 | 3 |
| ceilf | 150000 | 0.24% | 79 | 0.28% | 1 | 2 |
| support index / modulo | 7680 | 0.01% | 0 | 0% | 2 | 15 |

### Top source lines by instructions

| site | source | instructions | inst % | samples | sample % | group | top stall |
|---|---|---|---|---|---|---|---|
| `atomic_free_spread.cu:97` | weight[gx][gy] = kx[j][x0 + gx] * ky[j][y0 + gy]; | 14600000 | 22.98% | 5910 | 21.34% | weight / select / accumulate | mio |
| `atomic_free_spread.cu:110` | acc[gx][gy][gz].y += z[gz].y * weight[gx][gy]; | 12900000 | 20.30% | 2451 | 8.85% | weight / select / accumulate | selected |
| `atomic_free_spread.cu:109` | acc[gx][gy][gz].x += z[gz].x * weight[gx][gy]; | 12800000 | 20.15% | 2406 | 8.69% | weight / select / accumulate | selected |
| `atomic_free_spread.cu:101` | for (int gz = 0; gz < GROUP_Z; ++gz) z[gz] = kz[j][… | 5400000 | 8.50% | 3196 | 11.54% | Horner / FMA | mio |
| `spreadinterp.hpp:47` | return __fmaf_rn(a, b, c); | 3750000 | 5.90% | 818 | 2.95% | Horner / FMA | wait |
| `atomic_free_spread.cu:91` | for (int j = 0; j < staged; ++j) { | 1500000 | 2.36% | 3855 | 13.92% | other / kernel setup | barrier |
| `atomic_free_spread.cu:94` | for (int gx = 0; gx < GROUP_X; ++gx) | 1000000 | 1.57% | 724 | 2.61% | other / kernel setup | wait |
| `atomic_free_spread.cu:197` | const int staged = min(STAGE_POINTS, bin_end - base… | 1000000 | 1.57% | 396 | 1.43% | other / kernel setup | wait |
| `atomic_free_spread.cu:55` | cell -= (cell >= fine_size) ? fine_size : 0; | 900000 | 1.42% | 129 | 0.47% | other / kernel setup | selected |
| `atomic_free_spread.cuh:91` | if (bin < 0) bin += bins; | 786432 | 1.24% | 193 | 0.70% | other / kernel setup | wait |
| `atomic_free_spread.cu:80` | dest[local] = make_float2(ker[k] * strength.x, ker[… | 662500 | 1.04% | 279 | 1.01% | weight staging | selected |
| `atomic_free_spread.cu:196` | for (int base = bin_begin; base < bin_end; base += … | 662208 | 1.04% | 1633 | 5.90% | bin iteration / offsets | barrier |
| `atomic_free_spread.cu:58` | dest[local] = ker[k]; | 587500 | 0.92% | 149 | 0.54% | weight staging | wait |
| `atomic_free_spread.cu:189` | for (int x_bin_index = 0; x_bin_index < x_bins.coun… | 483360 | 0.76% | 153 | 0.55% | bin iteration / offsets | wait |
| `atomic_free_spread.cu:77` | cell -= (cell >= fine_size) ? fine_size : 0; | 450000 | 0.71% | 90 | 0.33% | other / kernel setup | selected |

### Top source lines by PC samples

| site | source | samples | sample % | instructions | inst % | top stall |
|---|---|---|---|---|---|---|
| `atomic_free_spread.cu:97` | weight[gx][gy] = kx[j][x0 + gx] * ky[j][y0 + gy]; | 5910 | 21.34% | 14600000 | 22.98% | mio |
| `atomic_free_spread.cu:91` | for (int j = 0; j < staged; ++j) { | 3855 | 13.92% | 1500000 | 2.36% | barrier |
| `atomic_free_spread.cu:101` | for (int gz = 0; gz < GROUP_Z; ++gz) z[gz] = kz[j][… | 3196 | 11.54% | 5400000 | 8.50% | mio |
| `atomic_free_spread.cu:199` | const Point p = points[base + stage_point]; | 2890 | 10.44% | 400000 | 0.63% | long_sb |
| `atomic_free_spread.cu:110` | acc[gx][gy][gz].y += z[gz].y * weight[gx][gy]; | 2451 | 8.85% | 12900000 | 20.30% | selected |
| `atomic_free_spread.cu:109` | acc[gx][gy][gz].x += z[gz].x * weight[gx][gy]; | 2406 | 8.69% | 12800000 | 20.15% | selected |
| `atomic_free_spread.cu:196` | for (int base = bin_begin; base < bin_end; base += … | 1633 | 5.90% | 662208 | 1.04% | barrier |
| `spreadinterp.hpp:47` | return __fmaf_rn(a, b, c); | 818 | 2.95% | 3750000 | 5.90% | wait |
| `atomic_free_spread.cu:94` | for (int gx = 0; gx < GROUP_X; ++gx) | 724 | 2.61% | 1000000 | 1.57% | wait |
| `atomic_free_spread.cu:197` | const int staged = min(STAGE_POINTS, bin_end - base… | 396 | 1.43% | 1000000 | 1.57% | wait |
| `atomic_free_spread.cu:206` | eval_kernel_vec_horner<float, WIDTH>(ker, float(sta… | 305 | 1.10% | 450000 | 0.71% | short_sb |
| `atomic_free_spread.cu:80` | dest[local] = make_float2(ker[k] * strength.x, ker[… | 279 | 1.01% | 662500 | 1.04% | selected |

### Top SASS opcode families

| family | instructions | inst % | samples | sample % | SASS PCs |
|---|---|---|---|---|---|
| FFMA | 29350000 | 46.20% | 5536 | 19.99% | 298 |
| LDS | 13197344 | 20.77% | 7783 | 28.11% | 135 |
| FMUL | 6750000 | 10.62% | 1415 | 5.11% | 75 |
| ISETP | 2832672 | 4.46% | 5203 | 18.79% | 116 |
| IADD3 | 2516888 | 3.96% | 893 | 3.23% | 157 |
| BRA | 1648160 | 2.59% | 1291 | 4.66% | 28 |
| STS | 1028072 | 1.62% | 357 | 1.29% | 48 |
| IMAD | 1020088 | 1.61% | 172 | 0.62% | 90 |
| MOV | 635408 | 1.00% | 109 | 0.39% | 25 |
| SEL | 633808 | 1.00% | 124 | 0.45% | 38 |
| LDG | 437144 | 0.69% | 2916 | 10.53% | 5 |
| UMOV | 402048 | 0.63% | 44 | 0.16% | 7 |
| UIMAD | 400000 | 0.63% | 280 | 1.01% | 8 |
| LEA | 399576 | 0.63% | 113 | 0.41% | 31 |
| LOP3 | 389336 | 0.61% | 90 | 0.33% | 25 |

### Artifacts for this case

- `reports/dense__atomic_free.ncu-rep` (report)
- `ncu/dense__atomic_free.raw.csv` (raw_csv)
- `ncu/dense__atomic_free.details.csv` (details_csv)
- `ncu/dense__atomic_free.details.txt` (details_txt)
- `ncu/dense__atomic_free.source.csv` (source_csv)
- `ncu/dense__atomic_free.source.txt` (source_txt)
- `analysis/dense__atomic_free/metrics_details.csv` (metrics_details_csv)
- `analysis/dense__atomic_free/stalls.csv` (stalls_csv)
- `analysis/dense__atomic_free/source_hotspots.csv` (source_hotspots_csv)
- `analysis/dense__atomic_free/source_stalls.csv` (source_stalls_csv)
- `analysis/dense__atomic_free/source_groups.csv` (source_groups_csv)
- `analysis/dense__atomic_free/sass_opcodes.csv` (sass_opcodes_csv)
- `analysis/dense__atomic_free/sass_opcode_families.csv` (sass_opcode_families_csv)

## Case: `dense__method3`

Workload: one dense cluster; long same-tile accumulation.  
Implementation: **cuFINUFFT method 3 (output-driven)** — Pinned cuFINUFFT reference: gpu_method=3 output-driven spread, spread-only, Horner kernel evaluation (ns=6, sigma=2).

Benchmark (same process for both implementations): atomic-free spread median **7.59 ms** (min 7.19, max 8.03); method 3 execute **4.07 ms** (setpts 4.10 ms); method-3 execute / spread **0.54x**; relL2 **8.649e-06**; checks **PASS**.

NCU: **1.78 ms**, compute **38.04%**, memory **40.65%**, DRAM **1.36%**, IPC **1.13**, occupancy **25.00%**, no-eligible **71.74%**.

### Resource and memory detail

| metric | value |
|---|---|
| elapsed cycles | 2,407,227 |
| waves per SM | 7.00 |
| memory throughput (GByte/s) | 1.52 |
| L1/TEX hit rate % | 27.50 |
| L2 hit rate % | 77.19 |
| L1 hit global loads % | 62.34 |
| L1 hit global stores % | 0 |
| avg bytes/sector load | 11.84 |
| avg bytes/sector store | 0 |
| shared-memory bank conflicts | 5,406,985 |
| shared load instructions | 5,612,544 |
| shared store instructions | 753,169 |
| global load instructions | 18,761 |
| global store instructions | 0 |
| global reduction (atomic) instructions | 25,088 |
| DRAM bytes read (MB) | 2.49 |
| DRAM bytes write (MB) | 227.81 |
| branch efficiency % | 95.75 |
| L1 conflicts shared N-way | 44.00 |
| L1 shared wavefronts excessive | 5,400,000 |
| L2 theoretical sectors global excessive (MB) | 276.23 |
| avg threads executed | 18768.00 |
| avg predicated-on threads executed | 17334.00 |
| block limit: shared memory | 1 |
| block limit: registers | 4 |
| theoretical active warps/SM | 8.00 |
| achieved active warps/SM | 8.00 |

### Stall breakdown

| warp state | cycles / issued inst | share of warp state | share of stalls |
|---|---|---|---|
| Stall Wait | 1.900 | 26.84% | 31.25% |
| Stall Barrier | 1.610 | 22.74% | 26.48% |
| Stall Short Scoreboard | 1.160 | 16.38% | 19.08% |
| Selected | 1.000 | 14.12% | -% |
| Stall MIO Throttle | 0.350 | 4.94% | 5.76% |
| Stall Branch Resolving | 0.290 | 4.10% | 4.77% |
| Stall Long Scoreboard | 0.190 | 2.68% | 3.12% |
| Stall No Instruction | 0.190 | 2.68% | 3.12% |
| Stall Dispatch Stall | 0.170 | 2.40% | 2.80% |
| Stall Not Selected | 0.140 | 1.98% | 2.30% |
| Stall Math Pipe Throttle | 0.070 | 0.99% | 1.15% |
| Stall Drain | 0.010 | 0.14% | 0.16% |

### Source groups (heuristic)

| group | instructions | inst % | samples | sample % | source lines | SASS PCs |
|---|---|---|---|---|---|---|
| other / kernel setup | 16181962 | 42.78% | 6059 | 25.96% | 20 | 178 |
| kernel eval / batch staging | 14943772 | 39.50% | 13999 | 59.98% | 13 | 76 |
| local subgrid accumulation | 5122702 | 13.54% | 2280 | 9.77% | 8 | 100 |
| global write / atomics | 828216 | 2.19% | 98 | 0.42% | 4 | 5 |
| Horner / FMA | 537500 | 1.42% | 562 | 2.41% | 4 | 289 |
| support index / modulo | 141904 | 0.38% | 149 | 0.64% | 2 | 26 |
| fold_rescale | 40625 | 0.11% | 150 | 0.64% | 3 | 13 |
| subproblem / subgrid setup | 31360 | 0.08% | 43 | 0.18% | 9 | 25 |

### Top source lines by instructions

| site | source | instructions | inst % | samples | sample % | group | top stall |
|---|---|---|---|---|---|---|---|
| `spread_output_driven_3d.cu:71` | for (auto i = 0; i < batch_size; i++) { | 4807032 | 12.71% | 2140 | 9.17% | kernel eval / batch staging | wait |
| `spread_output_driven_3d.cu:82` | int s = tmp % ns; | 4200000 | 11.10% | 1450 | 6.21% | other / kernel setup | short_sb |
| `spread_output_driven_3d.cu:75` | for (int idx = threadIdx.x; idx < total; idx += blo… | 3800000 | 10.05% | 1297 | 5.56% | other / kernel setup | wait |
| `array:173` | return __elems_[__n]; | 3500000 | 9.25% | 749 | 3.21% | other / kernel setup | wait |
| `spread_output_driven_3d.cu:73` | const auto start = shift[i]; | 3500000 | 9.25% | 1462 | 6.26% | kernel eval / batch staging | wait |
| `spread_output_driven_3d.cu:92` | local_subgrid[idxout] += cnow * kervalue; | 2800000 | 7.40% | 695 | 2.98% | local subgrid accumulation | wait |
| `spread_output_driven_3d.cu:83` | kervalue *= kerevals[i][idim][s]; | 2100000 | 5.55% | 1761 | 7.54% | kernel eval / batch staging | short_sb |
| `spread_output_driven_3d.cu:86` | tmp /= ns; | 2100000 | 5.55% | 784 | 3.36% | other / kernel setup | wait |
| `spread_output_driven_3d.cu:60` | const auto batch_size = min(np, nupts - batch_begin… | 1609376 | 4.25% | 6864 | 29.41% | kernel eval / batch staging | barrier |
| `spread_output_driven_3d.cu:72` | const auto cnow  = nupts_sm[i]; | 1400000 | 3.70% | 1072 | 4.59% | kernel eval / batch staging | mio |
| `spread_output_driven_3d.cu:89` | kervalue *= kerevals[i][ndim - 1][tmp]; | 1400000 | 3.70% | 342 | 1.47% | kernel eval / batch staging | wait |
| `helper_math.h:187` | a = cuCaddf(a, b); | 1400000 | 3.70% | 1518 | 6.50% | other / kernel setup | short_sb |
| `spreadinterp.hpp:254` | int idx0 = flatidx % padded_size[idim]; | 898464 | 2.38% | 285 | 1.22% | local subgrid accumulation | wait |
| `spread_output_driven_3d.cu:94` | __syncthreads(); | 800000 | 2.11% | 89 | 0.38% | global write / atomics | selected |
| `spread_output_driven_3d.cu:84` | idxout += strideout * (s + start[idim] + ns_2); | 700000 | 1.85% | 156 | 0.67% | local subgrid accumulation | wait |

### Top source lines by PC samples

| site | source | samples | sample % | instructions | inst % | top stall |
|---|---|---|---|---|---|---|
| `spread_output_driven_3d.cu:60` | const auto batch_size = min(np, nupts - batch_begin… | 6864 | 29.41% | 1609376 | 4.25% | barrier |
| `spread_output_driven_3d.cu:71` | for (auto i = 0; i < batch_size; i++) { | 2140 | 9.17% | 4807032 | 12.71% | wait |
| `spread_output_driven_3d.cu:83` | kervalue *= kerevals[i][idim][s]; | 1761 | 7.54% | 2100000 | 5.55% | short_sb |
| `helper_math.h:187` | a = cuCaddf(a, b); | 1518 | 6.50% | 1400000 | 3.70% | short_sb |
| `spread_output_driven_3d.cu:73` | const auto start = shift[i]; | 1462 | 6.26% | 3500000 | 9.25% | wait |
| `spread_output_driven_3d.cu:82` | int s = tmp % ns; | 1450 | 6.21% | 4200000 | 11.10% | short_sb |
| `spread_output_driven_3d.cu:75` | for (int idx = threadIdx.x; idx < total; idx += blo… | 1297 | 5.56% | 3800000 | 10.05% | wait |
| `spread_output_driven_3d.cu:90` | idxout += strideout * (tmp + start[ndim - 1] + ns_2… | 1140 | 4.88% | 700000 | 1.85% | short_sb |
| `spread_output_driven_3d.cu:72` | const auto cnow  = nupts_sm[i]; | 1072 | 4.59% | 1400000 | 3.70% | mio |
| `spread_output_driven_3d.cu:86` | tmp /= ns; | 784 | 3.36% | 2100000 | 5.55% | wait |
| `array:173` | return __elems_[__n]; | 749 | 3.21% | 3500000 | 9.25% | wait |
| `spread_output_driven_3d.cu:92` | local_subgrid[idxout] += cnow * kervalue; | 695 | 2.98% | 2800000 | 7.40% | wait |

### Top SASS opcode families

| family | instructions | inst % | samples | sample % | SASS PCs |
|---|---|---|---|---|---|
| IMAD | 5869619 | 15.52% | 1814 | 7.77% | 61 |
| LDS | 5612544 | 14.84% | 2883 | 12.35% | 9 |
| LEA | 4211719 | 11.13% | 1133 | 4.85% | 10 |
| IADD3 | 4031905 | 10.66% | 2627 | 11.26% | 85 |
| ISETP | 2564173 | 6.78% | 989 | 4.24% | 46 |
| BRA | 2354779 | 6.22% | 1641 | 7.03% | 19 |
| FFMA | 1868750 | 4.94% | 1601 | 6.86% | 260 |
| MOV | 1577787 | 4.17% | 204 | 0.87% | 54 |
| FMUL | 1409375 | 3.73% | 1712 | 7.33% | 5 |
| BSYNC | 906253 | 2.40% | 429 | 1.84% | 4 |
| BMOV | 806253 | 2.13% | 306 | 1.31% | 4 |
| BSSY | 806253 | 2.13% | 73 | 0.31% | 4 |
| IMNMX | 805472 | 2.13% | 5838 | 25.01% | 4 |
| BAR | 803128 | 2.12% | 90 | 0.39% | 3 |
| UIADD3 | 803128 | 2.12% | 57 | 0.24% | 3 |

### Artifacts for this case

- `reports/dense__method3.ncu-rep` (report)
- `ncu/dense__method3.raw.csv` (raw_csv)
- `ncu/dense__method3.details.csv` (details_csv)
- `ncu/dense__method3.details.txt` (details_txt)
- `ncu/dense__method3.source.csv` (source_csv)
- `ncu/dense__method3.source.txt` (source_txt)
- `analysis/dense__method3/metrics_details.csv` (metrics_details_csv)
- `analysis/dense__method3/stalls.csv` (stalls_csv)
- `analysis/dense__method3/source_hotspots.csv` (source_hotspots_csv)
- `analysis/dense__method3/source_stalls.csv` (source_stalls_csv)
- `analysis/dense__method3/source_groups.csv` (source_groups_csv)
- `analysis/dense__method3/sass_opcodes.csv` (sass_opcodes_csv)
- `analysis/dense__method3/sass_opcode_families.csv` (sass_opcode_families_csv)

## Method notes

- Two NCU reports are captured per workload: `atomic_free` profiles the first exact `spread` launch (`assign_bins` is filtered out); `method3` profiles the first `spread_output_driven` launch made by the in-process cuFINUFFT method-3 reference. Method-3 preprocessing kernels from `setpts` are not profiled.
- The cuFINUFFT static library was rebuilt with `-lineinfo` so method-3 source attribution is available; line info does not change the generated SASS.
- The capture was taken with uncommitted changes in `devel/morton_spread/atomic_free_spread.cu` (see Git status above), so the atomic-free numbers describe that working tree, not HEAD.
- The benchmark binary always runs both implementations, so the benchmark and relL2 columns are shared by the two reports for a workload.
- NCU runs with `--clock-control none` and kernel replay. Absolute durations differ from benchmark medians; utilization, instruction, and stall metrics are the useful part.
- Benchmark medians are short, unlocked-clock samples and vary between runs. Treat the NCU kernel comparisons as the stable kernel-level result and the wall-clock benchmark as indicative.
- Method-3 benchmark `execute` includes the output-grid clear and the spread kernel, so the NCU kernel duration is not directly comparable to that wall-clock number.
- Source instructions and PC samples are deduplicated by `(kernel, SASS PC)`, keeping the first source correlation. Nsight repeats PCs under inline definitions and call sites.
- Source-group percentages are a source-text heuristic, not a compiler data structure. They are meant to keep the high-level cost distribution visible across line-number drift.
- Stall shares are `Warp State (All Cycles)` values divided by all warp-state cycles. `share of stalls` excludes the `Selected` row.
- `raw.csv` contains every exported raw metric, including semicolon-packed lists that the detail CSV cannot expand into named columns.

