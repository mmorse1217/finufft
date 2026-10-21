# ISSUE-002 profile artifacts: method-3 spread-only baseline

`build/` is ignored by `.gitignore`, so a git-aware file browser may hide it. The
profile artifacts from the method-3 baseline have therefore been copied here, under
the (untracked) issue-tracker research directory.

## Files

- `dense_api.nsys-rep` — Nsight Systems report, dense case.
- `sparse_api.nsys-rep` — Nsight Systems report, sparse case.
- `dense_api_flamegraph.svg` — CPU NVTX ranges plus GPU activity for one median
  iteration, dense case. Open in a browser.
- `sparse_api_flamegraph.svg` — same, sparse case.
- `dense_api_stats.txt` — `nsys stats` output: NVTX summary, NVTX-kernel summary,
  CUDA kernel summary, CUDA API summary.
- `sparse_api_stats.txt` — same, sparse case.
- `dense_ncu.txt`, `sparse_ncu.txt` — Nsight Compute SpeedOfLight, occupancy,
  scheduler, instruction, and launch statistics.
- `dense_stalls.txt`, `sparse_stalls.txt` — Nsight Compute warp-stall metric dump.
- `tools/bench_spread.cpp` — the throwaway benchmark used for the run.
- `tools/make_flame.py` — renders the NVTX trace plus GPU activities as an SVG.
- `../ISSUE-002-profile.tar.gz` — all of the above in one archive.

## Cases

Both are 3D type 1, FP32, `ns=6`, `gpu_spreadinterponly=1`, `gpu_method=3`, uniform
points, T1000:

| Case | Fine grid | M | Grid occupancy |
|---|---|---:|---:|
| dense | `64^3` | 1,000,000 | 3.8 points/cell |
| sparse | `128^3` | 100,000 | 0.048 points/cell |

## What the profile shows

Dense case, one median iteration (19.71 ms):

```text
iteration                            19.71 ms
├─ execute                           18.67 ms  94.7%
│  └─ spread_output_driven           18.64 ms  94.6%
├─ setpts                             0.97 ms   4.9%
└─ makeplan                           0.08 ms   0.4%
```

Sparse case, one median iteration (3.55 ms):

```text
iteration                             3.55 ms
├─ execute                            3.27 ms  92.3%
│  └─ spread_output_driven            3.09 ms  87.3%
├─ setpts                              0.19 ms   5.4%
└─ makeplan                            0.08 ms   2.2%
```

The spread kernel is the dominant cost. Preprocessing, including the existing bin
histogram and inverse permutation, is about 5% of the iteration. A CUB Morton sort
would add to that, but it is not the main obstacle at these sizes; the kernel is.

## Bound analysis

Nsight Compute, `--clock-control none`:

| Metric | Dense `64^3`, M=1e6 | Sparse `128^3`, M=1e5 |
|---|---:|---:|
| DRAM throughput | 8.98% | 23.35% |
| L1/TEX throughput | 81.11% | 56.64% |
| L2 throughput | 4.20% | 32.92% |
| Compute (SM) throughput | 37.74% | 33.02% |
| Achieved occupancy | 25.0% | 25.0% |
| Dynamic shared memory/block | 65.15 KB | 65.15 KB |
| Registers/thread | 51 | 51 |
| No-eligible-warp cycles | 71.9% | 68.2% |

Highest compute pipeline was ALU at 21.4% in the dense case; no compute pipe is
saturated. Top stall ratios:

- Dense: `wait` 1.90, `barrier` 1.64, `short_scoreboard` 1.12, `mio_throttle` 0.33,
  `long_scoreboard` 0.26.
- Sparse: `wait` 1.83, `barrier` 1.45, `short_scoreboard` 0.59,
  `long_scoreboard` 0.27, `math_pipe` 0.21.

Conclusion: the kernel is **not DRAM-bandwidth bound and not compute bound**. It is
**occupancy / barrier / shared-memory-latency bound**. The dense case is closest to an
L1/shared-memory throughput limit (81% L1/TEX); the sparse case adds global-memory
work from the padded-tile merge (23% DRAM, 33% L2) but remains latency-bound. The
65 KB shared tile limits the kernel to one block per SM, i.e. 25% occupancy.

## Commands

Configure/build used the existing GPU build:

```bash
export PATH=$HOME/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-12.8/lib64:$LD_LIBRARY_PATH

cmake -S . -B build/gpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$HOME/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DFINUFFT_USE_CPU=OFF -DFINUFFT_USE_CUDA=ON \
  -DBUILD_TESTING=ON -DFINUFFT_BUILD_TESTS=ON \
  -DCPM_CCCL_SOURCE=/tmp/cccl-3.4.2
cmake --build build/gpu --target cufinufft cuperftest -j8
```

Benchmark app:

```bash
g++ -O3 -std=c++17 -Iinclude -I$HOME/cuda-12.8/include \
  docs/issues/research/ISSUE-002-profile/tools/bench_spread.cpp \
  -o /tmp/bench_spread \
  build/gpu/libcufinufft.a build/gpu/src/common/libfinufft_common.a \
  -L$HOME/cuda-12.8/lib64 -lcudart -lcufft -lnvToolsExt -lpthread -ldl -lm
```

Run:

```bash
/tmp/bench_spread 3 1000000 64 64 64 8 0
/tmp/bench_spread 3 100000 128 128 128 8 0
```

The app wraps `makeplan`/`setpts`/`execute` in NVTX ranges and calls
`cudaProfilerStart()` after the warm-up, so the report starts in steady state.

Nsight Systems (adapted from the suggested command: `-s cpu` is unsupported in this
container, and `--export=gputrace` is not a valid `nsys profile --export` value here):

```bash
nsys profile -w true -t nvtx,cuda,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true --output=dense_api \
  /tmp/bench_spread 3 1000000 64 64 64 8 0

nsys stats --report=nvtx_sum --report=nvtx_kern_sum \
  --report=cuda_gpu_kern_sum --report=cuda_api_sum dense_api.nsys-rep
```

Flamegraph:

```bash
nsys stats --report=nvtx_pushpop_trace --format csv dense_api.nsys-rep \
  | grep -E '^[0-9]' > dense_api_nvtx.csv
python3 tools/make_flame.py dense_api_nvtx.csv dense_api.sqlite \
  dense_api_flamegraph.svg "dense case" "64^3, M=1e6"
```

Nsight Compute:

```bash
ncu --kernel-name-base demangled --kernel-name regex:spread_output_driven \
  --launch-skip 1 --launch-count 1 --clock-control none \
  --section SpeedOfLight --section Occupancy --section SchedulerStats \
  --section WarpStateStats --section InstructionStats --section LaunchStats \
  /tmp/bench_spread 3 1000000 64 64 64 2 0
```

## Caveat

This is an ad-hoc baseline, not the complete ISSUE-002 baseline. It uses one GPU, one
kernel width, one precision, one point distribution, and spread-only mode. It establishes
where method 3 spends its time and what kind of bound it is under; it does not by itself
prove that an output-owned kernel will be faster.
