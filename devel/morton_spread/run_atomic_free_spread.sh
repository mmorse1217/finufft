#!/usr/bin/env bash
# Build and validate the readable atomic-free spreading baseline or the F0
# point-owned prototype.
#
#   ./run_atomic_free_spread.sh          # build + correctness matrix
#   ./run_atomic_free_spread.sh all      # correctness matrix + bin-size/seed sweep
#   KERNEL=1 ./run_atomic_free_spread.sh # same, using the F0 point-owned kernel
#   BIN_SORT=0 ./run_atomic_free_spread.sh # use the CPU fallback
#
# The program compares its fine-grid output with pinned cuFINUFFT method 3
# (spread-only, Horner, ns=6, sigma=2), checks bin-table/coverage/finiteness/total-sum
# invariants, reports per-phase timings, and exits non-zero when relL2 >= 1e-4 or
# any invariant fails. Set NRUN (default 5) for the number of timed spread runs.
# BIN_SORT selects preparation: 0 = CPU fallback, 1 = stable GPU radix (default).
# BIN_SORT=1 also adds the short GPU boundary matrix below. Requires the CUDA build in
# build/gpu (see docs/issues/research/ISSUE-002-baseline.md).
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"

export PATH=$HOME/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
CUDA=$HOME/cuda-12.8
CCCL=/tmp/cccl-3.4.2
BUILD_DIR=${BUILD_DIR:-build/gpu}
BIN=${BIN:-$BUILD_DIR/atomic_free_spread_bench}
NRUN=${NRUN:-5}
KERNEL=${KERNEL:-0}
BIN_SORT=${BIN_SORT:-1}

if [[ "$KERNEL" != 0 && "$KERNEL" != 1 ]]; then
  echo "KERNEL must be 0 (atomic-free baseline) or 1 (F0 point-owned), got '$KERNEL'" >&2
  exit 1
fi

if [[ "$BIN_SORT" != 0 && "$BIN_SORT" != 1 ]]; then
  echo "BIN_SORT must be 0 (CPU fallback) or 1 (stable GPU radix), got '$BIN_SORT'" >&2
  exit 1
fi

if [[ ! -f "$BUILD_DIR/libcufinufft.a" ]]; then
  echo "missing $BUILD_DIR/libcufinufft.a; configure and build the gpu preset first" >&2
  exit 1
fi

nvcc -O3 -std=c++17 -arch=sm_75 \
  -Iinclude \
  -I"$CCCL/thrust" -I"$CCCL/libcudacxx/include" -I"$CCCL/cub" \
  -isystem="$CUDA/include" -isystem="$BUILD_DIR/_deps/poet-src/include" \
  devel/morton_spread/atomic_free_spread.cu \
  devel/morton_spread/f0_point_owned_spread.cu \
  devel/morton_spread/gpu_bin_sort.cu \
  devel/morton_spread/atomic_free_spread_bench.cu -o "$BIN" \
  "$BUILD_DIR/libcufinufft.a" "$BUILD_DIR/src/common/libfinufft_common.a" \
  -lcudart -lcufft -lpthread -ldl -lm

run_with_sort() {
  local bin_sort=$1
  shift
  echo "--- M=$1 grid=$2x$3x$4 cluster=$5 bin_size=$6 pattern=$7 seed=$8 " \
       "nrun=$NRUN kernel=$KERNEL bin_sort=$bin_sort"
  "$BIN" "$@" "$NRUN" "$KERNEL" "$bin_sort"
}

run() {
  run_with_sort "$BIN_SORT" "$@"
}

# args: M N1 N2 N3 cluster bin_size pattern seed [nrun via NRUN env; kernel via
# KERNEL env, 0 = atomic-free baseline, 1 = F0 point-owned; bin sort via
# BIN_SORT env, 0 = CPU fallback, 1 = stable radix GPU bin sort]
# pattern 0: uniform / eight tight clusters
# pattern 1: exact seam and out-of-domain coordinates
# pattern 2: exact grid nodes and near-node offsets
# pattern 3: one dense cluster (long same-cell accumulation)

# Mandatory workload shapes (sparse because this baseline is not optimized).
run 1000000 64 64 64 0 4 0 42
run 1000000 64 64 64 1 4 0 42
run 100000 128 128 128 0 4 0 42

# Boundary coverage: the seam case is the regression test for the multi-period
# support wrap. The atomic baseline reduces support offsets modulo the grid and
# is expected to pass all of these.
run 100000 64 64 64 0 4 1 42
run 100000 64 64 64 0 4 2 42
run 100000 64 64 64 0 4 3 42
run 4096 6 6 6 0 6 1 42
run 4096 6 9 13 0 2 2 42
run 20000 33 35 31 0 4 2 42

if [[ "$BIN_SORT" != 0 ]]; then
  # Short GPU-bin-sort boundary matrix. The cases above already run with the
  # selected sort; these add one bin (end_bit = 0), bin_size=1 (a larger
  # end_bit), and non-power-of-two bin counts.
  run 20000 64 64 64 0 64 0 42
  run 20000 32 32 32 0 1 2 42
  run 100000 33 35 31 0 4 0 42
  run 100000 33 35 31 0 4 1 42
fi

if [[ "${1:-}" == "all" ]]; then
  for bin_size in 1 2 8 16 32; do
    run 20000 32 32 32 0 "$bin_size" 2 7
    run 20000 32 32 32 0 "$bin_size" 1 7
  done
  for seed in 1 2 3; do
    run 50000 64 64 64 0 4 1 "$seed"
    run 50000 64 64 64 1 4 0 "$seed"
  done
fi
