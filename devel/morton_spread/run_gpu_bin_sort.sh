#!/usr/bin/env bash
# Build and run the focused GPU bin sort correctness/timing test.
#
#   ./run_gpu_bin_sort.sh
#   BUILD_DIR=/path/to/build/gpu ./run_gpu_bin_sort.sh
#
# Compares GPU offsets and sorted records exactly with a stable CPU sort,
# including empty inputs and repeated workspace reuse. Exits non-zero on failure.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"

export PATH=$HOME/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
CUDA=$HOME/cuda-12.8
CCCL=/tmp/cccl-3.4.2
BUILD_DIR=${BUILD_DIR:-build/gpu}
BIN=${BIN:-$BUILD_DIR/gpu_bin_sort_test}

if [[ ! -f "$BUILD_DIR/libcufinufft.a" ]]; then
  echo "missing $BUILD_DIR/libcufinufft.a; configure and build the gpu preset first" >&2
  exit 1
fi

nvcc -O3 -std=c++17 -arch=sm_75 \
  -Iinclude \
  -I"$CCCL/thrust" -I"$CCCL/libcudacxx/include" -I"$CCCL/cub" \
  -isystem="$CUDA/include" -isystem="$BUILD_DIR/_deps/poet-src/include" \
  devel/morton_spread/atomic_free_spread.cu \
  devel/morton_spread/gpu_bin_sort.cu \
  devel/morton_spread/gpu_bin_sort_test.cu -o "$BIN" \
  "$BUILD_DIR/libcufinufft.a" "$BUILD_DIR/src/common/libfinufft_common.a" \
  -lcudart -lcufft -lpthread -ldl -lm

"$BIN" "$@"
