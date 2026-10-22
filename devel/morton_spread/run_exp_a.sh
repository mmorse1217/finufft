#!/usr/bin/env bash
# Build and run the Experiment A prototype (output-cell ownership + Cartesian bins).
#
#   ./run_exp_a.sh            # build, then run the mandatory N=1e6 uniform case
#   ./run_exp_a.sh all        # build, then run the mandatory matrix + qb sweep
#
# Requires the CUDA build in build/gpu (see docs/issues/research/ISSUE-002-baseline.md).
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"

export PATH=$HOME/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
CUDA=$HOME/cuda-12.8
CCCL=/tmp/cccl-3.4.2
BIN=build/gpu/exp_a_cartesian

if [[ ! -f build/gpu/libcufinufft.a ]]; then
  echo "missing build/gpu/libcufinufft.a; configure and build the gpu preset first" >&2
  exit 1
fi

nvcc -O3 -std=c++17 -arch=sm_75 \
  -Iinclude \
  -I"$CCCL/thrust" -I"$CCCL/libcudacxx/include" -I"$CCCL/cub" \
  -isystem="$CUDA/include" -isystem=build/gpu/_deps/poet-src/include \
  devel/morton_spread/exp_a_cartesian.cu -o "$BIN" \
  build/gpu/libcufinufft.a build/gpu/src/common/libfinufft_common.a \
  -lcudart -lcufft -lnvToolsExt -lpthread -ldl -lm

# args: M N1 N2 N3 cluster qb nrun.
# Each case runs twice: the GPU clock ramps during the first process, and method 3 is
# measured first, so a single cold run overstates method 3 by ~1.5x. The second run is
# the one to read.
run() {
  echo "--- M=$1 nf=$2^3 cluster=$4 qb=$5"
  "$BIN" "$@" >/dev/null
  "$BIN" "$@"
}

if [[ "${1:-}" == "all" ]]; then
  run 1000000 64 64 64 0 4 5    # mandatory: uniform 1e6
  run 1000000 64 64 64 1 4 5    # mandatory: clustered 1e6
  run 100000 128 128 128 0 4 5  # mandatory: sparse 1e5
  run 1000000 64 64 64 0 2 5    # qb sweep (diagnostic)
  run 1000000 64 64 64 1 2 5
  run 1000000 64 64 64 0 8 5
else
  run 1000000 64 64 64 0 4 5
fi
