// Interface for the readable, atomic-free 3D type-1 spreading baseline and the
// F0 point-owned spread prototype.
//
// The baseline implementation lives in atomic_free_spread.cu, the F0 prototype
// in f0_point_owned_spread.cu, and the command line, timing, reference
// comparison, and correctness checks in atomic_free_spread_bench.cu.

#ifndef ATOMIC_FREE_SPREAD_CUH
#define ATOMIC_FREE_SPREAD_CUH

#include <cuda_runtime.h>
#include <cufinufft.h>

namespace atomic_free_spread {

// Width-6 kernel (sigma=2, tolerance 1e-5).
constexpr int WIDTH = 6;
constexpr int HALO = WIDTH / 2;

// One CUDA block owns one output tile; one thread owns a small box of cells.
constexpr int TILE_X = 8;
constexpr int TILE_Y = 8;
constexpr int TILE_Z = 8;

// Four adjacent x cells share one strength-scaled z weight. Each half-warp
// spans one z slice.
constexpr int GROUP_X = 4;
constexpr int GROUP_Y = 1;
constexpr int GROUP_Z = 1;

constexpr int BLOCK_X = TILE_X / GROUP_X;
constexpr int BLOCK_Y = TILE_Y / GROUP_Y;
constexpr int BLOCK_Z = TILE_Z / GROUP_Z;
constexpr int BLOCK_THREADS = BLOCK_X * BLOCK_Y * BLOCK_Z;

// Candidate points whose kernel weights are staged into shared memory at once.
// Each point needs three staging slots, one per dimension, all live before the
// cell loop runs.
constexpr int STAGE_POINTS = 32;

static_assert(TILE_X % GROUP_X == 0 && TILE_Y % GROUP_Y == 0 &&
                  TILE_Z % GROUP_Z == 0,
              "a tile must be divisible into thread groups");
static_assert(3 * STAGE_POINTS <= BLOCK_THREADS, "staging must fit in one round");
static_assert(STAGE_POINTS % 32 == 0, "a warp must stage a single dimension");

struct Point {
  float x, y, z;
  cuFloatComplex strength;
};

struct Grid {
  int nx, ny, nz;
  int bin_size;
  int bins_x, bins_y, bins_z;
};

inline Grid make_grid(int nx, int ny, int nz, int bin_size) {
  return Grid{nx, ny, nz, bin_size, 1 + (nx - 1) / bin_size,
              1 + (ny - 1) / bin_size, 1 + (nz - 1) / bin_size};
}

// The last bin may be shorter than bin_size.
inline long long bin_count(const Grid &grid) {
  return (long long)grid.bins_x * grid.bins_y * grid.bins_z;
}

inline long long cell_count(const Grid &grid) {
  return (long long)grid.nx * grid.ny * grid.nz;
}

namespace detail {

// Truncating division plus a mask correction: floor(a/b) for positive b.
__device__ __forceinline__ int floor_div(int a, int b) {
  const int quotient = a / b;
  const int remainder = a - quotient * b;
  return quotient - ((remainder >> 31) & 1);
}

// The candidate bins for one dimension, in periodic (unwrapped) order. All
// bins are returned exactly once; when the expanded tile covers the whole grid
// the range is capped to [0, bins) so a wrapped bin is not visited twice.
struct NearBins {
  int first;
  int count;
  int bins;

  __device__ __forceinline__ int operator[](int i) const {
    int bin = first + i;
    if (bin < 0) bin += bins;
    else if (bin >= bins) bin -= bins;
    return bin;
  }
};

// Return the bins intersecting [tile_begin - HALO, tile_end + HALO). The last
// bin of each period may be shorter than bin_size, so map an extended
// coordinate to its bin copy first, then let operator[] fold the copy index
// back into [0, bins). This replaces the per-bin bin_near_tile test.
__device__ __forceinline__ int extended_bin_index(int coordinate, int bin_size,
                                                  int grid_size, int bins) {
  const int period = floor_div(coordinate, grid_size);
  const int local_coordinate = coordinate - period * grid_size;  // [0, grid_size)
  const int local_bin = min(local_coordinate / bin_size, bins - 1);
  return period * bins + local_bin;
}

__device__ __forceinline__ NearBins near_bins(int tile_begin, int tile_end,
                                              int bin_size, int grid_size, int bins) {
  const int query_begin = tile_begin - HALO;
  const int query_end = tile_end + HALO;  // exclusive
  const int first = extended_bin_index(query_begin, bin_size, grid_size, bins);
  const int last = extended_bin_index(query_end - 1, bin_size, grid_size, bins);
  int count = last - first + 1;
  if (count >= bins) return NearBins{0, bins, bins};
  return NearBins{first, count, bins};
}

}  // namespace detail

// Assign each point to a flattened Cartesian bin (x varies fastest). The
// kernel folds coordinates with cuFINUFFT's fold_rescale, so bin assignment
// uses the exact same floating-point fold as the spread kernel. The caller
// must check cudaGetLastError() after this launch.
void launch_assign_bins(const Point *points, int count, Grid grid, int *bin_ids);

// Launch the atomic-free spread. Each output cell is written exactly once by
// one thread, at the end of that thread's scan. The caller must check
// cudaGetLastError() after this launch.
void launch_spread(const Point *points, const int *bin_offsets, Grid grid,
                   cuFloatComplex *output);

// Launch the F0 point-owned spread. Each block owns one exact output tile and
// scatters per-point support into shared tiles with shared-memory atomics; the
// summation order within a tile is nondeterministic. Each in-grid tile cell is
// written exactly once after a block barrier. The caller must check
// cudaGetLastError() after this launch.
void launch_spread_f0(const Point *points, const int *bin_offsets, Grid grid,
                      cuFloatComplex *output);

}  // namespace atomic_free_spread

#endif  // ATOMIC_FREE_SPREAD_CUH
