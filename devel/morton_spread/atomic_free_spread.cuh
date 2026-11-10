// Interface for the readable, atomic-free 3D type-1 spreading baseline.
//
// The implementation lives in atomic_free_spread.cu; the command line,
// timing, reference comparison, and correctness checks live in
// atomic_free_spread_bench.cu.

#ifndef ATOMIC_FREE_SPREAD_CUH
#define ATOMIC_FREE_SPREAD_CUH

#include <cuda_runtime.h>
#include <cufinufft.h>

namespace atomic_free_spread {

// Width-6 kernel (sigma=2, tolerance 1e-5).
constexpr int WIDTH = 6;
constexpr int HALO = WIDTH / 2;

// One CUDA block owns one output tile; one thread owns one output cell.
constexpr int TILE_X = 8;
constexpr int TILE_Y = 8;
constexpr int TILE_Z = 4;

// One thread per cell, so the block size follows the tile.
constexpr int BLOCK_THREADS = TILE_X * TILE_Y * TILE_Z;

// Candidate points whose kernel weights are staged into shared memory at once.
// Each point needs three staging slots, one per dimension, all live before the
// cell loop runs.
constexpr int STAGE_POINTS = 64;

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

}  // namespace atomic_free_spread

#endif  // ATOMIC_FREE_SPREAD_CUH
