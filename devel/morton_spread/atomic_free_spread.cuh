// Interface for the warp-owned, atomic-free 3D type-1 spreading prototype and
// the F0 point-owned spread prototype.
//
// The warp-owned implementation lives in atomic_free_spread.cu, the F0 prototype
// in f0_point_owned_spread.cu, and the command line, timing, reference
// comparison, and correctness checks in atomic_free_spread_bench.cu.

#ifndef ATOMIC_FREE_SPREAD_CUH
#define ATOMIC_FREE_SPREAD_CUH

#include <cuda_runtime.h>
#include <cufinufft.h>

#include <vector>

namespace atomic_free_spread {

// Width-6 kernel (sigma=2, tolerance 1e-5).
constexpr int WIDTH = 6;

// Kernel support beyond a warp region: ceil(WIDTH / 2) cells per side. This is
// the support overhang, not the glossary's padding, which describes a local
// subgrid.
constexpr int SUPPORT_OVERHANG = (WIDTH + 1) / 2;

// One warp owns one 8x8x8 warp region, and one lane owns a 2x2x4 lane cell group.
// A CUDA threadblock packs WARP_REGIONS_PER_THREADBLOCK independent work items;
// each work item is one (warp region, split) pair, so the threadblock is only a
// launch container, not a tile.
constexpr int WARP_REGION_X = 8;
constexpr int WARP_REGION_Y = 8;
constexpr int WARP_REGION_Z = 8;

constexpr int LANE_CELLS_X = 2;
constexpr int LANE_CELLS_Y = 2;
constexpr int LANE_CELLS_Z = 4;

constexpr int LANES_X = WARP_REGION_X / LANE_CELLS_X;  // 4
constexpr int LANES_Y = WARP_REGION_Y / LANE_CELLS_Y;  // 4
constexpr int LANES_Z = WARP_REGION_Z / LANE_CELLS_Z;  // 2
constexpr int LANES_PER_WARP = 32;

constexpr int WARP_REGION_CELLS = WARP_REGION_X * WARP_REGION_Y * WARP_REGION_Z;

// Launch-policy knob: warps per threadblock. The spread launch packs work items
// into a 1D grid, so this only sets the threadblock size (two warps = 64
// threads) and the number of concurrent work items per threadblock.
constexpr int WARP_REGIONS_PER_THREADBLOCK = 2;  // 1, 2, or 4
constexpr int WARPS_PER_THREADBLOCK = WARP_REGIONS_PER_THREADBLOCK;
constexpr int THREADS_PER_THREADBLOCK = WARPS_PER_THREADBLOCK * 32;

static_assert(WARP_REGION_X % LANE_CELLS_X == 0);
static_assert(WARP_REGION_Y % LANE_CELLS_Y == 0);
static_assert(WARP_REGION_Z % LANE_CELLS_Z == 0);
static_assert(LANES_X * LANES_Y * LANES_Z == LANES_PER_WARP);
static_assert(WARP_REGION_CELLS ==
              LANES_PER_WARP * LANE_CELLS_X * LANE_CELLS_Y * LANE_CELLS_Z);

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
__host__ __device__ inline long long bin_count(const Grid &grid) {
  return (long long)grid.bins_x * grid.bins_y * grid.bins_z;
}

__host__ __device__ inline long long cell_count(const Grid &grid) {
  return (long long)grid.nx * grid.ny * grid.nz;
}

// Warp-region geometry: ceil(n / WARP_REGION_*) regions per dimension, packed
// x-fastest into flat region IDs. Edge regions are clipped by the kernel. The
// helpers are host/device because the launch and the kernels both decode IDs.
__host__ __device__ __forceinline__ int warp_regions_x(const Grid &grid) {
  return (grid.nx + WARP_REGION_X - 1) / WARP_REGION_X;
}

__host__ __device__ __forceinline__ int warp_regions_y(const Grid &grid) {
  return (grid.ny + WARP_REGION_Y - 1) / WARP_REGION_Y;
}

__host__ __device__ __forceinline__ int warp_regions_z(const Grid &grid) {
  return (grid.nz + WARP_REGION_Z - 1) / WARP_REGION_Z;
}

__host__ __device__ __forceinline__ int warp_region_count(const Grid &grid) {
  return warp_regions_x(grid) * warp_regions_y(grid) * warp_regions_z(grid);
}

namespace detail {

// Truncating division plus a mask correction: floor(a/b) for positive b.
__host__ __device__ __forceinline__ int floor_div(int a, int b) {
  const int quotient = a / b;
  const int remainder = a - quotient * b;
  return quotient - ((remainder >> 31) & 1);
}

// The candidate bins for one dimension, in periodic (unwrapped) order. A bin is
// a candidate when its box can contain a source whose kernel support reaches
// the region. All candidate bins are returned exactly once; when the expanded
// region covers the whole grid the range is capped to [0, bins) so a wrapped
// bin is not visited twice.
struct CandidateBins {
  int first;
  int count;
  int bins;

  __host__ __device__ __forceinline__ int operator[](int i) const {
    int bin = first + i;
    if (bin < 0) bin += bins;
    else if (bin >= bins) bin -= bins;
    return bin;
  }
};

// Return the bins intersecting [region_begin - SUPPORT_OVERHANG,
// region_end + SUPPORT_OVERHANG). The last bin of each period may be shorter
// than bin_size, so map an extended coordinate to its bin copy first, then let
// operator[] fold the copy index back into [0, bins). This replaces the per-bin
// bin_near_tile test.
__host__ __device__ __forceinline__ int extended_bin_index(int coordinate,
                                                           int bin_size,
                                                           int grid_size,
                                                           int bins) {
  const int period = floor_div(coordinate, grid_size);
  const int local_coordinate = coordinate - period * grid_size;  // [0, grid_size)
  const int last_bin = bins - 1;
  const int local_bin = local_coordinate / bin_size;
  return period * bins + (local_bin < last_bin ? local_bin : last_bin);
}

__host__ __device__ __forceinline__ CandidateBins candidate_bins(
    int region_begin, int region_end, int bin_size, int grid_size, int bins) {
  const int query_begin = region_begin - SUPPORT_OVERHANG;
  const int query_end = region_end + SUPPORT_OVERHANG;  // exclusive
  const int first = extended_bin_index(query_begin, bin_size, grid_size, bins);
  const int last = extended_bin_index(query_end - 1, bin_size, grid_size, bins);
  int count = last - first + 1;
  if (count >= bins) return CandidateBins{0, bins, bins};
  return CandidateBins{first, count, bins};
}

}  // namespace detail

// The launch policy selected by the host active-region plan.
enum class LaunchPath { Empty, FullGrid, ActiveSingleSplit, ActiveSplit };

// Host-side active-region plan. regions is the deterministic, compacted list of
// warp-region IDs whose candidate bins contain at least one source. split_count
// is the split heuristic result, and path selects the launch/reduction shape.
struct ActiveRegionPlan {
  std::vector<int> regions;
  int split_count = 1;
  LaunchPath path = LaunchPath::Empty;
};

// Plan the active warp regions from the host bin-offset table. This is computed
// inside launch_spread so the benchmark's spread interval includes it.
ActiveRegionPlan plan_active_regions(const int *host_bin_offsets, Grid grid);

// Assign each point to a flattened Cartesian bin (x varies fastest). The
// kernel folds coordinates with cuFINUFFT's fold_rescale, so bin assignment
// uses the exact same floating-point fold as the spread kernel. The caller
// must check cudaGetLastError() after this launch.
void launch_assign_bins(const Point *points, int count, Grid grid, int *bin_ids);

// Launch the atomic-free spread. Each output cell is written exactly once by
// one lane, either directly at the end of that lane's scan or by the split
// reduction that follows it. The launch plans the active warp regions on the
// host from host_bin_offsets (those with a non-empty candidate bin), uploads the
// compacted list when needed, and chooses a split count for hot inputs; inactive
// regions are written as zeros. device_bin_offsets drives the device scan. The
// caller must check cudaGetLastError() after this launch.
void launch_spread(const Point *points, const int *device_bin_offsets,
                   const int *host_bin_offsets, Grid grid,
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
