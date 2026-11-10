// Readable, atomic-free 3D type-1 spreading prototype: implementation.
//
// This is a standalone experiment, not part of the library build. It uses a
// width-6 kernel (sigma=2, tolerance 1e-5), single precision, and periodic
// boundaries. Each GPU thread owns one output cell and writes it exactly once.
// There are no splits, partial grids, active-tile lists, or spreading atomics.
//
// For simplicity, binning and sorting are done on the CPU after a GPU kernel
// computes the bin IDs. The latter ensures bin assignment uses the exact same
// GPU fold_rescale as spreading, including at floating-point boundaries.
// This is a correctness/readability baseline, not an optimized implementation.
//
// This translation unit contains only the experiment's kernels and launch
// wrappers. The entry point, timing, reference comparison, and correctness
// checks are in atomic_free_spread_bench.cu. Build and run both with
// ./devel/morton_spread/run_atomic_free_spread.sh.

#include "atomic_free_spread.cuh"

#include <cufinufft/spreadinterp.hpp>

#include <cmath>

namespace atomic_free_spread {
namespace {

// A flattened Cartesian bin ID, with x varying fastest.
__global__ void assign_bins(const Point *points, int count, Grid grid, int *bin_ids) {
  using cufinufft::spreadinterp::fold_rescale;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int bx = min(int(fold_rescale(points[i].x, grid.nx) / grid.bin_size), grid.bins_x - 1);
  int by = min(int(fold_rescale(points[i].y, grid.ny) / grid.bin_size), grid.bins_y - 1);
  int bz = min(int(fold_rescale(points[i].z, grid.nz) / grid.bin_size), grid.bins_z - 1);
  bin_ids[i] = bx + grid.bins_x * (by + grid.bins_y * bz);
}

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

// Return the support coefficient for this cell, or -1 if the point does not
// cover it. Grid sizes are at least WIDTH, so at most one coefficient matches.
//
// fold_rescale returns [0, grid_size), so support_start = ceil(px - HALO) is in
// [-HALO, grid_size - HALO]. The kernel only calls this for cells in
// [0, grid_size), so cell - support_start is always in
// (-grid_size, 2 * grid_size). One conditional add and one conditional
// subtract therefore reduce it to [0, grid_size); no loop or division is
// needed. This one-period bound is specific to the per-cell reference point:
// place_support() compares against a lane-group start, which can be near the
// far edge of the grid and requires the multi-period correction.
__device__ __forceinline__ int coefficient_index(int cell, int support_start,
                                                 int grid_size) {
  int offset = cell - support_start;
  offset += (offset < 0) ? grid_size : 0;
  offset -= (offset >= grid_size) ? grid_size : 0;
  return offset < WIDTH ? offset : -1;
}

// Weight of this point for cell `cell`, or 0 when the point does not cover it.
// coefficient_index() returns the position within the point's WIDTH kernel
// values, or -1 when the point's support does not reach that cell.
__device__ __forceinline__ float cell_weight(int cell, int start, int grid_size,
                                             const float *ker) {
  const int weight_index = coefficient_index(cell, start, grid_size);
  const bool covered = cell < grid_size && weight_index >= 0;
  return covered ? ker[weight_index] : 0.0f;
}

// Fill dest[0..COUNT) with the weights of the cells tile_begin + local.
template<int COUNT>
__device__ __forceinline__ void stage_tile_weights(float *dest, int tile_begin,
                                                   int start, int grid_size,
                                                   const float *ker) {
  for (int local = 0; local < COUNT; ++local)
    dest[local] = cell_weight(tile_begin + local, start, grid_size, ker);
}

// Same, pre-scaled by the point strength, for the dimension the cell loop reads
// as one float2 instead of loading the strength separately.
template<int COUNT>
__device__ __forceinline__ void stage_tile_weights(float2 *dest, int tile_begin,
                                                   int start, int grid_size,
                                                   const float *ker, float2 strength) {
  for (int local = 0; local < COUNT; ++local) {
    const float weight = cell_weight(tile_begin + local, start, grid_size, ker);
    dest[local] = make_float2(weight * strength.x, weight * strength.y);
  }
}

// One block owns one tile; one thread owns one cell. The block stages the kernel
// weights of a batch of candidate points into shared memory, indexed by cell
// position inside the tile, and each cell then reads the three values it needs:
// the x and y weights and the z weight already scaled by the point strength.
//
// The staged vectors are tile-local and zero-filled, so the cell loop has no
// support test, no branch, and no coefficient selection. A zero weight adds an
// exact zero, so uncovered cells keep their accumulator; a non-finite point
// strength would now reach cells the point does not cover.
__global__ void spread(const Point *points, const int *bin_offsets, Grid grid,
                       cuFloatComplex *output) {
  using cufinufft::spreadinterp::eval_kernel_vec_horner;
  using cufinufft::spreadinterp::fold_rescale;

  int tile_x = blockIdx.x * TILE_X;
  int tile_y = blockIdx.y * TILE_Y;
  int tile_z = blockIdx.z * TILE_Z;
  int x = tile_x + threadIdx.x;
  int y = tile_y + threadIdx.y;
  int z = tile_z + threadIdx.z;

  // The candidate-bin ranges depend only on blockIdx and the grid, not on the
  // thread, so compute them once per block instead of once per thread.
  __shared__ NearBins x_bins, y_bins, z_bins;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const int end_x = min(tile_x + TILE_X, grid.nx);
    const int end_y = min(tile_y + TILE_Y, grid.ny);
    const int end_z = min(tile_z + TILE_Z, grid.nz);
    x_bins = near_bins(tile_x, end_x, grid.bin_size, grid.nx, grid.bins_x);
    y_bins = near_bins(tile_y, end_y, grid.bin_size, grid.ny, grid.bins_y);
    z_bins = near_bins(tile_z, end_z, grid.bin_size, grid.nz, grid.bins_z);
  }

  // Staged weights for one batch: kx[j][local] and ky[j][local] are the weights
  // of staged point j for cell tile_x + local (likewise y), and kz[j][local] is
  // the z weight already multiplied by the point strength. Rows are padded to an
  // odd number of floats so that the staging lanes, which write one `local` of
  // many consecutive points at once, hit distinct shared-memory banks.
  __shared__ float kx[STAGE_POINTS][TILE_X + 1];
  __shared__ float ky[STAGE_POINTS][TILE_Y + 1];
  __shared__ float2 kz[STAGE_POINTS][TILE_Z + 1];
  __syncthreads();

  const int linear_tid = threadIdx.x + TILE_X * (threadIdx.y + TILE_Y * threadIdx.z);

  // One staging lane per (point, dimension): dim d owns the lanes
  // [d * STAGE_POINTS, (d + 1) * STAGE_POINTS), so every warp stages a single
  // dimension and the dimension branches below do not diverge. Lanes past
  // 3 * STAGE_POINTS only take part in the barriers.
  const int stage_dim = linear_tid / STAGE_POINTS;
  const int stage_point = linear_tid % STAGE_POINTS;
  const bool is_stager = stage_dim < 3;

  // Threads on the ragged edge of a partial tile own no cell, but they must keep
  // reaching the barriers, so the early return becomes a store predicate.
  const bool owns_cell = x < grid.nx && y < grid.ny && z < grid.nz;
  float real = 0.0f, imag = 0.0f;

  for (int z_bin_index = 0; z_bin_index < z_bins.count; ++z_bin_index) {
    const int bz = z_bins[z_bin_index];
    for (int y_bin_index = 0; y_bin_index < y_bins.count; ++y_bin_index) {
      const int by = y_bins[y_bin_index];
      for (int x_bin_index = 0; x_bin_index < x_bins.count; ++x_bin_index) {
        const int bx = x_bins[x_bin_index];
        const int bin = bx + grid.bins_x * (by + grid.bins_y * bz);
        const int bin_begin = bin_offsets[bin];
        const int bin_end = bin_offsets[bin + 1];
        // This loop carries the block's barriers, so every thread must run it.
        // Only the body is conditional: `staged` shrinks on the last batch of a
        // bin.
        for (int base = bin_begin; base < bin_end; base += STAGE_POINTS) {
          const int staged = min(STAGE_POINTS, bin_end - base);
          if (is_stager && stage_point < staged) {
            const Point p = points[base + stage_point];
            const float raw = stage_dim == 0 ? p.x : (stage_dim == 1 ? p.y : p.z);
            const int size =
                stage_dim == 0 ? grid.nx : (stage_dim == 1 ? grid.ny : grid.nz);
            const float rescaled = fold_rescale(raw, size);
            const int start = int(ceilf(rescaled - HALO));
            float ker[WIDTH];
            eval_kernel_vec_horner<float, WIDTH>(ker, float(start) - rescaled, 2.0);
            if (stage_dim == 0) {
              stage_tile_weights<TILE_X>(kx[stage_point], tile_x, start, grid.nx, ker);
            } else if (stage_dim == 1) {
              stage_tile_weights<TILE_Y>(ky[stage_point], tile_y, start, grid.ny, ker);
            } else {
              stage_tile_weights<TILE_Z>(kz[stage_point], tile_z, start, grid.nz, ker,
                                         make_float2(p.strength.x, p.strength.y));
            }
          }
          __syncthreads();
          for (int j = 0; j < staged; ++j) {
            const float weight = kx[j][threadIdx.x] * ky[j][threadIdx.y];
            const float2 z = kz[j][threadIdx.z];
            real += z.x * weight;
            imag += z.y * weight;
          }
          __syncthreads();
        }
      }
    }
  }
  if (owns_cell) {
    long long index = x + (long long)grid.nx * (y + (long long)grid.ny * z);
    output[index] = make_cuFloatComplex(real, imag);
  }
}

}  // namespace

void launch_assign_bins(const Point *points, int count, Grid grid, int *bin_ids) {
  constexpr int THREADS = 256;
  assign_bins<<<(count + THREADS - 1) / THREADS, THREADS>>>(points, count, grid, bin_ids);
}

void launch_spread(const Point *points, const int *bin_offsets, Grid grid,
                   cuFloatComplex *output) {
  dim3 threads(TILE_X, TILE_Y, TILE_Z);
  dim3 blocks(1 + (grid.nx - 1) / TILE_X, 1 + (grid.ny - 1) / TILE_Y,
              1 + (grid.nz - 1) / TILE_Z);
  spread<<<blocks, threads>>>(points, bin_offsets, grid, output);
}

}  // namespace atomic_free_spread
