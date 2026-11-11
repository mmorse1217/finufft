// Readable, atomic-free 3D type-1 spreading prototype: implementation.
//
// This is a standalone experiment, not part of the library build. It uses a
// width-6 kernel (sigma=2, tolerance 1e-5), single precision, and periodic
// boundaries. Each GPU thread owns a small box of output cells and writes each
// one exactly once. There are no splits, partial grids, active-tile lists, or
// spreading atomics.
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

// Fill dest[0..COUNT) with the weights for cells tile_begin + local. The row is
// zero outside the source support.
template<int COUNT>
__device__ __forceinline__ void stage_tile_weights(float *dest, int tile_begin,
                                                   int start, int fine_size,
                                                   const float *ker) {
  #pragma unroll
  for (int local = 0; local < COUNT; ++local)
    dest[local] = 0.0f;

  // start is in [-HALO, fine_size - HALO]; one add normalizes it.
  start += (start < 0) ? fine_size : 0;

  #pragma unroll
  for (int k = 0; k < WIDTH; ++k) {
    int cell = start + k;
    cell -= (cell >= fine_size) ? fine_size : 0;
    const int local = cell - tile_begin;
    if (local >= 0 && local < COUNT)
      dest[local] = ker[k];
  }
}

// Same, pre-scaled by the point strength, for the dimension the cell loop reads
// as one float2 instead of loading the strength separately.
template<int COUNT>
__device__ __forceinline__ void stage_tile_weights(float2 *dest, int tile_begin,
                                                   int start, int fine_size,
                                                   const float *ker, float2 strength) {
  #pragma unroll
  for (int local = 0; local < COUNT; ++local)
    dest[local] = make_float2(0.0f, 0.0f);

  start += (start < 0) ? fine_size : 0;

  #pragma unroll
  for (int k = 0; k < WIDTH; ++k) {
    int cell = start + k;
    cell -= (cell >= fine_size) ? fine_size : 0;
    const int local = cell - tile_begin;
    if (local >= 0 && local < COUNT)
      dest[local] = make_float2(ker[k] * strength.x, ker[k] * strength.y);
  }
}

// Add the staged batch to this thread's cell group. Each x-y product serves the
// group's z cells and each z value serves its x cells, so a point costs
// GROUP_X + GROUP_Y + GROUP_Z shared loads for GROUP_X * GROUP_Y * GROUP_Z cells.
__device__ __forceinline__ void accumulate_batch(
    const float (*kx)[TILE_X + 1], const float (*ky)[TILE_Y + 1],
    const float2 (*kz)[TILE_Z + 1], int staged, int x0, int y0, int z0,
    float2 (&acc)[GROUP_X][GROUP_Y][GROUP_Z]) {
  for (int j = 0; j < staged; ++j) {
    float weight[GROUP_X][GROUP_Y];
    #pragma unroll
    for (int gx = 0; gx < GROUP_X; ++gx)
      #pragma unroll
      for (int gy = 0; gy < GROUP_Y; ++gy)
        weight[gx][gy] = kx[j][x0 + gx] * ky[j][y0 + gy];

    float2 z[GROUP_Z];
    #pragma unroll
    for (int gz = 0; gz < GROUP_Z; ++gz) z[gz] = kz[j][z0 + gz];

    #pragma unroll
    for (int gx = 0; gx < GROUP_X; ++gx)
      #pragma unroll
      for (int gy = 0; gy < GROUP_Y; ++gy)
        #pragma unroll
        for (int gz = 0; gz < GROUP_Z; ++gz) {
          acc[gx][gy][gz].x += z[gz].x * weight[gx][gy];
          acc[gx][gy][gz].y += z[gz].y * weight[gx][gy];
        }
  }
}

// Write this thread's cell group, skipping cells past the end of a partial tile.
__device__ __forceinline__ void store_cell_group(
    cuFloatComplex *output, Grid grid, int tile_x, int tile_y, int tile_z, int x0,
    int y0, int z0, const float2 (&acc)[GROUP_X][GROUP_Y][GROUP_Z]) {
  #pragma unroll
  for (int gx = 0; gx < GROUP_X; ++gx)
    #pragma unroll
    for (int gy = 0; gy < GROUP_Y; ++gy)
      #pragma unroll
      for (int gz = 0; gz < GROUP_Z; ++gz) {
        const int x = tile_x + x0 + gx;
        const int y = tile_y + y0 + gy;
        const int z = tile_z + z0 + gz;
        if (x < grid.nx && y < grid.ny && z < grid.nz) {
          const long long index =
              x + (long long)grid.nx * (y + (long long)grid.ny * z);
          output[index] = make_cuFloatComplex(acc[gx][gy][gz].x, acc[gx][gy][gz].y);
        }
      }
}

// One block owns one tile; one thread owns a GROUP_X x GROUP_Y x GROUP_Z box of
// cells. The block stages tile-local, zero-filled weight vectors into shared
// memory, so summing a staged point needs no support test, branch, or
// coefficient selection.
__global__ void spread(const Point *points, const int *bin_offsets, Grid grid,
                       cuFloatComplex *output) {
  using cufinufft::spreadinterp::eval_kernel_vec_horner;
  using cufinufft::spreadinterp::fold_rescale;

  int tile_x = blockIdx.x * TILE_X;
  int tile_y = blockIdx.y * TILE_Y;
  int tile_z = blockIdx.z * TILE_Z;
  int x0 = threadIdx.x * GROUP_X;
  int y0 = threadIdx.y * GROUP_Y;
  int z0 = threadIdx.z * GROUP_Z;

  __shared__ detail::NearBins x_bins, y_bins, z_bins;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const int end_x = min(tile_x + TILE_X, grid.nx);
    const int end_y = min(tile_y + TILE_Y, grid.ny);
    const int end_z = min(tile_z + TILE_Z, grid.nz);
    x_bins = detail::near_bins(tile_x, end_x, grid.bin_size, grid.nx, grid.bins_x);
    y_bins = detail::near_bins(tile_y, end_y, grid.bin_size, grid.ny, grid.bins_y);
    z_bins = detail::near_bins(tile_z, end_z, grid.bin_size, grid.nz, grid.bins_z);
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

  const int linear_tid =
      threadIdx.x + BLOCK_X * (threadIdx.y + BLOCK_Y * threadIdx.z);

  // One staging lane per (point, dimension): dim d owns the lanes
  // [d * STAGE_POINTS, (d + 1) * STAGE_POINTS), so every warp stages a single
  // dimension and the dimension branches below do not diverge. Lanes past
  // 3 * STAGE_POINTS only take part in the barriers.
  const int stage_dim = linear_tid / STAGE_POINTS;
  const int stage_point = linear_tid % STAGE_POINTS;
  const bool is_stager = stage_dim < 3;

  float2 acc[GROUP_X][GROUP_Y][GROUP_Z] = {};

  for (int z_bin_index = 0; z_bin_index < z_bins.count; ++z_bin_index) {
    const int bz = z_bins[z_bin_index];
    // Skip accumulation when a bin's periodic support misses this warp's z slice.
    int dz = tile_z + z0 - bz * grid.bin_size;
    if (dz + GROUP_Z <= -HALO) dz += grid.nz;
    if (dz + GROUP_Z > grid.nz - HALO) dz -= grid.nz;
    const bool covered_z = dz < min(grid.bin_size, grid.nz - bz * grid.bin_size) + HALO;
    for (int y_bin_index = 0; y_bin_index < y_bins.count; ++y_bin_index) {
      const int by = y_bins[y_bin_index];
      for (int x_bin_index = 0; x_bin_index < x_bins.count;) {
        const int bx = x_bins[x_bin_index];
        // Adjacent x bins form one contiguous point range; stop at the seam.
        const int run = min(x_bins.count - x_bin_index, grid.bins_x - bx);
        x_bin_index += run;
        const int bin = bx + grid.bins_x * (by + grid.bins_y * bz);
        const int bin_begin = bin_offsets[bin];
        const int bin_end = bin_offsets[bin + run];
        // This loop carries the block's barriers, so every thread must run it;
        // only staging and accumulation are conditional.
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
          if (covered_z) accumulate_batch(kx, ky, kz, staged, x0, y0, z0, acc);
          __syncthreads();
        }
      }
    }
  }
  store_cell_group(output, grid, tile_x, tile_y, tile_z, x0, y0, z0, acc);
}

}  // namespace

void launch_assign_bins(const Point *points, int count, Grid grid, int *bin_ids) {
  constexpr int THREADS = 256;
  assign_bins<<<(count + THREADS - 1) / THREADS, THREADS>>>(points, count, grid, bin_ids);
}

void launch_spread(const Point *points, const int *bin_offsets, Grid grid,
                   cuFloatComplex *output) {
  dim3 threads(BLOCK_X, BLOCK_Y, BLOCK_Z);
  dim3 blocks(1 + (grid.nx - 1) / TILE_X, 1 + (grid.ny - 1) / TILE_Y,
              1 + (grid.nz - 1) / TILE_Z);
  spread<<<blocks, threads>>>(points, bin_offsets, grid, output);
}

}  // namespace atomic_free_spread
