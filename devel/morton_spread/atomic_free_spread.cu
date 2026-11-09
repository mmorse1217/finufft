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
// The modulo reduces by any number of grid periods. This is why this baseline
// does not need the multi-period correction that place_support() in the
// optimized kernel requires for lane groups near the far edge of the grid.
__device__ int coefficient_index(int cell, int support_start, int grid_size) {
  int offset = (cell - support_start) % grid_size;
  if (offset < 0) offset += grid_size;
  return offset < WIDTH ? offset : -1;
}

// One block owns one tile; one thread owns one cell. All threads independently
// enumerate the tile's candidate points. This repeats work, but makes the
// ownership and accumulation especially easy to follow.
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
  __syncthreads();
  if (x >= grid.nx || y >= grid.ny || z >= grid.nz) return;

  float real = 0.0f, imag = 0.0f;

  for (int z_bin_index = 0; z_bin_index < z_bins.count; ++z_bin_index) {
    const int bz = z_bins[z_bin_index];
    for (int y_bin_index = 0; y_bin_index < y_bins.count; ++y_bin_index) {
      const int by = y_bins[y_bin_index];
      for (int x_bin_index = 0; x_bin_index < x_bins.count; ++x_bin_index) {
        const int bx = x_bins[x_bin_index];
        const int bin = bx + grid.bins_x * (by + grid.bins_y * bz);
        for (int i = bin_offsets[bin]; i < bin_offsets[bin + 1]; ++i) {
          Point p = points[i];
          float px = fold_rescale(p.x, grid.nx);
          float py = fold_rescale(p.y, grid.ny);
          float pz = fold_rescale(p.z, grid.nz);
          int sx = int(ceilf(px - HALO));
          int sy = int(ceilf(py - HALO));
          int sz = int(ceilf(pz - HALO));
          int ix = coefficient_index(x, sx, grid.nx);
          int iy = coefficient_index(y, sy, grid.ny);
          int iz = coefficient_index(z, sz, grid.nz);
          if (ix < 0 || iy < 0 || iz < 0) continue;

          float kx[WIDTH], ky[WIDTH], kz[WIDTH];
          eval_kernel_vec_horner<float, WIDTH>(kx, float(sx) - px, 2.0);
          eval_kernel_vec_horner<float, WIDTH>(ky, float(sy) - py, 2.0);
          eval_kernel_vec_horner<float, WIDTH>(kz, float(sz) - pz, 2.0);
          float weight = kx[ix] * ky[iy] * kz[iz];
          real += p.strength.x * weight;
          imag += p.strength.y * weight;
        }
      }
    }
  }
  long long index = x + (long long)grid.nx * (y + (long long)grid.ny * z);
  output[index] = make_cuFloatComplex(real, imag);
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
