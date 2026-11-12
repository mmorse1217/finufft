#include "gpu_bin_sort.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace gpu_bin_sort {
namespace {
constexpr int THREADS = 256;
using atomic_free_spread::Point;

__global__ void histogram(const int *keys, int count, int bins, int *counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Invalid keys go to a sentinel rather than corrupting memory. The caller
  // must validate keys; offsets[bins] will exclude invalid points.
  if (i < count) atomicAdd(&counts[keys[i] >= 0 && keys[i] < bins ? keys[i] : bins], 1);
}

__global__ void iota(int *indices, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) indices[i] = i;
}

__global__ void gather(const Point *points, const int *indices, int count, Point *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = points[indices[i]];
}

void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}  // namespace

BinSorter::BinSorter(int capacity, int bins) : capacity_(capacity), bins_(bins) {
  if (capacity < 0 || bins < 1 || bins == INT_MAX)
    throw std::invalid_argument("BinSorter requires capacity >= 0 and 0 < bins < INT_MAX");
  bits_ = bins > 1 ? 32 - __builtin_clz(bins - 1) : 0;
  const size_t n = std::max(capacity, 1);
  void *p = nullptr;
  check(cudaMalloc(&p, (4 * n + size_t(bins) + 1) * sizeof(int)));
  storage_.reset(p);
  keys_ = static_cast<int *>(p);
  keys_out_ = keys_ + n;
  indices_ = keys_out_ + n;
  indices_out_ = indices_ + n;
  counts_ = indices_out_ + n;
  cub::DoubleBuffer<int> keys(keys_, keys_out_), indices(indices_, indices_out_);
  if (capacity > 1 && bits_)
    check(cub::DeviceRadixSort::SortPairs(nullptr, radix_bytes_, keys, indices,
                                         capacity, 0, bits_));
  check(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes_, counts_, counts_, bins + 1));
  check(cudaMalloc(&p, std::max(size_t(1), std::max(radix_bytes_, scan_bytes_))));
  temp_.reset(p);
}

#define TRY(call) do { cudaError_t e = (call); if (e != cudaSuccess) return e; } while (0)

cudaError_t BinSorter::run(const Point *points, const int *bin_ids, int count,
                           int *offsets, Point *sorted) {
  if (count < 0 || count > capacity_) return cudaErrorInvalidValue;
  if (!count) return cudaMemsetAsync(offsets, 0, (size_t(bins_) + 1) * sizeof(int));
  const int blocks = 1 + (count - 1) / THREADS;
  // CUB may modify both buffers, so preserve the caller's bin IDs.
  TRY(cudaMemcpyAsync(keys_, bin_ids, size_t(count) * sizeof(int), cudaMemcpyDeviceToDevice));
  TRY(cudaMemsetAsync(counts_, 0, (size_t(bins_) + 1) * sizeof(int)));
  histogram<<<blocks, THREADS>>>(keys_, count, bins_, counts_);
  TRY(cudaGetLastError());
  TRY(cub::DeviceScan::ExclusiveSum(temp_.get(), scan_bytes_, counts_, offsets, bins_ + 1));
  iota<<<blocks, THREADS>>>(indices_, count);
  TRY(cudaGetLastError());
  cub::DoubleBuffer<int> keys(keys_, keys_out_), indices(indices_, indices_out_);
  if (count > 1 && bits_)
    TRY(cub::DeviceRadixSort::SortPairs(temp_.get(), radix_bytes_, keys, indices,
                                       count, 0, bits_));
  gather<<<blocks, THREADS>>>(points, indices.Current(), count, sorted);
  return cudaGetLastError();
}
#undef TRY
}  // namespace gpu_bin_sort
