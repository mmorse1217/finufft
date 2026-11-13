#include "gpu_bin_sort.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace gpu_bin_sort {
namespace {
constexpr int THREADS = 256;
using atomic_free_spread::Point;

__global__ void initialize(const int *bin_ids, int count, int *keys, int *indices) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    keys[i] = bin_ids[i];
    indices[i] = i;
  }
}

__global__ void gather(const Point *points, const int *indices, const int *keys,
                       int count, int bins, Point *out, int *offsets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > count) return;
  if (i < count) out[i] = points[indices[i]];
  // Each key transition owns the intervening bin starts, including empty bins.
  // The extra thread at count fills the trailing empty bins and final offset.
  int previous = i ? max(-1, min(keys[i - 1], bins)) : -1;
  int current = i < count ? min(keys[i], bins) : bins;
  for (int bin = previous; bin < current; ++bin) offsets[bin + 1] = i;
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
  check(cudaMalloc(&p, 4 * n * sizeof(int)));
  storage_.reset(p);
  keys_ = static_cast<int *>(p);
  keys_out_ = keys_ + n;
  indices_ = keys_out_ + n;
  indices_out_ = indices_ + n;
  cub::DoubleBuffer<int> keys(keys_, keys_out_), indices(indices_, indices_out_);
  if (capacity > 1 && bits_)
    check(cub::DeviceRadixSort::SortPairs(nullptr, radix_bytes_, keys, indices,
                                         capacity, 0, bits_));
  check(cudaMalloc(&p, std::max(size_t(1), radix_bytes_)));
  temp_.reset(p);
}

#define TRY(call) do { cudaError_t e = (call); if (e != cudaSuccess) return e; } while (0)

cudaError_t BinSorter::run(const Point *points, const int *bin_ids, int count,
                           int *offsets, Point *sorted) {
  if (count < 0 || count > capacity_) return cudaErrorInvalidValue;
  if (!count) return cudaMemsetAsync(offsets, 0, (size_t(bins_) + 1) * sizeof(int));
  const int blocks = 1 + (count - 1) / THREADS;
  // CUB may modify both buffers, so preserve the caller's bin IDs.
  initialize<<<blocks, THREADS>>>(bin_ids, count, keys_, indices_);
  TRY(cudaGetLastError());
  cub::DoubleBuffer<int> keys(keys_, keys_out_), indices(indices_, indices_out_);
  if (count > 1 && bits_)
    TRY(cub::DeviceRadixSort::SortPairs(temp_.get(), radix_bytes_, keys, indices,
                                       count, 0, bits_));
  gather<<<1 + count / THREADS, THREADS>>>(points, indices.Current(), keys.Current(),
                                          count, bins_, sorted, offsets);
  return cudaGetLastError();
}
#undef TRY
}  // namespace gpu_bin_sort
