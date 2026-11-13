#include "gpu_bin_sort.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace gpu_bin_sort {
namespace {
constexpr int THREADS = 256;
using atomic_free_spread::Point;

__global__ void make_offsets(const int *keys, int count, int bins, int *offsets) {
  int bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin > bins) return;
  int first = 0, last = count;
  while (first < last) {
    int mid = first + (last - first) / 2;
    if (keys[mid] < bin) first = mid + 1;
    else last = mid;
  }
  offsets[bin] = first;
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
  check(cudaMalloc(&p, n * sizeof(int)));
  storage_.reset(p);
  keys_ = static_cast<int *>(p);
  check(cub::DeviceRadixSort::SortPairs(nullptr, radix_bytes_,
                                       static_cast<const int *>(nullptr), keys_,
                                       static_cast<const Point *>(nullptr),
                                       static_cast<Point *>(nullptr), capacity, 0, bits_));
  check(cudaMalloc(&p, std::max(size_t(1), radix_bytes_)));
  temp_.reset(p);
}

#define TRY(call) do { cudaError_t e = (call); if (e != cudaSuccess) return e; } while (0)

cudaError_t BinSorter::run(const Point *points, const int *bin_ids, int count,
                           int *offsets, Point *sorted) {
  if (count < 0 || count > capacity_) return cudaErrorInvalidValue;
  if (!count) return cudaMemsetAsync(offsets, 0, (size_t(bins_) + 1) * sizeof(int));
  TRY(cub::DeviceRadixSort::SortPairs(temp_.get(), radix_bytes_, bin_ids, keys_,
                                     points, sorted, count, 0, bits_));
  make_offsets<<<1 + bins_ / THREADS, THREADS>>>(keys_, count, bins_, offsets);
  return cudaGetLastError();
}
#undef TRY
}  // namespace gpu_bin_sort
