// Exact stable-sort checks, including empty inputs and repeated workspace reuse.
#include "gpu_bin_sort.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

template<class T> struct Device {
  T *p = nullptr;
  explicit Device(size_t n) { check(cudaMalloc(&p, std::max(size_t(1), n) * sizeof(T))); }
  ~Device() { cudaFree(p); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
};

void test(int capacity, int bins) {
  using atomic_free_spread::Point;
  gpu_bin_sort::BinSorter sorter(capacity, bins);
  Device<Point> input(capacity), output(capacity);
  Device<int> keys(capacity), offsets(bins + 1);
  // Vary size and keys on one workspace; large-small-large catches stale buffers.
  for (int count : {capacity, 0, std::min(1, capacity), capacity / 3, capacity}) {
    std::vector<int> host_keys(count), order(count), expected_offsets(bins + 1, 0);
    std::vector<Point> points(count), actual(count), expected(count);
    for (int i = 0; i < count; ++i) {
      host_keys[i] = int((unsigned(i) * 2654435761u + count) % bins);
      points[i] = {float(i), float(-i), float(i % 7), make_cuFloatComplex(float(i), 1)};
      ++expected_offsets[host_keys[i] + 1];
    }
    std::partial_sum(expected_offsets.begin(), expected_offsets.end(), expected_offsets.begin());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return host_keys[a] < host_keys[b];
    });
    for (int i = 0; i < count; ++i) expected[i] = points[order[i]];
    if (count) {
      check(cudaMemcpy(input.p, points.data(), count * sizeof(Point), cudaMemcpyHostToDevice));
      check(cudaMemcpy(keys.p, host_keys.data(), count * sizeof(int), cudaMemcpyHostToDevice));
    }
    check(cudaMemset(offsets.p, 0xff, (bins + 1) * sizeof(int)));
    check(cudaMemset(output.p, 0xff, std::max(1, count) * sizeof(Point)));
    check(sorter.run(input.p, keys.p, count, offsets.p, output.p));
    std::vector<int> actual_offsets(bins + 1), unchanged_keys(count);
    check(cudaMemcpy(actual_offsets.data(), offsets.p, (bins + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    if (count) {
      check(cudaMemcpy(actual.data(), output.p, count * sizeof(Point), cudaMemcpyDeviceToHost));
      check(cudaMemcpy(unchanged_keys.data(), keys.p, count * sizeof(int), cudaMemcpyDeviceToHost));
    }
    if (actual_offsets != expected_offsets || unchanged_keys != host_keys ||
        (count && std::memcmp(actual.data(), expected.data(), count * sizeof(Point))))
      throw std::runtime_error("sort differs from stable CPU reference");
  }
  if (sorter.run(input.p, keys.p, -1, offsets.p, output.p) != cudaErrorInvalidValue ||
      sorter.run(input.p, keys.p, capacity + 1, offsets.p, output.p) != cudaErrorInvalidValue)
    throw std::runtime_error("invalid count accepted");
  std::printf("capacity=%d bins=%d PASS\n", capacity, bins);
}

int main() {
  try {
    for (int bins : {1, 3, 4096, 32768})
      for (int count : {0, 1, 31, 20000, 1000000}) test(count, bins);
    std::puts("all cases PASS");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
