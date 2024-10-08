#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <random>
#include <list>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_arch.cuh>
#include <cuda/std/type_traits>
#include <thrust/device_vector.h>
using namespace std;
ofstream benchmarkData;

#define TUNE_THREADS_PER_BLOCK 256
#define TUNE_ITEMS_PER_THREAD 8
#define TUNE_RADIX_BITS 8

using value_t = cub::NullType;

constexpr bool is_descending   = false;
constexpr bool is_overwrite_ok = false;

// Custom policy hub for Onesweep-based Radix Sort
template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_hub_t {
    static constexpr bool KEYS_ONLY = std::is_same<ValueT, cub::NullType>::value;
    using DominantT = ::cuda::std::_If<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

    struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t> {
        static constexpr int ONESWEEP_RADIX_BITS = TUNE_RADIX_BITS;
        static constexpr bool ONESWEEP = true;

        using OnesweepPolicy = cub::AgentRadixSortOnesweepPolicy<
            TUNE_THREADS_PER_BLOCK,
            TUNE_ITEMS_PER_THREAD,
            DominantT,
            1,
            cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
            cub::BLOCK_SCAN_RAKING_MEMOIZE,
            cub::RADIX_SORT_STORE_DIRECT,
            ONESWEEP_RADIX_BITS>;
    };

    using MaxPolicy = policy_t;
};

template <typename KeyT, typename ValueT, typename OffsetT>
constexpr std::size_t max_onesweep_temp_storage_size()
{
  using portion_offset  = int;
  using onesweep_policy = typename policy_hub_t<KeyT, ValueT, OffsetT>::policy_t::OnesweepPolicy;
  using agent_radix_sort_onesweep_t = cub::AgentRadixSortOnesweep<onesweep_policy, is_descending, KeyT, ValueT, OffsetT, portion_offset>;

  return sizeof(typename agent_radix_sort_onesweep_t::TempStorage);
}

template <typename KeyT, typename ValueT, typename OffsetT>
constexpr std::size_t max_temp_storage_size()
{
  using policy_t = typename policy_hub_t<KeyT, ValueT, OffsetT>::policy_t;

  static_assert(policy_t::ONESWEEP);
  return max_onesweep_temp_storage_size<KeyT, ValueT, OffsetT>();
}

template <typename KeyT, typename ValueT, typename OffsetT>
constexpr bool fits_in_default_shared_memory()
{
  return max_temp_storage_size<KeyT, ValueT, OffsetT>() < cub::detail::max_smem_per_block;
}

// Benchmark function for onesweep-enabled radix sort
template <typename KeyT, typename OffsetT>
float radix_sort_keys(std::integral_constant<bool, true>, int num_items) {

    using offset_t = cub::detail::choose_offset_t<OffsetT>;

    using key_t = KeyT;
    using policy_t   = policy_hub_t<key_t, value_t, offset_t>;
    using dispatch_t = cub::DispatchRadixSort<is_descending, key_t, value_t, offset_t, policy_t>;

    constexpr int begin_bit = 0;
    constexpr int end_bit   = sizeof(key_t) * 8;

    const auto elements = num_items;

    thrust::device_vector<KeyT> buffer_1(elements);

    // need to seed a sequence of elements here

    key_t* d_buffer_1 = thrust::raw_pointer_cast(buffer_1.data());

    cub::DoubleBuffer<key_t> d_keys(d_buffer_1);
    cub::DoubleBuffer<value_t> d_values;
    std::size_t temp_size{};

    dispatch_t::Dispatch(
    nullptr,
    temp_size,
    d_keys,
    d_values,
    static_cast<offset_t>(elements),
    begin_bit,
    end_bit,
    is_overwrite_ok,
    0 /* stream */);

    thrust::device_vector<uint8_t> temp(temp_size);
    auto* temp_storage = thrust::raw_pointer_cast(temp.data());

    cudaEvent_t start, stop;

    float observed_rate = 0.0;

    for (int i = 1; i <= 3; i++) {
        cub::DoubleBuffer<key_t> keys     = d_keys;
        cub::DoubleBuffer<value_t> values = d_values;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        dispatch_t::Dispatch(
        temp_storage,
        temp_size,
        keys,
        values,
        static_cast<offset_t>(elements),
        begin_bit,
        end_bit,
        is_overwrite_ok,
        0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float kernelTime = 0.0;
        cudaEventElapsedTime(&kernelTime, start, stop);
        observed_rate += (float(num_items) / float((kernelTime / 1000.0)));

    }
    observed_rate /= float(3);

    return observed_rate;

}

template <typename T, typename OffsetT>
float benchmark_cub_onesweep(int num_items)
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;

  return radix_sort_keys<T, offset_t>(std::integral_constant<bool, fits_in_default_shared_memory<T, value_t, offset_t>()>{}, num_items);
}


// bar graph, single value, other one line graph
float benchmark_cub_radixsort(int* d_keys_in, int* d_keys_out, int num_items) {

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaEvent_t start, stop;

    float observed_rate = 0.0;

    for (int i = 1; i <= 3; i++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float kernelTime = 0.0;
        cudaEventElapsedTime(&kernelTime, start, stop);
        observed_rate += (float(num_items) / float((kernelTime / 1000.0)));

    }
    observed_rate /= float(3);

    cudaFree(d_keys_out);
    cudaFree(d_temp_storage);

    return observed_rate;
}
int main(int argc, char *argv[]) {

    // Setup benchmark results file 
    benchmarkData.open("result.txt"); 
    if (!benchmarkData.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return 1;
    }

    // number of elements
    for (int num_items = 1 << 20; num_items <= 1 << 28; num_items *= 2) {
        
        // Host input vector
        int* h_in;

        // Device input/output vector
        int *d_in, *d_out;

        //Size, in bytes, of input size
        size_t input_size = num_items * sizeof(int);

        // Allocate memory for vector on host
        h_in = (int *)malloc(input_size);

        // Allocate memory for vector on GPU
        cudaMalloc(&d_in, input_size);
        cudaMalloc(&d_out, input_size);

        // Initialize vectors on host
        for (int i = 0; i < num_items; i++) h_in[i] = rand() % 100; // seed this

        // Copy host vectors to device
        cudaMemcpy(d_in, h_in, input_size, cudaMemcpyHostToDevice);

        benchmarkData << "(" << input_size / (1024 * 1024) << ", ";

        // Run benchmark
        float onesweep_res = benchmark_cub_onesweep<int, int>(num_items);
        float radixsort_res = benchmark_cub_radixsort(d_in, d_out, num_items);

        benchmarkData << radixsort_res << ", " << onesweep_res << ")" << endl;

        // Release device and host memory
        cudaFree(d_in);
        free(h_in);
    }
    
    benchmarkData.close();

    return 0;

}
