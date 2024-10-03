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
//#include <cub/device/device_radix_sort.cuh>

using namespace std;
ofstream benchmarkData; 

// https://github.com/NVIDIA/cccl/blob/1cfe171ee948626668aa90a1922d744ff69b9ecd/cub/benchmarks/bench/radix_sort/keys.cu
// create policy
// attach policy to dispatch radix sort
// how to dispatch a dispatch_t
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
        //float onesweep_res = benchmark_cub_onesweep(d_in, num_items);
        float radixsort_res = benchmark_cub_radixsort(d_in, d_out, num_items);

        benchmarkData << radixsort_res << ")" << endl;

        // Release device and host memory
        cudaFree(d_in);
        free(h_in);
    }
    
    benchmarkData.close();

    return 0;

}
