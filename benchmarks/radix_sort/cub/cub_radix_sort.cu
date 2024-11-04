#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <list>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <cub/cub.cuh>
using namespace std;
ofstream benchmarkData;

//https://github.com/owensgroup/GpuMultisplit/blob/cd529b6495cdb91237a27acba0e876aec409c6a1/src/main/main_sort.cu

cub::CachingDeviceAllocator  g_allocator(true);

int main(int argc, char *argv[]) {

    // Setup benchmark results file 
    benchmarkData.open("result.txt"); 
    if (!benchmarkData.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return 1;
    }
    benchmarkData << "(num_items, Mkeys/sec)" << endl;

    srand(time(NULL));

    // number of elements
    for (uint32_t num_items = 131072; num_items <= 8388608; num_items += 131072) {

        //Num trials
        uint32_t num_trials = 3;

        //Timing
        float kernel_time = 0.0;
        float cub_time = 0.0;
        
        // Host input vector
        uint32_t* h_in, *h_out;

        // Device input/output vector
        uint32_t *d_in, *d_out;

        //Size, in bytes, of input size
        size_t input_size = num_items * sizeof(uint32_t);

        // Allocate memory for vector on host
        h_in = (uint32_t *)malloc(input_size);
        h_out = (uint32_t *)malloc(input_size);

        // Allocate memory for vector on GPU
        cudaMalloc((void**)&d_in, input_size);
        cudaMalloc((void**)&d_out, input_size);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
        g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

        benchmarkData << "(" << num_items << ", ";

        for (int curr_trial = 0; curr_trial < num_trials; curr_trial += 1) {

            // Initialize vectors on host
            for (int i = 0; i < num_items; i++) h_in[i] = rand();

            // Copy host vectors to device
            cudaMemcpy(d_in, h_in, input_size, cudaMemcpyHostToDevice);

            cudaEventRecord(start, 0);
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
            cudaEventRecord(stop, 0);

            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&kernel_time, start, stop);

            cub_time += kernel_time;

        }
        benchmarkData << (num_items / ((cub_time/num_trials) / 1000.0f)) * 1e-6 << ")" << endl;

        // Release device and host memory
        cudaEventDestroy(start);
	    cudaEventDestroy(stop);
        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);
        free(h_out);
        g_allocator.DeviceFree(d_temp_storage);
    }

    benchmarkData.close();

    return 0;

}
