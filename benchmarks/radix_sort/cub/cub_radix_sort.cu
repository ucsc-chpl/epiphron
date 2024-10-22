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

//change to report MKeys/sec, graph both chrono/cuda events, compare with cub, vkradixsort, google's radix sort - 32 bits

void benchmark_cub_radixsort(uint32_t* d_keys_in, uint32_t* d_keys_out, uint32_t num_items) {

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, 0, (sizeof(uint32_t) * 8), 0);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    cudaEventRecord(start, 0);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, 0, (sizeof(uint32_t) * 8), 0);
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cudaEventSynchronize(stop);

    double chronoTime = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) * std::pow(10, -3));

    float kernelTime = 0.0;
    cudaEventElapsedTime(&kernelTime, start, stop);

    benchmarkData << (num_items / (kernelTime / 1000.0)) * 1e-6 << ", " << (num_items / (chronoTime  / 1000.0)) * 1e-6 << ")" << endl;

    cudaFree(d_keys_out);
    cudaFree(d_temp_storage);

}
int main(int argc, char *argv[]) {

    // Setup benchmark results file 
    benchmarkData.open("result.txt"); 
    if (!benchmarkData.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return 1;
    }
    benchmarkData << "(num_items, cuda_events (ms), chrono (ms))" << endl;
    // number of elements
    for (uint32_t num_items = 1u << 7; num_items != 0 && num_items <= (1u << 31); num_items *= 2) {
        
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
        cudaMalloc(&d_in, input_size);
        cudaMalloc(&d_out, input_size);

        // Initialize vectors on host
        for (int i = 0; i < num_items; i++) h_in[i] = rand() % 10000; // seed this

        // Copy host vectors to device
        cudaMemcpy(d_in, h_in, input_size, cudaMemcpyHostToDevice);

        benchmarkData << "(" << num_items << ", ";

        // Run benchmark
        benchmark_cub_radixsort(d_in, d_out, num_items);

        cudaMemcpy(h_out, d_out, input_size, cudaMemcpyDeviceToHost);

        cout << "Test: " << num_items << endl;
        for (int i = 0; i < num_items - 1; i++) {
            if (h_out[i] > h_out[i+1]) {
                cout << "Incorrect" << endl;
                break;
            }
        }

        // Release device and host memory
        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);
        free(h_out);
    }

    benchmarkData.close();

    return 0;

}
