#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <list>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
using namespace std;
ofstream benchmarkData; 

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

__global__ void fetch_add(uint *res, uint iter, uint *indexes) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint index = indexes[global_id];

    for (uint i = 0; i < iter; i++) {
        atomicAdd(&res[index], 1);
    }
}


void run(uint32_t workgroups, uint32_t workgroup_size, uint32_t test_iters) {
    
    list<uint32_t> test_values;
    uint32_t tmp;
    uint32_t error_count = 0;
    if (workgroups * workgroup_size > 1024) tmp = 1024;
    else tmp = workgroup_size * workgroups;

    for (uint32_t i = 1; i <= tmp; i *= 2) {
        test_values.push_back(i);  
    } 

    for (auto it1 = test_values.begin(); it1 != test_values.end(); ++it1) {
        for (auto it2 = test_values.begin(); it2 != test_values.end(); ++it2) {
            uint32_t contention = *it1;
            uint32_t padding = *it2;
            benchmarkData << "(" + to_string(contention) + ", " + to_string(padding) + ", ";

            float observed_rate = 0.0;
            uint32_t rmw_iters = 16;

            uint32_t global_work_size = workgroup_size * workgroups;
            const int size = ((global_work_size) * padding) / contention;

            // Host input and output vectors
            uint32_t *h_strat;
            // Host output vector
            uint32_t *h_result;

            // Device input vectors
            uint32_t *d_strat;
            // Device output vector
            uint32_t *d_result;
            
            size_t bytes_strat = global_work_size * sizeof(uint32_t);
            size_t bytes_result = size * sizeof(uint32_t);

            // Allocate memory for each vector on host
            h_strat = (uint32_t *)malloc(bytes_strat);
            h_result = (uint32_t *)malloc(bytes_result);

            // Allocate memory for each vector on GPU
            cudaMalloc(&d_strat, bytes_strat);
            cudaMalloc(&d_result, bytes_result);

            for (int i = 0; i < global_work_size; i += 1) h_strat[i] = (i / contention) * padding;
            // Copy host vectors to device
            cudaMemcpy(d_strat, h_strat, bytes_strat, cudaMemcpyHostToDevice);

            while(true) { 
                for (int i = 0; i < size; i++) h_result[i] = 0;
                cudaMemcpy(d_result, h_result, bytes_result, cudaMemcpyHostToDevice);
                
                observed_rate = 0.0;
                for (int i = 1; i <= test_iters; i++) {
                    float kernelTime = 0;
                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);

                    // Execute the kernel
                    cudaEventRecord(start);
                    CUDA_SAFECALL((fetch_add<<<workgroups, workgroup_size>>>(d_result, rmw_iters, d_strat)));
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&kernelTime, start, stop);

                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);
                    cudaDeviceSynchronize();
                    observed_rate += (float(rmw_iters * workgroup_size * workgroups) / float((kernelTime / (double) 1000.0))); 
                }
                observed_rate /= float(test_iters);
                
                if (isinf(observed_rate)) rmw_iters *= 2;
                else break;
            }
            benchmarkData << to_string(observed_rate) + ")" << endl;
            cudaMemcpy(h_result, d_result, bytes_result, cudaMemcpyDeviceToHost);

            for (int access = 0; access < size; access += padding) {
                if (h_result[access] != rmw_iters * test_iters * contention) error_count += 1;
            }

            // Release device and host memory
            cudaFree(d_result);
            cudaFree(d_strat);
            free(h_result);
            free(h_strat);
    
        }
    }
    cout << "Error count: " << error_count << endl;

    return;
}


int main(int argc, char *argv[]) {
    benchmarkData.open("result.txt"); 

    if (!benchmarkData.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return 1;
    }
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    uint32_t test_iters = 64;
    uint32_t workgroup_size =  deviceProp.maxThreadsPerBlock;
    uint32_t workgroups = deviceProp.multiProcessorCount * (deviceProp.maxThreadsPerMultiProcessor / workgroup_size);

    benchmarkData << to_string(workgroup_size) + "," + to_string(workgroups) + ":" + deviceProp.name + ", contiguous_access: atomic_fa_relaxed\n";

    run(workgroups, workgroup_size, test_iters);

    benchmarkData.close();


    return 0;
}
