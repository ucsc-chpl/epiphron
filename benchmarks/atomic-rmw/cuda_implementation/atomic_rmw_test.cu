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

#pragma nv_exec_check_disable
__global__ void fetch_add(uint *res, uint iter, uint *mapping) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint index = mapping[global_id];

    for (uint i = 0; i < iter; i++) {
        atomicAdd(&res[index], 1);
    }
}

void run(uint32_t workgroups, uint32_t workgroup_size, uint32_t test_iters) {
    
    list<uint32_t> test_values;
    uint32_t error_count = 0;

    uint32_t tmp = (workgroups * workgroup_size > 1024) ? 1024 : workgroup_size * workgroups;

    for (uint32_t i = 1; i <= tmp; i *= 2) {
        test_values.push_back(i);  
    } 

    for (auto c = test_values.begin(); c != test_values.end(); c++) {
        for (auto p = test_values.begin(); p != test_values.end(); p++) {
            uint32_t contention = *c;
            uint32_t padding = *p;
            benchmarkData << "(" + to_string(contention) + ", " + to_string(padding) + ", ";

            float observed_rate = 0.0;
            uint32_t rmw_iters = 16;

            uint32_t global_work_size = workgroup_size * workgroups;
            const int size = ((global_work_size) * padding) / contention;

            uint32_t *h_strat, *h_result;
            uint32_t *d_strat, *d_result;

            size_t bytes_strat = global_work_size * sizeof(uint32_t);
            size_t bytes_result = size * sizeof(uint32_t);

            h_strat = (uint32_t *)malloc(bytes_strat);
            h_result = (uint32_t *)malloc(bytes_result);

            cudaMalloc(&d_strat, bytes_strat);
            cudaMalloc(&d_result, bytes_result);

            for (int i = 0; i < global_work_size; i += 1) h_strat[i] = (i / contention) * padding;
            cudaMemcpy(d_strat, h_strat, bytes_strat, cudaMemcpyHostToDevice);

            while(true) { 
                for (int i = 0; i < size; i++) h_result[i] = 0;
                cudaMemcpy(d_result, h_result, bytes_result, cudaMemcpyHostToDevice);
                
                observed_rate = 0.0;
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                for (int i = 1; i <= test_iters; i++) {
                    float kernelTime = 0;
                    cudaEventRecord(start);
                    CUDA_SAFECALL((fetch_add<<<workgroups, workgroup_size>>>(d_result, rmw_iters, d_strat)));
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&kernelTime, start, stop);
                    observed_rate += (float(rmw_iters * workgroup_size * workgroups) / float((kernelTime / (double) 1000.0))); 
                }
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                observed_rate /= float(test_iters);
                
                if (isinf(observed_rate)) rmw_iters *= 2;
                else break;
            }
            benchmarkData << to_string(observed_rate) + ")" << endl;

            cudaMemcpy(h_result, d_result, bytes_result, cudaMemcpyDeviceToHost);
            for (int access = 0; access < size; access += padding) {
                if (h_result[access] != rmw_iters * test_iters * contention) error_count += 1;
            }
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

    int device, numBlocks, activeWarps;
    cudaGetDevice(&device);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    uint32_t test_iters = 32;
    int blockSize = deviceProp.maxThreadsPerBlock;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        fetch_add,
        blockSize,
        0);
    activeWarps = numBlocks * blockSize / deviceProp.warpSize;

    cout << "Occupancy: " << (double)activeWarps / (deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize) * 100 << "%" << endl;

    benchmarkData << to_string(blockSize) + "," + to_string(activeWarps) + ":" + deviceProp.name + ", contiguous_access: atomic_fetch_add\n";
    run(activeWarps, blockSize, test_iters);

    benchmarkData.close();


    return 0;
}
