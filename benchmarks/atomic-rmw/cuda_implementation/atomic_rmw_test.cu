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

static __device__ __inline__ int __mysmid(){
  int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

// CUDA kernel
#pragma nv_exec_check_disable
__global__ void fetch_add(uint *res, uint iter, uint *mapping, float *timing, uint *sm_id) {
    // Get our global thread ID to obtain atomic location
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint index = mapping[global_id];

    // Measure time elapsed (in nanoseconds)
    long long int startTime, stopTime;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(startTime));

    // each thread performs atomicAdd on predetermined location, for fixed iterations
    for (uint i = 0; i < iter; i++) {
            atomicAdd(&res[index], 1);
    }
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stopTime));

    // Representative thread from each block to gather statistics
    if (threadIdx.x == 0) {
        timing[blockIdx.x] = (float)(stopTime - startTime);
        sm_id[blockIdx.x] = __mysmid();
    }
}

void atomic_benchmark(uint32_t workgroups, uint32_t workgroup_size, uint32_t test_iters, uint32_t rmw_iters) {
    
    // Calculate contention/padding testing range based on queried grid/block size
    list<uint32_t> test_values;
    uint32_t error_count = 0;
    uint32_t test_range = (workgroups * workgroup_size > 1024) ? 1024 : workgroup_size * workgroups;
    for (uint32_t i = 1; i <= test_range; i *= 2) test_values.push_back(i);  
    
    // Sweep through predetermined configurations of contention and padding
    for (auto c = test_values.begin(); c != test_values.end(); c++) {
        for (auto p = test_values.begin(); p != test_values.end(); p++) {
            
            // Initialize/Print testing parameters (contention, padding, atomic ops per microsecond)
            uint32_t contention = *c;
            uint32_t padding = *p;
            benchmarkData << "(" + to_string(contention) + ", " + to_string(padding) + ", ";
            cout << "Contention: " + to_string(contention) + ", Padding: " + to_string(padding) << endl;;
            float observed_rate = 0.0;
            
            // Global thread count and predetermined buffer size
            uint32_t global_work_size = workgroup_size * workgroups;
            const int size = ((global_work_size) * padding) / contention;
            
            // Host input/output vector
            uint32_t *h_strat, *h_result, *h_smid;
            float *h_timing;
            
            // Device input/output vectors
            uint32_t *d_strat, *d_result, *d_smid;
            float *d_timing;
            
            // Size, in bytes, of each vector
            size_t bytes_strat = global_work_size * sizeof(uint32_t);
            size_t bytes_result = size * sizeof(uint32_t);
            size_t bytes_timing = workgroups * sizeof(float);
            size_t bytes_smid = workgroups * sizeof(uint32_t);
            
            // Allocate memory for each vector on host
            h_strat = (uint32_t *)malloc(bytes_strat);
            h_result = (uint32_t *)malloc(bytes_result);
            h_timing = (float *)malloc(bytes_timing);
            h_smid = (uint32_t *)malloc(bytes_smid);
            
            // Allocate memory for each vector on GPU
            cudaMalloc(&d_strat, bytes_strat);
            cudaMalloc(&d_result, bytes_result);
            cudaMalloc(&d_timing, bytes_timing);
            cudaMalloc(&d_smid, bytes_smid);

            // Initialize vectors on host
            // Calculate predetermined memory location for each thread, based on contiguous access pattern
            for (int i = 0; i < global_work_size; i += 1) h_strat[i] = (i / contention) * padding;
            for (int i = 0; i < size; i++) h_result[i] = 0;  
            for (int i = 0; i < workgroups; i++) {
                h_smid[i] = 0;
                h_timing[i] = 0;
            }
            
            // Copy host vectors to device
            cudaMemcpy(d_strat, h_strat, bytes_strat, cudaMemcpyHostToDevice);
            cudaMemcpy(d_result, h_result, bytes_result, cudaMemcpyHostToDevice);
            cudaMemcpy(d_smid, h_smid, bytes_smid, cudaMemcpyHostToDevice);
            cudaMemcpy(d_timing, h_timing, bytes_timing, cudaMemcpyHostToDevice);

            // Run for number of trials, take an average throughput
            for (int i = 1; i <= test_iters; i++) {
                // Execute the kernel and synchronize
                CUDA_SAFECALL((fetch_add<<<workgroups, workgroup_size>>>(d_result, rmw_iters, d_strat, d_timing, d_smid)));
                cudaDeviceSynchronize();

                // Sum up timings from each thread block, convert from nanoseconds to microseconds 
                // Calculate currently observed throughput
                cudaMemcpy(h_timing, d_timing, bytes_timing, cudaMemcpyDeviceToHost);
                float kernelTime = 0.0;
                for (int i = 0; i < workgroups; ++i) kernelTime += h_timing[i];
                observed_rate += (float(rmw_iters * workgroup_size * workgroups) / float((kernelTime / 1000.0)));
            }
            // Take the average from sum throughput
            observed_rate /= float(test_iters);
            
            // Copy information back to host
            cudaMemcpy(h_smid, d_smid, bytes_smid, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_result, d_result, bytes_result, cudaMemcpyDeviceToHost);

            // Print configuration statistics and validate results
            for (int i = 0; i < workgroups; i++) printf ("  Block %02d: cycles: %f, SM id: %d\n", i, h_timing[i], h_smid[i]);
            for (int access = 0; access < size; access += padding) {
                if (h_result[access] != rmw_iters * test_iters * contention) error_count += 1;
            }

            // Record atomic throughput for current configuration
            benchmarkData << to_string(observed_rate) + ")" << endl;

            // Release device and host memory
            cudaFree(d_result);
            cudaFree(d_strat);
            cudaFree(d_timing);
            cudaFree(d_smid);
            free(h_result);
            free(h_strat);
            free(h_timing);
            free(h_smid);
        }
    }
    // Print total error count
    cout << " Error count: " << error_count << endl;

    return;
}


int main(int argc, char *argv[]) {

    // Setup benchmark results file 
    benchmarkData.open("result.txt"); 
    if (!benchmarkData.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return 1;
    }

    // Get device properties 
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    
    // Statistics and information on SM id: https://www.stuffedcow.net/research/cudabmk
    printf(" Name: %s\n",deviceProp.name );
    printf(" Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor );
    printf(" Clock rate: %d\n",deviceProp.clockRate );
    printf(" Total global memory: %ld (%d MB)\n", deviceProp.totalGlobalMem, int(deviceProp.totalGlobalMem*9.5367e-7));
    printf(" Multiprocessor count: %d\n", deviceProp.multiProcessorCount);

    // Initialize test (how many iterations of kernel invocation) and rmw (how many executed rmws per kernel invocation)
    uint32_t test_iters = 64;
    uint32_t rmw_iters = 2048;

    // Khronos (OpenCL) terminology: workgroup_size -> thread block size, workgroups -> grid size
    uint32_t workgroup_size =  deviceProp.maxThreadsPerBlock;
    uint32_t workgroups = deviceProp.multiProcessorCount * (deviceProp.maxThreadsPerMultiProcessor / workgroup_size);

    // Title current microbenchmark (currently hardcoded to contiguous access)
    benchmarkData << to_string(workgroup_size) + "," + to_string(workgroups) + ":" + deviceProp.name + ", contiguous_access: atomic_fa_relaxed\n";
    atomic_benchmark(workgroups, workgroup_size, test_iters, rmw_iters);

    benchmarkData.close();

    return 0;
}
