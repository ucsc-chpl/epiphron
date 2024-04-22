#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>

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

__global__ void fetch_add(uint *res, uint *iter, uint *indexes) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint index = indexes[global_id];

    for (uint i = 0; i < *iter; i++) {
        atomicAdd(&res[index], 1);
    }
}



int main(int argc, char *argv[]) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    uint32_t test_iters = 64;
    uint32_t workgroup_size =  deviceProp.maxThreadsPerBlock;
    uint32_t workgroups = deviceProp.multiProcessorCount * (deviceProp.maxThreadsPerMultiProcessor / workgroup_size);

    return 0;
}
