# l1-primitive-barrier


## Background 

GPU programming models typically provide an execution and memory abstraction which reflects the hiearchical nature of the hardware. 
Individual threads reside at the base level, organized into *subgroups*, which typically execute on the same SIMD execution group (which AMD calls *wavefronts* and Nvidia calls *warps*) [1].
This means that threads within a subgroup can exploit fast (thread-local) private memory for intra-subgroup communication.
Subgroups are further organized into *workgroups*, where subgroups within the same workgroup get executed in the same *compute unit* (Nvidia calls *streaming multiprocessors* and Intel calls *execution units*).
Intra-workgroup communication can be efficiently achieved via fast local memory which is local to a CU (but not visible across workgroups).
Finally, a kernel is executed by multiple workgroups. 
Inter-workgroup communication can be achieved via global memory (typically just GPU main memory), which is slower but very large.

OpenCL provides primitives which allow for direct control across varying levels of the hiearchy. 
The `sub_group_barrier` ensures all threads within the same subgroup must reach the barrier before continuing their execution [2]. 
An additonal `cl_mem_fence_flag` can be provided to the barrier call to specify whether global or local memory should be flushed.
Intra-workgroup thread synchronization can be achived with the `work_group_barrier`, which can also be supplied with a memory fence flag.

## Goal

By evaluating the performance of these barriers across increasing levels of the execution and memory hiearchy (subgroup local -> subgroup global -> workgroup local -> workgroup global), we hope to attain a performance model for how a given GPU implements these barrier primitives. 
This can help to inform programmers what is worth optimizing on a given device and how performance may drastically change across multiple devices.


## Implementation


### Host
```C++
for (auto i = 0; i < numTrials; i++) {
    times[i] = program.runWithDispatchTiming();
}

auto avgTime = calculate_average(times);
auto timeStdDev = calculate_std_dev(times);
```


### Kernel
```C
#define LOCAL_BUF_SIZE 256 * 1

kernel void benchmark(global uint *buf, global uint *buf_size, global uint *num_iters) {
    // Workgroup-local memory.
    local uint local_buf[LOCAL_BUF_SIZE]; 
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

        // Barrier here (one of these): 
        // work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        // work_group_barrier(CLK_LOCAL_MEM_FENCE);
        // sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
        // sub_group_barrier(CLK_LOCAL_MEM_FENCE;
        // no barrier
    }
}
```

We are measuring the throughput of a given barrier, so we run a loop for a fixed number of iterations and time how long it takes. 
Work is done within the loop to ensure that the barrier is actually needed and compiler cannot optimize anything away.


# References
1. Sorensen, “Inter-Workgroup Barrier Synchronisation on Graphics Processing Units.”
2. https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html