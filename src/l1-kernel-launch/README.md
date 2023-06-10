# l1-kernel-launch

The goal of these benchmarks is to measure the overhead of launching a compute kernel. 

This is done by calculating the average $utilization$ of a kernel, which is the ratio of time spent within the GPU doing work to the total time measured by the host to submit and complete the kernel:

$$
utilization = \frac{gpuTime}{cpuTime}
$$

The pseudocode for how we calculate utilization is as follows:

```Python
total = 0
for _ in num_trials:
    cpu_start = time.now()
    gpu_time = program.run()
    cpu_end = time.now()
    total += gpu_time / (cpu_end - cpu_start)
 
# Record average utilization amount for this workload 
utilization = total / num_trials
```

`gpuTime` is measured using Vulkan timestamp queries to find the exact time spent in the compute kernel.

## Varied Dispatch Benchmark 

This benchmark runs a kernel which performs vector addition on two vectors. 
Workgroup size stays fixed, but the number of workgroups dispatched varies.
The kernel is written so that every thread only operates on one element of the vector (thread workload remains fixed), so the vector size for a given run of the test is `num_workgroups` * `workgroup_size`.

## Varied Thread Workload Benchmark

This benchmark also runs a vector addition kernel, but the threads operate on fixed size chunks of the vector (instead of just a single element). This chunk size is parameterized as `threadWorkload` across the test, and `numWorkgroups` and `workgroupSize` remains fixed. For a given run of the test, the amount of work each thread is doing is `threadWorkload` and the total size of the vector being operated on is `threadWorkload` * `numWorkgroups` * `workgroupSize`.