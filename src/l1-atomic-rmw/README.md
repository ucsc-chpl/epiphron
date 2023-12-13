# Atomic-RMW: GPU Microbenchmark 

## Introduction
A microbenchmark made to observe the behavior and measure performance of low level synchronization primitives, atomic RMWs on GPUs. An atomic RMW is an operation that does both read from a memory location and write a new value into it simultaneously, all happens in one atomic, indivisible step. The development of this microbenchmark will help us look for interesting hardware features, specifically on the coalescing of atomics. The act of coalescing is the idea of combining individual pushes into a single push of multiple items utilizing a single rmw. Furthermore, from obtaining data with this microbenchmark, we want to derive a descriptive model for a wide variety of GPUs across multiple vendors, to understand the undisclosed characteristics of GPU microarchitecture and the manufacturer's design decisions.

## Motivation
The most general forms of fine-grained synchronization are one of the most expensive operations on a GPU, such as using fine-grained mutexes to provide mutual exclusion to global memory updates. Also applying GPUs to irregular computations like graph algorithms have become more common in recent times. These computations require the use of a global worklist for dynamic workload balancing, and that each modification to it will require an RMW. The main goal of this microbenchmark is to provide insights on reducing the cost of these expensive operations through reordering or rewriting atomics in a way that provide performance optimizations. 

However, very little is known about these characteristics of GPU's architecture beyond what the manufacturer has documented and there is not a simple solution for this as GPU architecture is very diverse, adding more complexity if we want to identify portable optimizations. Hence, taking this reverse-engineering approach will help us build an intuition and classification about how coalescing patterns fit into certain underlying architectures and further hypothesize about hardware features that lead to these observed shapes and trends. We also need to understand if potential coalescing of atomics could happen in the compiler or hardware level as an OpenCL compiler could exploit atomics (maybe implement a RMW-coalescing transformation for subgroups or have one rmw per thread block) or could automatically be done by the hardware. 

## OpenCL Background

OpenCL is a Khronos standard, which means it is defined by the Khronos group, all the graphics card companies are members. There's a number of implementations of this from different GPU vendors. When you write an OpenCL program, you write a C program to run on the GPU and you will write a C program that is going to run on the host. The host will run device program or kernel launching APIs to launch those kernels on the GPU.

<p align="center">
<figure>
  <img src="https://drive.google.com/uc?id=1hGx5pzDK1TeXS4nzFqy5p1dfOtdR-1NU" alt="OpenCL Memory Model (from [Khronos 2011])" width="300">
  <figcaption>OpenCL Memory Model (from [Khronos 2011])</figcaption>
</figure>
</p>

All threads executing a kernel can access the device’s global memory; where threads can perform a variety of rmw instructions with this memory. And these threads in the same workgroup can communicate through faster local memory

## Heterogeneous System Architecture Cache Hierarchy

<p align="center">
<figure>
  <img src="https://drive.google.com/uc?id=1pZL3SoEZvNtDAlGZmDH3yFFpeV9YC9CV" alt="From Paper 'Synchronization Using Remote-Scope Promotion'" width="300">
  <figcaption>From Paper "Synchronization Using Remote-Scope Promotion"</figcaption>
</figure>
</p>

In a heterogeneous system architecture, threads are partitioned into subgroups that make up workgroups (wg in the picture) and threads within these are scheduled together and can communicate with each other much faster than with threads in other workgroups. 
These workgroups are partitioned into smaller sets called warps or wavefronts (wv in the picture), to match the GPU's execution width. These warps execute on these SIMD units and a workgroup executes on a compute unit which is composed of multiple of these as seen in the picture. Each CU has an L1 and they share a common L2 cache


## Experimental Setup

We want to measure the number of atomic operations per ms by performing an atomic RMW in a loop X times and gather the kernel time, where X is a reasonable estimate for that particular GPU. We will also be filling the GPU by maximizing its global work size, meaning we query the maximum workgroup size and run an occupancy discovery protocol to find the maximum number of concurrently running workgroups possible for the GPU.

Furthermore, we will be testing the behavior of atomic RMWs by simulating increments on an array of atomic ints under the conditions of contention and padding. Contention is the number of threads accessing the same atomic int at the same time and padding is the number of atomic ints in between accesses. To simulate this, we can set our array size to the following: 
`workgroup_size * workgroups * padding / contention`
This is to ensure the maximum number of threads are used based on the value of contention/padding, where increasing padding increases the array size (inserting more ints) and increasing contention would decrease it (more threads accessing the same value).

Now we can proceed with setting up our microbenchmark:
```
__kernel void rmw_test( __global atomic_uint* array, global uint* iters) {
  for (uint i = 0; i < *iters; i++) {
    atomic_fetch_add_explicit(&array[?], 1, memory_order_relaxed);
  }
}
```
This device code will be executed in SPMD, which means that each thread executes the same program, but has access to unique identifiers (thread ID or defined in OpenCL as get_global_id(0)) that can be used to guide threads to different program locations. Knowing this, we can guide threads to perform an atomic add on certain memory locations. We can run different experiments regarding this index calculation, where the contention and/or padding can influence where each thread will perform an atomic add. 

### Contiguous Access
Taking in contention and padding into account, we can perform a contiguous access, which is defined as an operation (atomic add) performed on contiguous indices (same region of memory). This allows for threads from the same warp to access the same atomic location.

`(get_global_id(0) / contention) * padding`

We are dividing the thread ID by contention because we want the number of threads accessing this particular atomic int to be equal to the contention size, then the index is scaled by padding to mark the distance between accesses depending on its value. 
```
__kernel void rmw_test( __global atomic_uint* array, global uint* iters, global uint* chunk) {
  uint index = chunk[get_global_id(0)]; // (get_global_id(0) / contention) * padding
  for (uint i = 0; i < *iters; i++) {
    atomic_fetch_add_explicit(&array[index], 1, memory_order_relaxed);
  }
}
```
The contiguous location is calculated on the host side and we map the corresponding threads calculation by its thread ID to obtain the respective index. This is a compiler trick to avoid any potential optimizations, so we can focus on looking for hardware features. 

### Cross-Warp
With cross-warp, we will only be taking into account padding and the array size. A cross-warp calculation is defined an atomic location determined by thread ID across warps. This allows for threads from different warps to access the same atomic location.

`get_global_id(0) * padding % array_size`

Instead of dividing by contention size, we just scale by padding to ensure that threads within the same warp aren't accessing the same atomic int. Modulate by array size to allow for threads from different warps to access the same atomic int.

```
__kernel void rmw_test( __global atomic_uint* array, global uint* iters, global uint* stride) {
  uint index = stride[get_global_id(0)]; // tid * padding % size
  for (uint i = 0; i < *iters; i++) {
    atomic_fetch_add_explicit(&array[index], 1, memory_order_relaxed);
  }
}
```
The setup from contigious is mostly the same, only difference is that we are performing a different calculation on the host side.

### Branched
This is an experiment where we add a condition on top of an atomic add which will result in half the warp performing atomic operations. The memory locations to operate on are determined by a branch. 
```
__kernel void rmw_test( __global atomic_uint* array, global uint* iters, global uint* stride, global uint* branch) {
  uint index = stride[get_global_id(0)]; // tid * padding % size                          
  for (uint i = 0; i < *iters; i++) {
    if (branch[get_global_id(0)]) { // tid % 2: 0,1,0,1,...
      atomic_fetch_add_explicit(&array[index], 1, memory_order_relaxed);
    }
  }
}
```
We create a bit-vector on the host side where each element is calculated by modulating the thread ID by 2, which results in half the global work size performing atomic operations.

### Random Access 
In Random Access, we have operations (atomic add) performed on random (non-contiguous) indices. To accomplish this, we calculate a seed for each thread on the host side and then perform an LCG per iteration within the kernel allowing each thread to be contending on a new location with each atomic access.

```
__kernel void rmw_test( __global atomic_uint* res, global uint* iters, global uint* seed, global uint* buf_size) {
  uint prev = seed[get_global_id(0)];
  uint index;
  for (uint i = 0; i < *iters; i++) {
    index = ((prev * 1664525) + 1013904223) % (*buf_size);
    atomic_fetch_add_explicit(&res[index], 1, memory_order_relaxed);
    prev = index;
  }
}
```
The multiplicative and additive congruent generators are constant throughout the whole experiment. Any potential optimizations by the compiler are limited as the seed isn't explicit.

## Results

With the setup, we examined the following mix of Integrated and Discrete GPUs (from Concurrency and Heterogeneous Programming Lab):

- AMD Radeon RX 7900 XT
- AMD Ryzen 7 5700G / Radeon Graphics
- NVIDIA Quadro RTX 4000 
- NVIDIA Geforce RTX 4070 
- Intel(R) UHD Graphics 770 (ADL-S GT1) 
- Intel(R) Arc(tm) A770 Graphics (DG2)

### Performance by Vendor (Contiguous Access)

#### AMD

<table>
  <tr>
    <td><img src="https://drive.google.com/uc?id=1U3MPncIHY5OEMyIHEooiiFIoLpieU2lY" alt="Radeon Graphics"></td>
    <td><img src="https://drive.google.com/uc?id=1cHxZyHhficFStVDr4rGyUuRKDz3XEycj" alt="AMD Radeon RX 7900 XT"></td>
  </tr>
</table>

#### NVIDIA

<table>
  <tr>
    <td><img src="https://drive.google.com/uc?id=1VNShbcsS3IS5ko72wj0SocjyqTP0J_Pz" alt="NVIDIA Quadro RTX 4000"></td>
    <td><img src="https://drive.google.com/uc?id=1bs6Ix7_Gy4YpO9T9Z8NWJcf3dkcIeSwq" alt="NVIDIA Geforce RTX 4070"></td>
  </tr>
</table>

#### Intel

<table>
  <tr>
    <td><img src="https://drive.google.com/uc?id=1cE7A24SxF8z5xdADocP2mfk0hvT9AciO" alt="Intel(R) UHD Graphics 770 (ADL-S GT1)"></td>
    <td><img src="https://drive.google.com/uc?id=13z2KrmZptoxKzDHzWhstxnMBGN363hl2" alt="Intel(R) Arc(tm) A770 Graphics (DG2)"></td>
  </tr>
</table>

### Performance by Driver (Contiguous Access)

#### Proprietary vs Mesa Driver

<table>
  <tr>
    <td><img src="https://drive.google.com/uc?id=1GJplb3yD6r2xOqtu83WvKWaVE3mDk66f" alt="AMD Radeon RX 7900 XT (RADV)"></td>
    <td><img src="https://drive.google.com/uc?id=1cHxZyHhficFStVDr4rGyUuRKDz3XEycj" alt="AMD Radeon RX 7900 XT"></td>
  </tr>
</table>

### Performance via Sensitivity (Cross-warp vs. Branched, Random Access)

#### AMD 

<table>
  <tr>
    <td><img src="https://drive.google.com/uc?id=1rT417dxaq_eRBAUC0VLYXgsW0oLyQfO-" alt="AMD Radeon RX 7900 XT Cross Warp"></td>
    <td><img src="https://drive.google.com/uc?id=1Bw3ObpEMKJOkVShS0u2Cld2mx2yEouLr" alt="AMD Radeon RX 7900 XT Branched"></td>
  </tr>
</table>

#### Random Access across Vendors

<table>
  <tr>
    <td><img src="https://drive.google.com/uc?id=1jL7IhSLwST_jWv3HQmN9VcnhJrjG1e2k" alt="AMD Radeon RX 7900 XT"></td>
    <td><img src="https://drive.google.com/uc?id=1yC45igaFyDzV3mOkOPwUhHXK8Fol2plc" alt="Intel(R) Arc(tm) A770 Graphics (DG2)"></td>
  </tr>
  <tr>
    <td><img src="https://drive.google.com/uc?id=1OHxifREdFU2xm0Wv2Ywt4Vf5dAF2-Mdz" alt="NVIDIA Geforce RTX 4070"></td>
    <td><img src="https://drive.google.com/uc?id=1UjAEy7wAQfQnOv_zwEiOZhYWyMygRGLQ" alt="NVIDIA Quadro RTX 4000"></td>
  </tr>
</table>

## Future work

- Use data to create a descriptive model (insights into performance optimizations)
- Examine implementations from similar work: scope promotion for work stealing, reducing use of global locks via client-server system represented as thread blocks
- OpenMP: Implementing atomic coalescing as a compiler pass on observed atomic operations
- Classify trends/patterns into groups and hypothesize on those features
