# Atomic-RMW: GPU Microbenchmark 

## Introduction
A microbenchmark made to observe the behavior and measure performance of low level synchronization primitives, atomic RMWs on GPUs. An atomic RMW is an operation that does both read a memory location and write a new value into it simultaneously, all happens in one atomic indivisible step. The development of this microbenchmark will help us look for interesting hardware features, specifically on the coalescing of atomics. The act of coalescing is the idea of combining individual pushes into a single push of multiple items utilizing a single rmw. 

Furthermore, from obtaining data with this microbenchmark, we want to derive a descriptive model for a wide variety of GPUs across multiple vendors, to understand the undisclosed characteristics of GPU microarchitecture and the manufacturer's design decisions.

## Motivation
The most general forms of fine-grained synchronization are one of the most expensive operations on a GPU, like using fine-grained mutexes to provide mutual exclusion to global memory updates. Also applying GPUs to irregular computations like graph algorithms have become more common nowadays. These computations require the use of a global worklist for dynamic workload balancing, and that each modification to it will require an RMW. The main goal of this microbenchmark is to provide insights on reducing the cost of these expensive operations through reordering or rewriting atomics in a way that provide performance optimizations. 

However, very little is known about these characteristics of GPU's architecture beyond what the manufacturer has documented and there is not a simple solution for this as GPU architecture is very diverse, adding more complexity if we want to identify portable optimizations. Hence, taking this reverse-engineering approach will help us build an intuition and classification about how coalescing patterns fit into certain underlying architectures and further hypothesize about hardware features that lead to these observed shapes and trends. We also need to understand if potential coalescing of atomics could happen in the compiler or hardware level as an OpenCL compiler could exploit atomics (maybe implement a RMW-coalescing transformation for subgroups or have one rmw per thread block) or could automatically be done by the hardware. 

## OpenCL Background


## Heterogeneous System Architecture Cache Hierarchy


## Experimental Setup


### Contiguous Access


### Cross Warp


### Branched


### Random Access 


