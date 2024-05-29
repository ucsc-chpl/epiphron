# Atomic-RMW: GPU Microbenchmark 

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Benchmarks](#running-benchmarks)

## Introduction
A parameterized microbenchmark suite for characterizing performance profiles of RMWs on GPUs, written in two GPU frameworks, Vulkan and CUDA. The goal of these microbenchmarks is to characterize GPU atomic RMW performance results, towards the direction of providing application programmers with the tools to reason about their performance. We want to provide insights on reducing the cost of these operations through reordering or rewriting atomics in a way that provide performance optimizations.

## Prerequisites
Before you begin, ensure you have setup the following requirements:

### Frameworks and Tools
- **Vulkan:**
  - [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
  - [clspv](https://github.com/google/clspv)
  - [Android ADB](https://developer.android.com/studio/command-line/adb) / [NDK](https://developer.android.com/ndk)
  - [Python](https://www.python.org/)
  - `make` utility for building

- **CUDA:**
  - CUDA compilation tools (e.g., nvcc, CUDA Toolkit)
  - [Python](https://www.python.org/)
  - `make` utility for building

## Installation
To install the necessary dependencies, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/username/repo.githttps://github.com/ucsc-chpl/epiphron.git
    ```
    
2. Build easyvk:
    ```bash
    cd epiphron/easyvk/
    git submodule update --init --recursive
    make
    ```

## Running Benchmarks
To run the benchmarks, execute the following series of commands:

### Vulkan

```bash
cd epiphron/benchmarks/atomic-rmw/
mkdir results
make
./atomic_rmw_test
python3 heatmap.py
python3 random_access.py #if tested
```

### Vulkan (Android)
```bash
cd epiphron/benchmarks/atomic-rmw/
adb devices # get serial number
adb -s [SERIAL_NUMBER] shell getprop ro.product.cpu.abilist # get supported ABI, use ro.product.cpu.abi if pre-lollipop version
mkdir results
make android
adb -s [SERIAL_NUMBER] push build/android/obj/local/[SUPPORTED_CPU]/ /data/local/tmp/rmw
adb -s [SERIAL_NUMBER] shell
  cd data/local/tmp/rmw/[SUPPORTED_CPU]
  ./rmw_benchmark
  exit
adb -s [SERIAL_NUMBER] pull /data/local/tmp/rmw/[SUPPORTED_CPU]/results/ .
python3 heatmap.py
```

### CUDA

```bash
cd epiphron/benchmarks/atomic-rmw/cuda_implementation/
make
./atomic_rmw_test
python3 ../heatmap.py
```
