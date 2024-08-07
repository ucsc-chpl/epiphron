# Epiphron (Atomic RMW)

## Running

We've included a Windows version built against MSVC Clang / Vulkan SDK 1.3.290.0 here, which can be run with the below:

```
cd atomic-rmw
.\atomic_rmw_test.exe
```

This will take you to a series of menus, which can be navigated by typing numbers and hitting enter to select the device, the specific test, and an atomic operation. For the test we've been discussing, the parameters should be `NVIDIA_instance_access` for the test and `atomic_fa_relaxed` for the operation. 

For the `NVIDIA_instance_access` test, we also have two extra parameters - the padding size (in KB, as we're examining the effects of high padding) and the number of threads to group together.

After entering these parameters, the tests will run, a percentage to complete will be shown, and afterwards the program will exit, and you'll be left with a results file in `results/`.

To then generate a heatmap for these results, you'll need to run `heatmap.py`, which will generate a heatmap and place it in `graphs/` as both a .svg and .png file. The random access test has its own Python script to generate that graph, `random_access.py`. The only Python modules required should be `matplotlib`, `SciencePlots`, and `numpy`.

## Building

In case that doesn't work, we use the included Makefile to build, which has a couple notable requirements:

- Some C++ compiler named by `$CXX` (in our case we use MSVC's clang on Windows)
- Vulkan SDK (with `$VULKAN_SDK` pointing to the installation)
- [clspv](https://github.com/google/clspv), which we include a prebuilt version of but can of course be built from source
- GNU Make for Windows

After that, it should be as simple as:

```
cd atomic-rmw
make
```

However, if there are any issues with this, feel free to reach out and we can try to resolve them.
