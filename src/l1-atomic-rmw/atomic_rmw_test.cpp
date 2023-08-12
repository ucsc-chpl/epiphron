#include <stdexcept>
#include <stdarg.h>
#include <string>
#include <chrono>
#include <list>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include "easyvk.h"
#include "../_example/json.h"

#ifdef __ANDROID__
#include <android/log.h>
#define APPNAME "GPURmwTests"
#endif

using namespace std;
using nlohmann::json;
using easyvk::Instance;
using easyvk::Device;
using easyvk::Buffer;
using easyvk::Program;
using easyvk::vkDeviceType;
using namespace chrono;

const char* os_name() {
    #ifdef _WIN32
    return "Windows (32-bit)";
    #elif _WIN64
    return "Windows (64-bit)";
    #elif __APPLE__
        #include <TargetConditionals.h>
        #if TARGET_IPHONE_SIMULATOR
        return "iPhone (Simulator)";
        #elif TARGET_OS_MACCATALYST
        return "macOS Catalyst";
        #elif TARGET_OS_IPHONE
        return "iPhone";
        #elif TARGET_OS_MAC
        return "macOS";
        #else
        return "Other (Apple)";
        #endif
    #elif __ANDROID__
    return "Android";
    #elif __linux__
    return "Linux";
    #elif __unix || __unix||
    return "Unix";
    #else
    return "Other";
    #endif
}

ofstream benchmarkData; 

void log(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    #ifdef __ANDROID__
    __android_log_vprint(ANDROID_LOG_INFO, APPNAME, fmt, args);
    #else
    vprintf(fmt, args);
    #endif
    va_end(args);
}

vector<uint32_t> getSPVCode(const string& filename) {
    ifstream file(filename);
    vector<uint32_t> spv_code;
    char ch;

    while (file >> ch) {
        if (isdigit(ch)) {
            file.unget();
            uint32_t value;
            file >> value;
            spv_code.push_back(value);
        }
    }

    file.close();
    return spv_code;
}

extern "C" float rmw_benchmark(easyvk::Device device, uint32_t workgroups, uint32_t workgroup_size, uint32_t rmw_iters, uint32_t test_iters, vector<uint32_t> spv_code, vector<Buffer> buffers) {
    
    Program rmwProgram = Program(device, spv_code, buffers);
    rmwProgram.setWorkgroups(workgroups);
    rmwProgram.setWorkgroupSize(workgroup_size);
    rmwProgram.initialize("rmw_test");
  
    float rate = 0.0;
    for (int i = 1; i <= test_iters; i++) {
        auto start = high_resolution_clock::now();
        rmwProgram.run();
        auto stop = high_resolution_clock::now();
        auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        auto duration = s2 - s1;
        rate += (float(rmw_iters) / static_cast<float>(duration));
    }
    rate /= float(test_iters);
    rmwProgram.teardown();
    return rate;
}

extern "C" void run(easyvk::Device device, uint32_t workgroups, uint32_t workgroup_size, uint32_t test_iters, uint32_t thread_dist) {

    string folder;
    if (thread_dist) folder = "chunking";
    else folder = "striding";
    vector<uint32_t> spv_code_relaxed, spv_code_acq_rel;
    benchmarkData << to_string(workgroup_size) + "," + to_string(workgroups) + ":" + device.properties.deviceName;
    char currentTest[100];
    sprintf(currentTest, ", %s: global_fetch_add_relaxed\n", folder.c_str());
    spv_code_relaxed = getSPVCode(folder + "/atomic_fa_relaxed.cinit");
    spv_code_acq_rel = getSPVCode(folder + "/atomic_fa_relaxed.cinit");
    benchmarkData << currentTest;

    // Contention/Padding Values
    list<uint32_t> test_values;
    //For only testing global memory 
    uint32_t max_pair;
    if (workgroup_size * workgroups > 32768) max_pair = 32768;
    else max_pair = workgroup_size * workgroups;

    for (uint32_t i = 1; i <= max_pair; i *= 2) {
        test_values.push_back(i);  
    } 
    for (auto it1 = test_values.begin(); it1 != test_values.end(); ++it1) {
        for (auto it2 = test_values.begin(); it2 != test_values.end(); ++it2) {
            uint32_t contention = *it1;
            uint32_t padding = *it2;
            benchmarkData << "(" + to_string(contention) + ", " + to_string(padding) + ", ";

            const int size = ((workgroup_size * workgroups) * padding) / contention;
            uint32_t rmw_iters = 64;
            float rate_relaxed = 0.0, rate_acq_rel = 0.0;
            Buffer resultBuf = Buffer(device, size);
            Buffer sizeBuf = Buffer(device, 1);
            Buffer paddingBuf = Buffer(device, 1);
            Buffer rmwItersBuf = Buffer(device, 1);
            Buffer contentionBuf = Buffer(device, 1);
            sizeBuf.store(0, size);
            paddingBuf.store(0, padding);
            contentionBuf.store(0, contention);
            while(true) {
                rmwItersBuf.store(0, rmw_iters);
                vector<Buffer> buffers = {resultBuf, rmwItersBuf, paddingBuf};
                if (thread_dist) { // Chunking
                    buffers.emplace_back(contentionBuf);
                    //rate_acq_rel = rmw_benchmark(device, workgroups, workgroup_size, rmw_iters, test_iters, spv_code_acq_rel, buffers);
                    rate_relaxed = rmw_benchmark(device, workgroups, workgroup_size, rmw_iters, test_iters, spv_code_relaxed, buffers);
                } else { // Striding
                    buffers.emplace_back(sizeBuf);
                    //rate_acq_rel = rmw_benchmark(device, workgroups, workgroup_size, rmw_iters, test_iters, spv_code_acq_rel, buffers);
                    buffers.emplace_back(contentionBuf);
                    rate_relaxed = rmw_benchmark(device, workgroups, workgroup_size, rmw_iters, test_iters, spv_code_relaxed, buffers);
                }
                if (isinf(rate_relaxed) || isinf(rate_acq_rel)) rmw_iters *= 2;
                else break;
            }
            resultBuf.teardown();
            sizeBuf.teardown();
            paddingBuf.teardown();
            rmwItersBuf.teardown();
            contentionBuf.teardown();
            benchmarkData << to_string(rate_relaxed) + ")" << endl;
        }
    }
    return;
}

extern "C" void run_rmw_tests(easyvk::Device device) {
    uint32_t test_iters = 64;

    uint32_t maxComputeWorkGroupCount = device.properties.limits.maxComputeWorkGroupCount[0];
    if (maxComputeWorkGroupCount > 65536) maxComputeWorkGroupCount = 65536;

    uint32_t workgroup_size = device.properties.limits.maxComputeWorkGroupInvocations;
    double quotient = static_cast<double>(maxComputeWorkGroupCount) / workgroup_size;

    uint32_t workgroups = static_cast<uint32_t>(ceil(quotient));

    // workgroup_size = 64;
    // workgroups = 1;

    // run(device, workgroups, workgroup_size, test_iters, 0);
    // run(device, workgroups, workgroup_size, test_iters, 1);

    // workgroup_size = 64;
    // workgroups = 64;

    run(device, workgroups, workgroup_size, test_iters, 0);
    run(device, workgroups, workgroup_size, test_iters, 1);
    return;
}

int main() {
    benchmarkData.open("result.txt"); 

    if (!benchmarkData.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return 1;
    }

    auto instance = easyvk::Instance(true);
	auto physicalDevices = instance.physicalDevices();

    for (size_t i = 0; i < physicalDevices.size(); i++) {

        if (i == 1) continue;
        auto device = easyvk::Device(instance, physicalDevices.at(i));

        run_rmw_tests(device);
        device.teardown();

    }
    
    benchmarkData.close();

    instance.teardown();
    system("python3 heatmap.py");
    return 0;
}