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
#define USE_VALIDATION_LAYERS false
#define APPNAME "GPURmwTests"
#else
#define USE_VALIDATION_LAYERS true
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
    vector<float> rates;
    for (int i = 1; i <= test_iters; i++) {
        auto start = high_resolution_clock::now();
        rmwProgram.run();
        auto stop = high_resolution_clock::now();
        auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        auto duration = s2 - s1;
        rate += (float(rmw_iters) / static_cast<float>(duration));
        rates.push_back(float(rmw_iters) / static_cast<float>(duration));
    }
    rate /= float(test_iters);
    rmwProgram.teardown();
    return rate;
    // if (isinf(rate)) {
    //     return rate;
    // }
    // // Calculate the variance of the rates
    // float variance = 0.0;
    // for (float r : rates) {
    //     variance += pow(r - rate, 2);
    // }
    // variance /= float(test_iters);
    // //rate /= float(test_iters);
    // rmwProgram.teardown();
    // //return rate;
    // return variance;
}

extern "C" float striding_performance_model(float peak_throughput, uint32_t X, uint32_t C, uint32_t P) {
    if ((C <= X) && P == 1) {
        return peak_throughput;
    } 
    if ((C >= (P * X)) && (P >= 1 && P <= X/4)) {
        return peak_throughput * static_cast<float>(X) / static_cast<float>(C);
    } 
    else if (((X * P) > C) && (P > 1 && P <= X/4)) {
        return peak_throughput / static_cast<float>(P);
    }
    else if (P > X/4) {
        return peak_throughput / 20.0;
    }
    return 0.0;
}

extern "C" void run(easyvk::Device device, uint32_t workgroups, uint32_t workgroup_size, uint32_t test_iters, uint32_t thread_dist, vector<uint32_t> spv_code, string test_name) {

    string folder;
    if (thread_dist) folder = "chunking";
    else folder = "striding";
    benchmarkData << to_string(workgroup_size) + "," + to_string(workgroups) + ":" + device.properties.deviceName;
    char currentTest[100];
    sprintf(currentTest, ", %s: %s\n", folder.c_str(), test_name.c_str());
    benchmarkData << currentTest;

    // Contention/Padding Values
    list<uint32_t> test_values;

    uint32_t tmp;
    if (workgroups * workgroup_size > 1024) tmp = 1024;
    else tmp = workgroup_size * workgroups;
    for (uint32_t i = 1; i <= tmp; i *= 2) { //1024 for global, 64 local
        test_values.push_back(i);  
    } 

    // Find Peak
    // float peak_throughput = 0.0;
    // for (uint32_t c = 1; c <= 64; c *= 2) { 
    //     uint32_t contention = c;
    //     uint32_t padding = 1;
    //     const int size = ((workgroup_size * workgroups) * padding) / contention;
    //     uint32_t rmw_iters = 64;
    //     float curr_throughput = 0.0;
    //     Buffer resultBuf = Buffer(device, size);
    //     Buffer sizeBuf = Buffer(device, 1);
    //     Buffer paddingBuf = Buffer(device, 1);
    //     Buffer rmwItersBuf = Buffer(device, 1);
    //     Buffer contentionBuf = Buffer(device, 1);
    //     sizeBuf.store(0, size);
    //     paddingBuf.store(0, padding);
    //     contentionBuf.store(0, contention);
    //     while(true) {
    //         rmwItersBuf.store(0, rmw_iters);
    //         resultBuf.clear();
    //         vector<Buffer> buffers = {resultBuf, rmwItersBuf, paddingBuf};
    //         if (thread_dist) { // Chunking
    //             buffers.emplace_back(contentionBuf);
    //             curr_throughput = rmw_benchmark(device, workgroups, workgroup_size, rmw_iters, test_iters, spv_code, buffers);
    //         } else { // Striding
    //             buffers.emplace_back(sizeBuf);
    //             curr_throughput = rmw_benchmark(device, workgroups, workgroup_size, rmw_iters, test_iters, spv_code, buffers);
    //             //buffers.emplace_back(contentionBuf); //local striding
    //         }
    //         if (isinf(curr_throughput)) rmw_iters *= 2;
    //         else break;
    //     }
    //     if (curr_throughput > peak_throughput) peak_throughput = curr_throughput;
    //     resultBuf.teardown();
    //     sizeBuf.teardown();
    //     paddingBuf.teardown();
    //     rmwItersBuf.teardown();
    //     contentionBuf.teardown();
    // }
    int errors = 0;
    for (auto it1 = test_values.begin(); it1 != test_values.end(); ++it1) {
        for (auto it2 = test_values.begin(); it2 != test_values.end(); ++it2) {
            uint32_t contention = *it1;
            uint32_t padding = *it2;
            float expected_rate, observed_rate = 0.0;
            benchmarkData << "(" + to_string(contention) + ", " + to_string(padding) + ", ";
            //expected_rate = striding_performance_model(peak_throughput, (workgroups * workgroup_size) / device.properties.limits.maxComputeWorkGroupInvocations, contention, padding);
            const int size = ((workgroup_size * workgroups) * padding) / contention;
            uint32_t rmw_iters = 8;
            Buffer resultBuf = Buffer(device, size, sizeof(uint32_t));
            Buffer sizeBuf = Buffer(device, 1, sizeof(uint32_t));
            Buffer paddingBuf = Buffer(device, 1, sizeof(uint32_t));
            Buffer rmwItersBuf = Buffer(device, 1, sizeof(uint32_t));
            Buffer contentionBuf = Buffer(device, 1, sizeof(uint32_t));
            Buffer indexBuf = Buffer(device, workgroup_size * workgroups, sizeof(uint32_t)); // y
            if (thread_dist) { //chunking
                for (int i = 0; i < workgroup_size * workgroups; i += 1) {
                    indexBuf.store<uint32_t>(i, (i / contention) * padding);
                }
            } else { //striding
                for (int i = 0; i < workgroup_size * workgroups; i += 1) {
                    indexBuf.store<uint32_t>(i, i * padding % size);
                }
            }
            sizeBuf.store(0, size);
            paddingBuf.store(0, padding);
            contentionBuf.store(0, contention);
            while(true) {
                rmwItersBuf.store(0, rmw_iters);
                resultBuf.clear();
                vector<Buffer> buffers = {resultBuf, rmwItersBuf, indexBuf};
                if (thread_dist) { // Chunking
                    //buffers.emplace_back(contentionBuf);
                } else { // Striding
                    //buffers.emplace_back(sizeBuf);
                    if (test_name == "atomic_fa_relaxed_local" || test_name == "atomic_fa_local") {
                        buffers.emplace_back(contentionBuf); //local striding
                    }
                }
                observed_rate = rmw_benchmark(device, workgroups, workgroup_size, rmw_iters, test_iters, spv_code, buffers);
                if (isinf(observed_rate)) rmw_iters *= 2;
                else break;
            }
            // Buffer Validation
            for (int access = 0; access < size; access += padding) {
                if (resultBuf.load<uint32_t>(access) != rmw_iters * test_iters * contention) errors += 1;
            }
            resultBuf.teardown();
            sizeBuf.teardown();
            paddingBuf.teardown();
            rmwItersBuf.teardown();
            contentionBuf.teardown();
            indexBuf.teardown();
            // if (expected_rate > observed_rate) {
            //     benchmarkData << to_string(expected_rate/observed_rate) + ")" << endl;
            // } else {
            //     benchmarkData << to_string(observed_rate/expected_rate) + ")" << endl;
            // }
            benchmarkData << to_string(observed_rate) + ")" << endl;
        }
    }
    log("Error count: %d\n", errors);
    return;
}

extern "C" void run_rmw_tests(easyvk::Device device) {
    uint32_t test_iters = 64;

    uint32_t maxComputeWorkGroupCount = device.properties.limits.maxComputeWorkGroupCount[0];
    if (maxComputeWorkGroupCount > 65536) maxComputeWorkGroupCount = 65536;

    uint32_t workgroup_size = device.properties.limits.maxComputeWorkGroupInvocations;
    if (workgroup_size > 1024) workgroup_size = 1024;

    double quotient = static_cast<double>(maxComputeWorkGroupCount) / workgroup_size;

    uint32_t workgroups = static_cast<uint32_t>(ceil(quotient));


    vector<string> thread_dist = {"striding", "chunking"};
    vector<string> atomic_rmws = {
        "atomic_cas_succeed_store", "atomic_cas_succeed_no_store", 
        "atomic_cas_fail_no_store", "atomic_ex_relaxed", 
        "atomic_fa_relaxed", "atomic_ex", "atomic_fa", 
        "atomic_fa_local", "atomic_fa_relaxed_local"
    };

    for (const string& strategy : thread_dist) {
        for (const string& rmw : atomic_rmws) {
            int isChunking = (strategy == "chunking") ? 1 : 0;
            if (rmw == "atomic_fa_relaxed_local" || rmw == "atomic_fa_local") {
                //run(device, 1, 64, test_iters, isChunking, getSPVCode(strategy + "/" + rmw + ".cinit"), rmw);
                //run(device, 64, 64, test_iters, isChunking, getSPVCode(strategy + "/" + rmw + ".cinit"), rmw);
            } else {
                run(device, workgroups, workgroup_size, test_iters, isChunking, getSPVCode(strategy + "/" + rmw + ".cinit"), rmw);
            }
        }
    }
    return;
}

int main() {
    benchmarkData.open("result.txt"); 

    if (!benchmarkData.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return 1;
    }
    auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);
	auto physicalDevices = instance.physicalDevices();

    for (size_t i = 0; i < physicalDevices.size(); i++) {

        if (i != 0) continue;
        auto device = easyvk::Device(instance, physicalDevices.at(i));

        run_rmw_tests(device);
        device.teardown();

    }
    
    benchmarkData.close();

    instance.teardown();
    #ifdef __ANDROID__
    return 0;
    #else
    system("python3 heatmap.py");
    return 0;
    #endif

}
