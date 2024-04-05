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
#include <random>

#include "easyvk.h"
#include "json.hpp"

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
    for (int i = 1; i <= test_iters; i++) {
        // auto start = high_resolution_clock::now();
        // rmwProgram.run();
        // auto stop = high_resolution_clock::now();
        // auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        // auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        // auto duration = s2 - s1;
        // rate += (float(rmw_iters * workgroup_size * workgroups) / static_cast<float>(duration));
        auto kernelTime = rmwProgram.runWithDispatchTiming();
        rate += (float(rmw_iters * workgroup_size * workgroups) / float((kernelTime / (double) 1000.0))); 
    }
    rate /= float(test_iters);
    rmwProgram.teardown();
    return rate;
}

extern "C" uint32_t occupancy_discovery(easyvk::Device device, uint32_t workgroup_size, uint32_t workgroups, vector<uint32_t> spv_code, uint32_t test_iters) {
        
        int maxOccupancyBound = -1;
        for (int i = 0; i < test_iters; i++) {
            Buffer count_buf = Buffer(device, 1, sizeof(uint32_t));
            count_buf.store<uint32_t>(0, 0);
            Buffer poll_open_buf = Buffer(device, 1, sizeof(uint32_t));
            poll_open_buf.store<uint32_t>(0, 1); // Poll is initially open.
            Buffer M_buf = Buffer(device, workgroups, sizeof(uint32_t));
            Buffer now_serving_buf = Buffer(device, 1, sizeof(uint32_t));
            now_serving_buf.store<uint32_t>(0, 0);
            Buffer next_ticket_buf = Buffer(device, 1, sizeof(uint32_t));
            next_ticket_buf.store<uint32_t>(0, 0);

            vector<Buffer> kernelInputs = {            count_buf, 
                                                        poll_open_buf,
                                                        M_buf,
                                                        now_serving_buf,
                                                        next_ticket_buf};
            auto program = Program(device, spv_code, kernelInputs);
            program.setWorkgroups(workgroups);
            program.setWorkgroupSize(workgroup_size);
            program.initialize("occupancy_discovery");
            program.run();
            if ((int) count_buf.load<uint32_t>(0) > maxOccupancyBound) {
                maxOccupancyBound = count_buf.load<uint32_t>(0);
            }
            program.teardown();
            count_buf.teardown();
            poll_open_buf.teardown();
            M_buf.teardown();
            next_ticket_buf.teardown();
            now_serving_buf.teardown();
        }

        return (uint32_t) maxOccupancyBound;


}


extern "C" void run(easyvk::Device device, uint32_t workgroups, uint32_t workgroup_size, uint32_t test_iters, string thread_dist, vector<uint32_t> spv_code, string test_name) {
    benchmarkData << to_string(workgroup_size) + "," + to_string(workgroups) + ":" + device.properties.deviceName;
    char currentTest[100];
    snprintf(currentTest, 100, ", %s: %s\n", thread_dist.c_str(), test_name.c_str());
    benchmarkData << currentTest;

    // Contention/Padding Values
    list<uint32_t> test_values;
    uint32_t tmp;
    if (workgroups * workgroup_size > 1024) tmp = 1024;
    else tmp = workgroup_size * workgroups;
    if (test_name == "local_atomic_fa_relaxed") tmp = 256;
    for (uint32_t i = 1; i <= tmp; i *= 2) {
        test_values.push_back(i);  
    } 
    
    for (auto it1 = test_values.begin(); it1 != test_values.end(); ++it1) {
        int random_access_status = 0;
        for (auto it2 = test_values.begin(); it2 != test_values.end(); ++it2) {
            if (test_name == "local_atomic_fa_relaxed" && *it2 > 8) continue; 
            if (!random_access_status && thread_dist == "random_access") random_access_status = 1;
            else if (random_access_status && thread_dist == "random_access") continue;

            uint32_t contention = *it1;
            uint32_t padding = *it2;
            float observed_rate = 0.0;
            benchmarkData << "(" + to_string(contention) + ", " + to_string(padding) + ", ";
            uint global_work_size = workgroup_size * workgroups;
            const int size = ((global_work_size) * padding) / contention;
            uint32_t rmw_iters = 16;

            // thread_dist == "random_access" ? contention : size -> contention breaks on AMD Discrete
            Buffer resultBuf = Buffer(device, thread_dist == "random_access" ? contention : size, sizeof(uint32_t));
            Buffer contentionBuf = Buffer(device, 1, sizeof(uint32_t));
            Buffer paddingBuf = Buffer(device, 1, sizeof(uint32_t));
            Buffer rmwItersBuf = Buffer(device, 1, sizeof(uint32_t));
            Buffer stratBuf = Buffer(device, global_work_size, sizeof(uint32_t)); 
            Buffer branchBuf = Buffer(device, global_work_size, sizeof(uint32_t)); 
            Buffer localBuf = Buffer(device, 1, sizeof(uint32_t));

            Buffer outBuf = Buffer(device, global_work_size, sizeof(uint32_t)); 
            random_device rd;
            mt19937 gen(rd()); 
            uniform_int_distribution<> distribution(0, contention-1);
            for (int i = 0; i < global_work_size; i += 1) {
                if (thread_dist == "branched") {    
                    branchBuf.store<uint32_t>(i, i % 2);
                    stratBuf.store<uint32_t>(i, (i / contention) * padding);
                } else if (thread_dist == "cross_warp") {
                    stratBuf.store<uint32_t>(i, (i * padding) % size);
                } else if (thread_dist == "contiguous_access") {
                    stratBuf.store<uint32_t>(i, (i / contention) * padding);
                } else if (thread_dist == "random_access") {
                    stratBuf.store<uint32_t>(i, distribution(gen));
                }
            }
            contentionBuf.store(0, contention);
            paddingBuf.store(0, padding);
            localBuf.store(0, (workgroup_size*padding)/contention);
            while(true) {
                rmwItersBuf.store(0, rmw_iters);
                resultBuf.clear();
                vector<Buffer> buffers = {resultBuf, rmwItersBuf, stratBuf};
                if (thread_dist == "branched") buffers.emplace_back(branchBuf);
                else if (thread_dist == "random_access") buffers.emplace_back(contentionBuf);
                if (test_name == "atomic_fa_relaxed_out") buffers.emplace_back(outBuf);
                else if (test_name == "local_atomic_fa_relaxed" && thread_dist == "cross_warp") {
                    buffers.emplace_back(paddingBuf);
                    buffers.emplace_back(localBuf);
                }
                observed_rate = rmw_benchmark(device, workgroups, workgroup_size, rmw_iters, test_iters, spv_code, buffers);
                if (isinf(observed_rate)) rmw_iters *= 2;
                else break;
            }
            resultBuf.teardown();
            contentionBuf.teardown();
            rmwItersBuf.teardown();
            branchBuf.teardown();
            paddingBuf.teardown();
            localBuf.teardown();
            stratBuf.teardown();
            outBuf.teardown();
            benchmarkData << to_string(observed_rate) + ")" << endl;
        }
    }
    return;
}

extern "C" void run_rmw_tests(easyvk::Device device) {  
    uint32_t test_iters = 64;
    uint32_t workgroup_size = device.properties.limits.maxComputeWorkGroupInvocations;
    uint32_t workgroups = occupancy_discovery(device, workgroup_size, 256, getSPVCode("occupancy_discovery.cinit"), 16);
    // We desperately need a better way to handle this - command flags?? interactive menu?? please don't make me recompile to change test type
    vector<string> thread_dist = {
        //"branched",
        //"cross_warp",
        "contiguous_access",

        // Only setup with atomic_fetch_add at the moment
        //"random_access"
    };
    vector<string> atomic_rmws = {
        "atomic_fa_relaxed",
        //"atomic_fa_relaxed_out",
        //"local_atomic_fa_relaxed",
        //"atomic_cas_succeed_store",
        //"atomic_cas_succeed_no_store",

        // Only setup with contiguous access at the moment
        //"atomic_fetch_min",
        //"atomic_fetch_max",
        //"atomic_exchange",
    };

    for (const string& strategy : thread_dist) {
        for (const string& rmw : atomic_rmws) {
            vector<uint32_t> spv_code = getSPVCode(strategy + "/" + rmw + ".cinit");
            if (rmw == "local_atomic_fa_relaxed") run(device, workgroups, 256, test_iters, strategy, spv_code, rmw);
            else run(device, workgroups, workgroup_size, test_iters, strategy, spv_code, rmw);
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


        auto device = easyvk::Device(instance, physicalDevices.at(i));

        run_rmw_tests(device);
        device.teardown();

    }
    
    benchmarkData.close();

    instance.teardown();
    return 0;
}
