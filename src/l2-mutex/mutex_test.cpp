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
#define APPNAME "GPUMutexTests"
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

extern "C" float mutex_benchmark(easyvk::Device device, uint32_t workgroups, uint32_t workgroup_size, uint32_t mutex_iters, uint32_t test_iters, vector<uint32_t> spv_code, vector<Buffer> buffers) {
    
    Program mutexProgram = Program(device, spv_code, buffers);
    mutexProgram.setWorkgroups(workgroups);
    mutexProgram.setWorkgroupSize(workgroup_size);
    mutexProgram.initialize("mutex_test");
  
    float rate = 0.0;
    for (int i = 1; i <= test_iters; i++) {
        auto start = high_resolution_clock::now();
        mutexProgram.run();
        auto stop = high_resolution_clock::now();
        auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        auto duration = s2 - s1;
        rate += (float(mutex_iters) / static_cast<float>(duration));
    }
    rate /= float(test_iters);
    mutexProgram.teardown();
    return rate;
}



extern "C" void run(easyvk::Device device, uint32_t workgroups, uint32_t workgroup_size, uint32_t test_iters, uint32_t thread_dist) {

    string folder;
    if (thread_dist) folder = "chunking";
    else folder = "striding";
    vector<uint32_t> spv_code;
    benchmarkData << to_string(workgroup_size) + "," + to_string(workgroups) + ":" + device.properties.deviceName;
    char currentTest[100];
    sprintf(currentTest, ", %s: CAS_lock\n", folder.c_str());
    spv_code = getSPVCode(folder + "/cas_lock.cinit");
    benchmarkData << currentTest;

    // Contention/Padding Values
    list<uint32_t> test_values;


    for (uint32_t i = 1; i <= workgroup_size; i *= 2) {
        test_values.push_back(i);  
    } 

    int errors = 0;
    for (auto it1 = test_values.begin(); it1 != test_values.end(); ++it1) {
        for (auto it2 = test_values.begin(); it2 != test_values.end(); ++it2) {
            uint32_t contention = *it1;
            uint32_t padding = *it2;
            float observed_rate = 0.0;
            benchmarkData << "(" + to_string(contention) + ", " + to_string(padding) + ", ";
            const int size = ((workgroup_size * workgroups) * padding) / contention;
            uint32_t mutex_iters = 16;
            Buffer lockBuf = Buffer(device, size);
            Buffer resultBuf = Buffer(device, size);
            Buffer sizeBuf = Buffer(device, 1);
            Buffer paddingBuf = Buffer(device, 1);
            Buffer mutexItersBuf = Buffer(device, 1);
            Buffer contentionBuf = Buffer(device, 1);
            sizeBuf.store(0, size);
            paddingBuf.store(0, padding);
            contentionBuf.store(0, contention);
            while(true) {
                mutexItersBuf.store(0, mutex_iters);
                lockBuf.clear();
                resultBuf.clear();
                vector<Buffer> buffers = {lockBuf, resultBuf, mutexItersBuf, paddingBuf};
                if (thread_dist) { // Chunking
                    buffers.emplace_back(contentionBuf);
                    observed_rate = mutex_benchmark(device, workgroups, workgroup_size, mutex_iters, test_iters, spv_code, buffers);
                } else { // Striding
                    buffers.emplace_back(sizeBuf);
                    observed_rate = mutex_benchmark(device, workgroups, workgroup_size, mutex_iters, test_iters, spv_code, buffers);
                }
                if (isinf(observed_rate)) mutex_iters *= 2;
                else break;
            }
            // Buffer Validation
            for (int access = 0; access < size; access += padding) {
                if (resultBuf.load(access) != mutex_iters * test_iters * contention) errors += 1;
            }
            lockBuf.teardown();
            resultBuf.teardown();
            sizeBuf.teardown();
            paddingBuf.teardown();
            mutexItersBuf.teardown();
            contentionBuf.teardown();
            benchmarkData << to_string(observed_rate) + ")" << endl;
        }
    }
    log("Error count: %d\n", errors);
    return;
}
/*
Implement: CAS, Ticket, CAS /w relaxed peeking
*/

extern "C" void run_mutex_tests(easyvk::Device device) {
    uint32_t test_iters = 64;

    uint32_t maxComputeWorkGroupCount = device.properties.limits.maxComputeWorkGroupCount[0];
    if (maxComputeWorkGroupCount > 65536) maxComputeWorkGroupCount = 65536;

    uint32_t workgroup_size = device.properties.limits.maxComputeWorkGroupInvocations;
    if (workgroup_size > 1024) workgroup_size = 1024;

    double quotient = static_cast<double>(maxComputeWorkGroupCount) / workgroup_size;

    uint32_t workgroups = static_cast<uint32_t>(ceil(quotient));

    // Striding
    run(device, workgroups, workgroup_size, test_iters, 0);

    // Chunking
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

        if (i != 1) continue;
        auto device = easyvk::Device(instance, physicalDevices.at(i));

        run_mutex_tests(device);
        device.teardown();

    }
    
    benchmarkData.close();

    instance.teardown();
    //system("python3 heatmap.py");
    return 0;
}