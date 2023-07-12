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

extern "C" void run(easyvk::Device device, uint32_t workgroups, uint32_t workgroup_size, uint32_t test_iters, uint32_t thread_dist, uint32_t curr_rmw) {

    string folder;
    if (thread_dist) folder = "chunking";
    else folder = "striding";
    vector<uint32_t> spv_code;
    benchmarkData << to_string(workgroup_size) + "," + to_string(workgroups) + ":" + device.properties.deviceName;
    char currentTest[100];
    switch (curr_rmw) {
        case 1:
            sprintf(currentTest, ", %s: atomic_cas_fail_no_store\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_cas_fail_no_store.cinit");
            break;
        case 2:
            sprintf(currentTest, ", %s: atomic_cas_succeed_no_store\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_cas_succeed_no_store.cinit");
            break;
        case 3:
            sprintf(currentTest, ", %s: atomic_cas_succeed_store\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_cas_succeed_store.cinit");
            break;
        case 4:
            sprintf(currentTest, ", %s: atomic_ex_relaxed\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_ex_relaxed.cinit");
            break;
        case 5:
            sprintf(currentTest, ", %s: atomic_ex\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_ex.cinit");
            break;
        case 6:
            sprintf(currentTest, ", %s: atomic_fa_relaxed\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_fa_relaxed.cinit");
            break;
        case 7:
            sprintf(currentTest, ", %s: atomic_fa\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_fa.cinit");
            break;
        default:
            return;
    }  
    benchmarkData << currentTest;

    // Contention/Padding Values
    list<uint32_t> test_values;
    for (uint32_t i = 1; i <= workgroup_size; i *= 2) {
        test_values.push_back(i);  
    } 
    for (auto it1 = test_values.begin(); it1 != test_values.end(); ++it1) {
        for (auto it2 = test_values.begin(); it2 != test_values.end(); ++it2) {
            uint32_t contention = *it1;
            uint32_t padding = *it2;
            benchmarkData << "(" + to_string(contention) + ", " + to_string(padding) + ", ";

            const int size = workgroup_size * padding / contention;
            bool isINF = true;
            uint32_t rmw_iters = 64;
            float rate = 0.0;
            Buffer resultBuf = Buffer(device, size);
            Buffer rmwItersBuf = Buffer(device, 1);
            Buffer paddingBuf = Buffer(device, 1);
            Buffer contentionBuf = Buffer(device, 1);
            Buffer sizeBuf = Buffer(device, 1);
            Buffer garbageBuf = Buffer(device, workgroups * workgroup_size);
            paddingBuf.store(0, padding);
            contentionBuf.store(0, contention);
            sizeBuf.store(0, size);
            while(isINF) {
                vector<Buffer> buffers = { resultBuf, rmwItersBuf, paddingBuf, contentionBuf, sizeBuf, garbageBuf };
                rmwItersBuf.store(0, rmw_iters);
                rate = rmw_benchmark(device, workgroups, workgroup_size, rmw_iters, test_iters, spv_code, buffers);
                if (isinf(rate)) rmw_iters *= 2;
                else isINF = false;
            }
            resultBuf.teardown();
            rmwItersBuf.teardown();
            paddingBuf.teardown();
            contentionBuf.teardown();
            sizeBuf.teardown();
            garbageBuf.teardown();
            benchmarkData << to_string(rate) + ")" << endl;
        }
    }
    return;
}

extern "C" void run_rmw_tests(easyvk::Device device) {
    uint32_t test_iters = 16;

    for (uint32_t workgroup_size = 64; workgroup_size <= device.properties.limits.maxComputeWorkGroupInvocations; workgroup_size *= 2) {

        double quotient = static_cast<double>(device.properties.limits.maxComputeWorkGroupCount[0]) / workgroup_size;
        uint32_t workgroups = static_cast<uint32_t>(ceil(quotient));

        for (uint32_t curr_rmw = 1; curr_rmw <= 7; curr_rmw++) { //1...7
            run(device, workgroups, workgroup_size, test_iters, 0, curr_rmw);
            run(device, workgroups, workgroup_size, test_iters, 1, curr_rmw);
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

    auto instance = easyvk::Instance(true);
	auto physicalDevices = instance.physicalDevices();

    for (size_t i= 0; i < physicalDevices.size(); i++) {

        if (i == 1) continue; // Skipping Nvidia
        auto device = easyvk::Device(instance, physicalDevices.at(i));

        run_rmw_tests(device);
        device.teardown();

    }
    
    benchmarkData.close();

    instance.teardown();
    system("python3 heatmap.py");
    return 0;
}