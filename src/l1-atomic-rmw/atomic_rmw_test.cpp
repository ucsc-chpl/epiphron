#include <stdexcept>
#include <stdarg.h>
#include <string>
#include <chrono>
#include <list>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdlib>

#include "easyvk.h"
#include "../_example/json.h"

#ifdef __ANDROID__
#include <android/log.h>
#define APPNAME "GPULockTests"
#endif

using std::ifstream;
using std::list;
using std::vector;
using std::runtime_error;
using std::string;
using std::copy;
using nlohmann::json;
using easyvk::Instance;
using easyvk::Device;
using easyvk::Buffer;
using easyvk::Program;
using easyvk::vkDeviceType;
using namespace std::chrono;

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

extern "C" void rmw_benchmark(easyvk::Device device, uint32_t workgroups, uint32_t workgroup_size, uint32_t rmw_iters, uint32_t test_iters, vector<uint32_t> spv_code, vector<Buffer> buffers) {
    
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
    log("%f", rate);
    rmwProgram.teardown();
    return;
}

extern "C" void run(easyvk::Device device, uint32_t workgroups, uint32_t workgroup_size, uint32_t rmw_iters, uint32_t test_iters, uint32_t thread_dist, uint32_t curr_rmw) {

    uint32_t test_total = workgroups * rmw_iters;
    uint32_t total_rmws = test_total * test_iters;

    string folder;
    if (thread_dist) folder = "chunking";
    else folder = "striding";
    vector<uint32_t> spv_code;
    log(device.properties.deviceName);
    switch (curr_rmw) {
        case 1:
            log(", %s: atomic_cas_fail_no_store\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_cas_fail_no_store.cinit");
            break;
        case 2:
            log(", %s: atomic_cas_succeed_no_store\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_cas_succeed_no_store.cinit");
            break;
        case 3:
            log(", %s: atomic_cas_succeed_store\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_cas_succeed_store.cinit");
            break;
        case 4:
            log(", %s: atomic_ex_relaxed\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_ex_relaxed.cinit");
            break;
        case 5:
            log(", %s: atomic_ex\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_ex.cinit");
            break;
        case 6:
            log(", %s: atomic_fa_relaxed\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_fa_relaxed.cinit");
            break;
        case 7:
            log(", %s: atomic_fa\n", folder.c_str());
            spv_code = getSPVCode(folder + "/atomic_fa.cinit");
            break;
        default:
            return;
    }  

    // Contention/Padding Values
    list<uint32_t> test_values = {1, 2, 4, 8, 16, 32, 64, 128}; 
    for (auto it1 = test_values.begin(); it1 != test_values.end(); ++it1) {
        for (auto it2 = test_values.begin(); it2 != test_values.end(); ++it2) {
            uint32_t contention = *it1;
            uint32_t padding = *it2;
            log("(%d, %d, ", contention, padding);

            const int size = workgroup_size * padding / contention;
            if (size < 1) {
                log("0.0)\n");
                continue;
            }
    
            Buffer resultBuf = Buffer(device, size);
            Buffer rmwItersBuf = Buffer(device, 1);
            Buffer paddingBuf = Buffer(device, 1);
            Buffer contentionBuf = Buffer(device, 1);
            Buffer sizeBuf = Buffer(device, 1);
            Buffer garbageBuf = Buffer(device, workgroups * workgroup_size);
            vector<Buffer> buffers = { resultBuf, rmwItersBuf, paddingBuf, contentionBuf, sizeBuf, garbageBuf };
            rmwItersBuf.store(0, rmw_iters);
            paddingBuf.store(0, padding);
            contentionBuf.store(0, contention);
            sizeBuf.store(0, size);

            rmw_benchmark(device, workgroups, workgroup_size, rmw_iters, test_iters, spv_code, buffers);

            log(")\n");

            resultBuf.teardown();
            rmwItersBuf.teardown();
            paddingBuf.teardown();
            contentionBuf.teardown();
            sizeBuf.teardown();
            garbageBuf.teardown();
        }
    }
    return;
}

extern "C" void run_rmw_tests(easyvk::Device device) {
    uint32_t rmw_iters = 65536;
    uint32_t test_iters = 64;
    // uint32_t maxComputeWorkGroupInvocations = device.properties.limits.maxComputeWorkGroupInvocations;
    uint32_t workgroups = 256;
    //uint32_t workgroups = device.properties.limits.maxComputeWorkGroupCount[0]
    uint32_t workgroup_size = 256;
    // uint32_t workgroup_size = device.properties.limits.maxComputeWorkGroupSize[0]

    // if (workgroups > maxComputeWorkGroupInvocations) {
    //     workgroups = maxComputeWorkGroupInvocations;
    // } else {
    //     workgroups = maxComputeWorkGroupInvocations / workgroup_size;
    // }

    for (uint32_t curr_rmw = 1; curr_rmw <= 7; curr_rmw++) {
        run(device, workgroups, workgroup_size, rmw_iters, test_iters, 0, curr_rmw);
        run(device, workgroups, workgroup_size, rmw_iters, test_iters, 1, curr_rmw);
    }

    return;
}

int main() {
    std::ofstream outputFile("result.txt");

    if (!outputFile.is_open()) {
        std::cerr << "Failed to open the output file." << std::endl;
        return 1;
    }

    std::streambuf* coutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(outputFile.rdbuf());

    auto instance = easyvk::Instance(true);
	auto physicalDevices = instance.physicalDevices();

    for (size_t i= 0; i < physicalDevices.size(); i++) {

        if (i != 1) continue;
        auto device = easyvk::Device(instance, physicalDevices.at(i));

        run_rmw_tests(device);
        device.teardown();

    }
    std::cout.rdbuf(coutBuffer);
    
    outputFile.close();

    instance.teardown();
    system("python3 heatmap.py");
    return 0;
}