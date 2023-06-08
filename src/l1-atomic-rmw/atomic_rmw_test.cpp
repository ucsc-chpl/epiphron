#include <stdexcept>
#include <stdarg.h>
#include <string>
#include <chrono>
#include <list>
#include <iostream>

#include "easyvk.h"
#include "../_example/json.h"

#ifdef __ANDROID__
#include <android/log.h>
#define APPNAME "GPULockTests"
#endif

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

extern "C" char* run(uint32_t workgroups, uint32_t workgroup_size, uint32_t padding, uint32_t contention, uint32_t rmw_iters, uint32_t test_iters) {
    //log("Initializing test...\n");

    auto instance = easyvk::Instance(true);
	auto physicalDevices = instance.physicalDevices();
	auto device = easyvk::Device(instance, physicalDevices.at(0));

    //log("Using device '%s'\n", device.properties.deviceName);

    uint32_t maxComputeWorkGroupInvocations = device.properties.limits.maxComputeWorkGroupInvocations;
    // workgroups = maxComputeWorkGroupInvocations/32
    log("MaxComputeWorkGroupInvocations: %d\n", maxComputeWorkGroupInvocations);
    if (workgroups > maxComputeWorkGroupInvocations)
        workgroups = maxComputeWorkGroupInvocations;

    uint32_t test_total = workgroups * rmw_iters;
    uint32_t total_rmws = test_total * test_iters;

    const int size = workgroup_size * padding / contention;
    if (size < 1) {
        return {};
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

    // add chunking v striding check to modify string, temp setup 

    //log("%d workgroups\n%d threads per workgroup\n%d rmw accesses per thread\ntests run %d times\ncontention at %d\npadding at %d\n", 
    //workgroups, workgroup_size, rmw_iters, test_iters, contention, padding);

    // -------------- ATOMIC CAS SUCCEED STORE --------------

    log("----------------------------------------------------------\n");
    log("Testing Atomic Compare and Swap SUCCEED-STORE...\n");
    vector<uint32_t> casSpvCode =
    #include "chunking/atomic_cas_succeed_store.cinit"
    ;

    Program casProgram = Program(device, casSpvCode, buffers);
    casProgram.setWorkgroups(workgroups);
    casProgram.setWorkgroupSize(workgroup_size);
    casProgram.initialize("rmw_test");
  
    float cas_rate = 0.0;
    for (int i = 1; i <= test_iters; i++) {
        resultBuf.clear();
        auto start = high_resolution_clock::now();
        casProgram.run();
        auto stop = high_resolution_clock::now();
        auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        auto duration = s2 - s1;
        cas_rate += (float(rmw_iters) / static_cast<float>(duration));
    }
    cas_rate /= float(test_iters);
    log("\nAvg rate of operations per ms over %d tests: %f\n", cas_rate, test_iters);
    log("%f\n", cas_rate);

    // -------------- ATOMIC CAS SUCCEED NO STORE --------------
    
    log("----------------------------------------------------------\n");
    log("Testing Atomic Compare and Swap SUCCEED-NO-STORE...\n");
    vector<uint32_t> cas2SpvCode =
    #include "chunking/atomic_cas_succeed_no_store.cinit"
    ;

    Program cas2Program = Program(device, cas2SpvCode, buffers);
    cas2Program.setWorkgroups(workgroups);
    cas2Program.setWorkgroupSize(workgroup_size);
    cas2Program.initialize("rmw_test");
  
    float cas2_rate = 0.0;
    for (int i = 1; i <= test_iters; i++) {
        resultBuf.clear();
        auto start = high_resolution_clock::now();
        cas2Program.run();
        auto stop = high_resolution_clock::now();
        auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        auto duration = s2 - s1;
        cas2_rate += (float(rmw_iters) / static_cast<float>(duration));
    }
    cas2_rate /= float(test_iters);
    log("\nAvg rate of operations per ms over %d tests: %f\n", cas2_rate, test_iters);

    // -------------- ATOMIC CAS FAIL NO STORE --------------

    log("----------------------------------------------------------\n");
    log("Testing Atomic Compare and Swap FAIL-NO-STORE...\n");
    vector<uint32_t> cas3SpvCode =
    #include "chunking/atomic_cas_fail_no_store.cinit"
    ;

    Program cas3Program = Program(device, cas3SpvCode, buffers);
    cas3Program.setWorkgroups(workgroups);
    cas3Program.setWorkgroupSize(workgroup_size);
    cas3Program.initialize("rmw_test");
  
    float cas3_rate = 0.0;
    for (int i = 1; i <= test_iters; i++) {
        resultBuf.clear();
        auto start = high_resolution_clock::now();
        cas3Program.run();
        auto stop = high_resolution_clock::now();
        auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        auto duration = s2 - s1;
        cas3_rate += (float(rmw_iters) / static_cast<float>(duration));
    }
    cas3_rate /= float(test_iters);
    log("\nAvg rate of operations per ms over %d tests: %f\n", cas3_rate, test_iters);

    // -------------- ATOMIC EX --------------

    log("----------------------------------------------------------\n");
    log("Testing Atomic Exchange...\n");
    vector<uint32_t> exSpvCode =
    #include "chunking/atomic_ex.cinit"
    ;

    Program exProgram = Program(device, exSpvCode, buffers);
    exProgram.setWorkgroups(workgroups);
    exProgram.setWorkgroupSize(workgroup_size);
    exProgram.initialize("rmw_test");
  
    float ex_rate = 0.0;
    for (int i = 1; i <= test_iters; i++) {
        resultBuf.clear();
        auto start = high_resolution_clock::now();
        exProgram.run();
        auto stop = high_resolution_clock::now();
        auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        auto duration = s2 - s1;
        ex_rate += (float(rmw_iters) / static_cast<float>(duration));
    }
    ex_rate /= float(test_iters);
    log("\nAvg rate of operations per ms over %d tests: %f\n", ex_rate, test_iters);

    // -------------- ATOMIC EX RELAXED --------------

    log("----------------------------------------------------------\n");
    log("Testing Atomic Exchange RELAXED...\n");
    vector<uint32_t> ex2SpvCode =
    #include "chunking/atomic_ex_relaxed.cinit"
    ;

    Program ex2Program = Program(device, ex2SpvCode, buffers);
    ex2Program.setWorkgroups(workgroups);
    ex2Program.setWorkgroupSize(workgroup_size);
    ex2Program.initialize("rmw_test");
  
    float ex2_rate = 0.0;
    for (int i = 1; i <= test_iters; i++) {
        resultBuf.clear();
        auto start = high_resolution_clock::now();
        ex2Program.run();
        auto stop = high_resolution_clock::now();
        auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        auto duration = s2 - s1;
        ex2_rate += (float(rmw_iters) / static_cast<float>(duration));
    }
    ex2_rate /= float(test_iters);
    log("\nAvg rate of operations per ms over %d tests: %f\n", test_iters, ex2_rate);

    //-------------- ATOMIC FA --------------

    log("----------------------------------------------------------\n");
    log("Testing Atomic Fetch Add...\n");
    vector<uint32_t> faSpvCode =
    #include "chunking/atomic_fa.cinit"
    ;

    Program faProgram = Program(device, faSpvCode, buffers);
    faProgram.setWorkgroups(workgroups);
    faProgram.setWorkgroupSize(workgroup_size);
    faProgram.initialize("rmw_test");
  
    float fa_rate = 0.0;
    for (int i = 1; i <= test_iters; i++) {
        resultBuf.clear();
        auto start = high_resolution_clock::now();
        faProgram.run();
        auto stop = high_resolution_clock::now();
        auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        auto duration = s2 - s1;
        fa_rate += (float(rmw_iters) / static_cast<float>(duration));
    }
    fa_rate /= float(test_iters);
    log("\nAvg rate of operations per ms over %d tests: %f\n", fa_rate, test_iters);

    //-------------- ATOMIC FA RELAXED --------------

    log("----------------------------------------------------------\n");
    log("Testing Atomic Fetch Add RELAXED...\n");
    vector<uint32_t> fa2SpvCode =
    #include "chunking/atomic_fa_relaxed.cinit"
    ;

    Program fa2Program = Program(device, fa2SpvCode, buffers);
    fa2Program.setWorkgroups(workgroups);
    fa2Program.setWorkgroupSize(workgroup_size);
    fa2Program.initialize("rmw_test");
  
    float fa2_rate = 0.0;
    for (int i = 1; i <= test_iters; i++) {
        resultBuf.clear();
        auto start = high_resolution_clock::now();
        fa2Program.run();
        auto stop = high_resolution_clock::now();
        auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        auto duration = s2 - s1;
        fa2_rate += (float(rmw_iters) / static_cast<float>(duration));
    }
    fa2_rate /= float(test_iters);
    log("\nAvg rate of operations per ms over %d tests: %f\n", fa2_rate, test_iters);

    log("----------------------------------------------------------\n");
    log("Cleaning up...\n");

    casProgram.teardown();
    cas2Program.teardown();
    cas3Program.teardown();
    exProgram.teardown();
    ex2Program.teardown();
    faProgram.teardown();
    fa2Program.teardown();

    resultBuf.teardown();
    rmwItersBuf.teardown();
    paddingBuf.teardown();
    contentionBuf.teardown();
    sizeBuf.teardown();
    garbageBuf.teardown();
        
    device.teardown();
    instance.teardown();

    json result_json = {
        {"os-name", os_name()},
        {"device-name", device.properties.deviceName},
        {"device-type", vkDeviceType(device.properties.deviceType)},
        {"workgroups", workgroups},
        {"rmw-iters", rmw_iters},
        {"test-iters", test_iters},
        {"total-rmws", total_rmws},
        {"cas-rate", cas_rate},
        {"cas2-rate", cas2_rate},
        {"cas3-rate", cas3_rate},
        {"ex-rate", ex_rate},
        {"ex2-rate", ex2_rate},
        {"fa-rate", fa_rate},
        {"fa2-rate", fa2_rate},
    };

    string json_string = result_json.dump();
    char* json_cstring = new char[json_string.size() + 1];
    copy(json_string.data(), json_string.data() + json_string.size() + 1, json_cstring);
    return json_cstring;
}

extern "C" char* run_default(uint32_t padding, uint32_t contention, uint32_t workgroup_size) {
    uint32_t workgroups = 8;
    uint32_t rmw_iters = 8192;
    uint32_t test_iters = 16;
    return run(workgroups, workgroup_size, padding, contention, rmw_iters, test_iters);
}

int main() {
    // Contention/Padding Values
    list<uint32_t> test_values = {1, 2, 4, 8, 16, 32, 64, 128}; 
    // Number of Threads
    list<uint32_t> workgroup_size_values = {32}; 
    char* res = NULL;
    for (const auto& workgroup_size : workgroup_size_values) {
        log("WORKGROUP_SIZE = %d\n", workgroup_size);
        log("--------------------------------------------------------------------------------------------------------------------\n");
        for (auto it1 = test_values.begin(); it1 != test_values.end(); ++it1) {
            for (auto it2 = test_values.begin(); it2 != test_values.end(); ++it2) {
                uint32_t contention = *it1;
                uint32_t padding = *it2;
                log("CONTENTION = %d\tPADDING = %d\n", contention,padding);
                res = run_default(padding, contention, workgroup_size);
                //log("%s\n", res);
            }
        }
        // switch statement 
        log("--------------------------------------------------------------------------------------------------------------------\n");
    }


    log("FINISHED TEST SUITES");
    delete[] res;
    return 0;
}