#include <stdexcept>
#include <stdarg.h>
#include <string>
#include <chrono>

#include "easyvk.h"
#include "../_example/json.h"

#ifdef __ANDROID__
#include <android/log.h>
#define APPNAME "GPULockTests"
#endif

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
    log("Initializing test...\n");

    auto instance = easyvk::Instance(true);
	auto physicalDevices = instance.physicalDevices();
	auto device = easyvk::Device(instance, physicalDevices.at(0));

    log("Using device '%s'\n", device.properties.deviceName);

    uint32_t maxComputeWorkGroupInvocations = device.properties.limits.maxComputeWorkGroupInvocations;
    log("MaxComputeWorkGroupInvocations: %d\n", maxComputeWorkGroupInvocations);
    if (workgroups > maxComputeWorkGroupInvocations)
        workgroups = maxComputeWorkGroupInvocations;

    uint32_t test_total = workgroups * rmw_iters;
    uint32_t total_rmws = test_total * test_iters;

    const int size = workgroup_size * padding / contention;
    
    Buffer resultBuf = Buffer(device, size);
    Buffer rmwItersBuf = Buffer(device, 1);
    Buffer paddingBuf = Buffer(device, 1);
    Buffer sizeBuf = Buffer(device, 1);
    Buffer garbageBuf = Buffer(device, workgroups * workgroup_size);
    vector<Buffer> buffers = { resultBuf, rmwItersBuf, paddingBuf, sizeBuf, garbageBuf };
    rmwItersBuf.store(0, rmw_iters);
    paddingBuf.store(0, padding);
    sizeBuf.store(0, size);

    log("%d workgroups\n%d threads per workgroup\n%d rmw accesses per thread\ntests run %d times\ncontention at %d\npadding at %d\n", 
    workgroups, workgroup_size, rmw_iters, test_iters, contention, padding);

    // -------------- ATOMIC CAS --------------

    log("----------------------------------------------------------\n");
    log("Testing Atomic Compare and Swap...\n");
    vector<uint32_t> casSpvCode =
    #include "atomic_cas.cinit"
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

    // -------------- ATOMIC EX --------------

    log("----------------------------------------------------------\n");
    log("Testing Atomic Exchange...\n");
    vector<uint32_t> exSpvCode =
    #include "atomic_ex.cinit"
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

    // -------------- ATOMIC FA --------------

    log("----------------------------------------------------------\n");
    log("Testing Atomic Fetch Add...\n");
    vector<uint32_t> faSpvCode =
    #include "atomic_fa.cinit"
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

    // -------------- ATOMIC LS --------------

    log("----------------------------------------------------------\n");
    log("Testing Atomic Load/Store...\n");
    vector<uint32_t> lsSpvCode =
    #include "atomic_ls.cinit"
    ;

    Program lsProgram = Program(device, lsSpvCode, buffers);
    lsProgram.setWorkgroups(workgroups);
    lsProgram.setWorkgroupSize(workgroup_size);
    lsProgram.initialize("rmw_test");
  
    float ls_rate = 0.0;
    for (int i = 1; i <= test_iters; i++) {
        resultBuf.clear();
        auto start = high_resolution_clock::now();
        lsProgram.run();
        auto stop = high_resolution_clock::now();
        auto s1 = duration_cast<milliseconds>(start.time_since_epoch()).count();
        auto s2 = duration_cast<milliseconds>(stop.time_since_epoch()).count();
        auto duration = s2 - s1;
        ls_rate += (float(rmw_iters) / static_cast<float>(duration));
    }
    ls_rate /= float(test_iters);
    log("\nAvg rate of operations per ms over %d tests: %f\n", ls_rate, test_iters);

    log("----------------------------------------------------------\n");
    log("Cleaning up...\n");

    casProgram.teardown();
    exProgram.teardown();
    faProgram.teardown();
    lsProgram.teardown();

    resultBuf.teardown();
    rmwItersBuf.teardown();
    paddingBuf.teardown();
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
        {"ex-rate", ex_rate},
        {"fa-rate", fa_rate},
        {"ls-rate", ls_rate},
    };

    string json_string = result_json.dump();
    char* json_cstring = new char[json_string.size() + 1];
    copy(json_string.data(), json_string.data() + json_string.size() + 1, json_cstring);
    return json_cstring;
}

extern "C" char* run_default() {
    uint32_t workgroups = 8;
    uint32_t workgroup_size = 32;
    uint32_t padding = 16;
    uint32_t contention = 8;
    uint32_t rmw_iters = 1000; //20k
    uint32_t test_iters = 64; //1024
    // programmatically change contention/padding
    //heatmap, ypadding, xcontention
    //spit out csv of timings
    return run(workgroups, workgroup_size, padding, contention, rmw_iters, test_iters);
}

int main() {
    char* res = run_default();
    log("%s\n", res);
    delete[] res;
    return 0;
}