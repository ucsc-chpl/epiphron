#include <list>
#include <iostream>
#include <random>
#include "vk_utils.h"

#ifdef __ANDROID__
#include <android/log.h>
#define USE_VALIDATION_LAYERS false
#define APPNAME "GPURmwTests"
#else
#define USE_VALIDATION_LAYERS true
#endif

using namespace std;
using easyvk::Instance;
using easyvk::Device;
using easyvk::Buffer;
using easyvk::Program;
using easyvk::vkDeviceType;

extern "C" void mutex_microbenchmark(easyvk::Device device, uint32_t workgroups, uint32_t workgroup_size, 
                                uint32_t test_iters, vector<uint32_t> spv_code, string test_name) {
    
    ofstream benchmarkData; 
    benchmarkData.open(string("results/") + device.properties.deviceName + "_" + test_name + ".txt"); 
    if (!benchmarkData.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return;
    }

    benchmarkData << to_string(workgroup_size) + "," + to_string(workgroups) + ":" + device.properties.deviceName 
                    + ", " + test_name + "\n";

    list<uint32_t> test_values;
    for (uint32_t i = 1; i <= workgroups; i *= 2) {
        test_values.push_back(i);  
    }

    for (uint32_t number_atomics : test_values) {
        
        benchmarkData << "(" + to_string(number_atomics) + ", ";

        Buffer lockBuf = Buffer(device, number_atomics, sizeof(uint32_t)); 
        lockBuf.clear();
        
        Buffer resultBuf = Buffer(device, number_atomics, sizeof(uint32_t)); 
        resultBuf.clear();

        Buffer mutexItersBuf = Buffer(device, 1, sizeof(uint32_t));

        random_device rd;
        mt19937 gen(rd()); 
        uniform_int_distribution<> distribution(0, number_atomics-1);
        Buffer indexBuf = Buffer(device, workgroup_size * workgroups, sizeof(uint32_t)); 
        for (int i = 0; i < workgroup_size * workgroups; i += 1) {
            indexBuf.store<uint32_t>(i, distribution(gen));
        }

        Buffer sizeBuf = Buffer(device, 1, sizeof(uint32_t));
        sizeBuf.store<uint32_t>(0, number_atomics); 

        Buffer ticketBuf = Buffer(device, number_atomics, sizeof(uint32_t));
        ticketBuf.clear();
        
        Buffer backoffBuf = Buffer(device, (workgroup_size * workgroups) * 32, sizeof(unsigned char));
        backoffBuf.clear();

        uint32_t mutex_iters = 128;
        while(1) {
            resultBuf.clear();
            mutexItersBuf.store<uint32_t>(0, mutex_iters);
            vector<Buffer> buffers = {lockBuf, resultBuf, mutexItersBuf, indexBuf, sizeBuf};
            if (test_name == "ticket_lock") {
                buffers.emplace_back(ticketBuf); 
            } else if (test_name == "cas_lock_backoff") {
                buffers.emplace_back(backoffBuf); 
            } else if (test_name == "ticket_lock_backoff") {
                buffers.emplace_back(ticketBuf);
                buffers.emplace_back(backoffBuf); 
            }
            Program mutexProgram = Program(device, spv_code, buffers);
            mutexProgram.setWorkgroups(workgroups);
            mutexProgram.setWorkgroupSize(workgroup_size);
            mutexProgram.initialize("mutex_test");
            float total_rate = 0.0;
            float total_duration = 0.0;
            for (int i = 1; i <= test_iters; i++) {
                auto kernel_time = mutexProgram.runWithDispatchTiming();
                total_duration += (kernel_time / (double) 1000.0);
                total_rate += ((static_cast<float>(mutex_iters) * workgroup_size * workgroups) / (kernel_time / (double) 1000.0)); 
            }
            mutexProgram.teardown();
            if ((total_duration/test_iters) > 1000000.0) {
                benchmarkData << total_rate/test_iters << ")" << endl;
                break;
            }
            mutex_iters *= 2;
        }   
        
        lockBuf.teardown();
        resultBuf.teardown();
        ticketBuf.teardown();
        backoffBuf.teardown();
        mutexItersBuf.teardown();
        indexBuf.teardown();
        sizeBuf.teardown();
    }
    benchmarkData.close();
    return;
}

extern "C" void mutex_benchmark_suite(easyvk::Device device, const vector<string> &mutexes) {
    uint32_t test_iters = 3;
    uint32_t workgroup_size = 1;
    uint32_t workgroups = occupancy_discovery(device, device.properties.limits.maxComputeWorkGroupInvocations, 256, get_spv_code("occupancy_discovery.cinit"), 16);

    for (const string& mutex : mutexes) {
        mutex_microbenchmark(device, workgroups, workgroup_size, test_iters, get_spv_code(mutex + ".cinit"), mutex);
    }
    return;
}

int main() {

    auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);
	auto physicalDevices = instance.physicalDevices();

    vector<string> device_options;
    for (size_t i = 0; i < physicalDevices.size(); i++) {
        auto device = easyvk::Device(instance, physicalDevices.at(i));
        device_options.push_back(device.properties.deviceName);
        device.teardown();
    }
    vector<string> mutex_options = {
        "cas_lock_peeking",
        "cas_lock",
        //"cas_lock_backoff",
        //"ticket_lock",
        //"ticket_lock_backoff",
    };

    auto selected_devices = select_configurations(device_options, "Select devices:");
    auto mutex_choices = select_configurations(mutex_options, "\nSelect mutex implementation:");

    vector<string> selected_mutexes;

    for (const auto& choice : mutex_choices) {
        selected_mutexes.push_back(mutex_options[choice]);
    }

    for (const auto& choice : selected_devices) {
        auto device = easyvk::Device(instance, physicalDevices.at(choice));
        mutex_benchmark_suite(device, selected_mutexes);
        device.teardown();
    }
    
    instance.teardown();
    return 0;
}