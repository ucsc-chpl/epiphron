#include <list>
#include <iostream>
#include <random>
#include "vk_utils.h"
#include "json.hpp"

#ifdef __ANDROID__
#include <android/log.h>
#define USE_VALIDATION_LAYERS false
#define APPNAME "GPURmwTests"
#else
#define USE_VALIDATION_LAYERS false
#endif

using namespace std;
using easyvk::Instance;
using easyvk::Device;
using easyvk::Buffer;
using easyvk::Program;
using easyvk::vkDeviceType;

extern "C" void rmw_microbenchmark(easyvk::Device device, uint32_t workgroup_size, uint32_t test_iters, 
                                    string thread_dist, vector<uint32_t> spv_code, string test_name) {
    
    ofstream benchmark_data; 
    benchmark_data.open(string("results/") + device.properties.deviceName + "_" + thread_dist + "_" + test_name + ".txt"); 
    if (!benchmark_data.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return;
    }

    // benchmark_data << to_string(workgroup_size) + "," + to_string(workgroups) + ":" + device.properties.deviceName
    //               << ", " << thread_dist << ": " << test_name << endl;

    list<uint32_t> test_values;
    uint32_t max_local_memory_size = device.properties.limits.maxComputeSharedMemorySize / 4; //using buffers of uint32_t
    uint32_t bucket_size = 256; //global bucket // local: bucket_size * (workgroup_size/thread_count)
    for (uint32_t i = 1; i <= workgroup_size; i *= 2) {
        if (((workgroup_size/i)*bucket_size) <= max_local_memory_size) {
            test_values.push_back(i);
        }
    }
    uint32_t loading_counter = 0;
    uint32_t occupancy_test_iters = 16;
    uint32_t occupancy_rmw_iters = 1024;
    uint32_t occupancy_upper_bound = 3072;
    vector<uint32_t> occupancy_spv_code = get_spv_code("occupancy_discovery.cinit");
    uint32_t prev_local_memory_size = 8192;
    for (uint32_t thread_count : test_values) {
              
        //modify local mem 
        // occupancy discovery
        uint32_t curr_local_memory_size = (workgroup_size/thread_count)*bucket_size;

        modifyLocalMemSize(occupancy_spv_code, curr_local_memory_size, prev_local_memory_size);
        
        uint32_t workgroups = occupancy_discovery(device, workgroup_size, occupancy_upper_bound, occupancy_spv_code, 
                                                  occupancy_test_iters, occupancy_rmw_iters, bucket_size, thread_count, curr_local_memory_size);
        
        benchmark_data << "(" << thread_count << ", " << (workgroup_size/thread_count)*bucket_size << ", " << workgroups << ", ";
    
        modifyLocalMemSize(spv_code, curr_local_memory_size, prev_local_memory_size);

        prev_local_memory_size = curr_local_memory_size;

        uint64_t global_work_size = workgroup_size * workgroups;

        Buffer result_buf = Buffer(device, bucket_size * sizeof(uint32_t), true);
        Buffer size_buf = Buffer(device, sizeof(uint32_t), true);
        size_buf.store(&bucket_size, sizeof(uint32_t));
        //Buffer thread_buf = Buffer(device, sizeof(uint32_t), true);
        //thread_buf.store(&thread_count, sizeof(uint32_t));
        Buffer rmw_iters_buf = Buffer(device, sizeof(uint32_t), true);
        Buffer strat_buf = Buffer(device, global_work_size * sizeof(uint32_t), true); 
        Buffer local_strat_buf = Buffer(device, workgroup_size * sizeof(uint32_t), true); 

        random_device rd;
        mt19937 gen(rd()); 
        uniform_int_distribution<> distribution(0, bucket_size-1);

        vector<uint32_t> strat_buf_host, local_strat_buf_host; 
        for (int i = 0; i < global_work_size; i++) {
            strat_buf_host.push_back(distribution(gen));
        }
        for (int i = 0; i < workgroup_size; i++) {
            local_strat_buf_host.push_back((i / thread_count) * (bucket_size));
        }
        if (strat_buf_host.size() > 0)
            strat_buf.store(strat_buf_host.data(), strat_buf_host.size() * sizeof(uint32_t));
        if (local_strat_buf_host.size() > 0)
            local_strat_buf.store(local_strat_buf_host.data(), local_strat_buf_host.size() * sizeof(uint32_t));

        uint32_t rmw_iters = 128;
        while(1) {
            result_buf.clear();
            rmw_iters_buf.store(&rmw_iters, sizeof(uint32_t));
            vector<Buffer> buffers = {result_buf, rmw_iters_buf, strat_buf};

            if (thread_dist == "random_access") {
                buffers.emplace_back(size_buf);
                buffers.emplace_back(local_strat_buf);
                //buffers.emplace_back(thread_buf);
            }
            //if (test_name == "local_atomic_fa_relaxed") buffers.emplace_back(local_strat_buf);

            Program rmw_program = Program(device, spv_code, buffers);
            rmw_program.setWorkgroups(workgroups);
            rmw_program.setWorkgroupSize(workgroup_size);
            rmw_program.initialize("rmw_test");
            float total_rate = 0.0;
            float total_duration = 0.0;
            for (int i = 1; i <= test_iters; i++) {
                auto kernel_time = rmw_program.runWithDispatchTiming();
                total_duration += (kernel_time / (double) 1000.0);
                total_rate += ((static_cast<float>(rmw_iters) * workgroup_size * workgroups) / (kernel_time / (double) 1000.0)); 
            }
            rmw_program.teardown();
            if ((total_duration/test_iters) > 500000.0) {
                benchmark_data << total_rate/test_iters << ")" << endl;
                break;
            }
            rmw_iters *= 2;
        }

        result_buf.teardown();
        rmw_iters_buf.teardown();
        strat_buf.teardown();
        size_buf.teardown();
        local_strat_buf.teardown();
        //thread_buf.teardown();

        loading_counter++;
        if (thread_dist == "random_access") {
            cout << "\r" << thread_dist << ", " << test_name << ": "
            << int(((float)loading_counter / (test_values.size() * test_values.size())) * 100.0) << "% ";
        } else if (test_name == "local_atomic_fa_relaxed" && thread_dist != "random_access") {
            cout << "\r" << thread_dist << ", " << test_name << ": "
            << int(((float)loading_counter / (test_values.size() * 4)) * 100.0) << "% ";
        } else {
            cout << "\r" << thread_dist << ", " << test_name << ": " 
            << int(((float)loading_counter / (test_values.size() * test_values.size())) * 100.0) << "% ";
        }
        cout.flush();
    }

    benchmark_data.close();
    return;
}

extern "C" void rmw_benchmark_suite(easyvk::Device device, const vector<string> &thread_dist, const vector<string> &atomic_rmws) {  
    uint32_t test_iters = 3;
    uint32_t workgroup_size = device.properties.limits.maxComputeWorkGroupInvocations;
    // cout << "Workgroups: (" << workgroup_size << ", 1) x " << workgroups << endl;
    for (const string& strategy : thread_dist) {
        for (const string& rmw : atomic_rmws) {
            vector<uint32_t> spv_code = get_spv_code(strategy + "/" + rmw + ".cinit");
            rmw_microbenchmark(device, workgroup_size, test_iters, strategy, spv_code, rmw);
            cout << endl;
        }
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
    vector<string> thread_dist_options = {
        "random_access"
    };
    vector<string> atomic_rmw_options = {
        "atomic_fa_relaxed",
        "local_atomic_fa_relaxed",
    };

    auto selected_devices = select_configurations(device_options, "Select devices:");
    auto thread_dist_choices = select_configurations(thread_dist_options, "\nSelect thread distributions:");
    auto atomic_rmws_choices = select_configurations(atomic_rmw_options, "\nSelect atomic RMWs:");

    vector<string> selected_thread_dist, selected_atomic_rmws;

    for (const auto& choice : thread_dist_choices) {
        selected_thread_dist.push_back(thread_dist_options[choice]);
    }
    for (const auto& choice : atomic_rmws_choices) {
        selected_atomic_rmws.push_back(atomic_rmw_options[choice]);
    }

    for (const auto& choice : selected_devices) {
        auto device = easyvk::Device(instance, physicalDevices.at(choice));
        cout << "\nRunning RMW benchmarks on " << device.properties.deviceName << endl;
        rmw_benchmark_suite(device, selected_thread_dist, selected_atomic_rmws);
        device.teardown();
    }
    
    instance.teardown();
    return 0;
}
