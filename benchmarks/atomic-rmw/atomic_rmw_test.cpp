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
#define USE_VALIDATION_LAYERS true
#endif

using namespace std;
using easyvk::Instance;
using easyvk::Device;
using easyvk::Buffer;
using easyvk::Program;
using easyvk::vkDeviceType;

extern "C" void rmw_microbenchmark(easyvk::Device device, uint32_t workgroups, uint32_t workgroup_size, uint32_t test_iters, 
                                    string thread_dist, vector<uint32_t> spv_code, string test_name) {
    
    ofstream benchmark_data; 
    benchmark_data.open(string("results/") + device.properties.deviceName + "_" + thread_dist + "_" + test_name + ".txt"); 
    if (!benchmark_data.is_open()) {
        cerr << "Failed to open the output file." << endl;
        return;
    }
    
    benchmark_data << to_string(workgroup_size) + "," + to_string(workgroups) + ":" + device.properties.deviceName
                  << ", " << thread_dist << ": " << test_name << endl;

    list<uint32_t> test_values;
    uint32_t test_range = min((workgroups * workgroup_size > 1024) ? 1024 : workgroup_size * workgroups, 
                    (test_name == "local_atomic_fa_relaxed") ? 256 : workgroup_size * workgroups);

    for (uint32_t i = 1; i <= test_range; i *= 2) {
        test_values.push_back(i);
    }
    uint32_t loading_counter = 0;
    for (uint32_t contention : test_values) {

        int random_access_status = 0;
        for (uint32_t padding : test_values) {
            
            if (test_name == "local_atomic_fa_relaxed" && padding > 8) continue;
            
            if  (thread_dist == "random_access") {
                if (!random_access_status) random_access_status = 1;
                else continue;
            }

            benchmark_data << "(" << contention << ", " << padding << ", ";


            uint32_t global_work_size = workgroup_size * workgroups;
            uint32_t size = ((global_work_size) * padding) / contention;

            Buffer result_buf = Buffer(device, thread_dist == "random_access" ? contention : size, sizeof(uint32_t));


            Buffer random_access_buf = Buffer(device, 1, sizeof(uint32_t));
            random_access_buf.store<uint32_t>(0, contention);

            Buffer rmw_iters_buf = Buffer(device, 1, sizeof(uint32_t));

            Buffer out_buf = Buffer(device, global_work_size, sizeof(uint32_t));
            Buffer strat_buf = Buffer(device, global_work_size, sizeof(uint32_t)); 
            Buffer local_strat_buf = Buffer(device, workgroup_size, sizeof(uint32_t)); 
            Buffer branch_buf = Buffer(device, global_work_size, sizeof(uint32_t)); 
            Buffer mixed_buf = Buffer(device, global_work_size, sizeof(uint32_t));

            random_device rd;
            mt19937 gen(rd()); 
            uniform_int_distribution<> distribution(0, contention-1);

            for (int i = 0; i < global_work_size; i += 1) {
                mixed_buf.store<uint32_t>(i, (i % 32) < 16); // Thread instruction masking
                if (thread_dist == "branched") {    
                    branch_buf.store<uint32_t>(i, (i % 2));
                    strat_buf.store<uint32_t>(i, (i / contention) * padding);
                } else if (thread_dist == "cross_warp") {
                    strat_buf.store<uint32_t>(i, (i * padding) % size);
                } else if (thread_dist == "contiguous_access") {
                    strat_buf.store<uint32_t>(i, (i / contention) * padding);
                } else if (thread_dist == "random_access") {
                    strat_buf.store<uint32_t>(i, distribution(gen));
                }
            }
            for (int i = 0; i < workgroup_size; i += 1) {
                if (thread_dist == "branched") {    
                    local_strat_buf.store<uint32_t>(i, (i / contention) * padding);
                } else if (thread_dist == "cross_warp") {
                    local_strat_buf.store<uint32_t>(i, (i * padding) % ((workgroup_size * padding) / contention));
                } else if (thread_dist == "contiguous_access") {
                    local_strat_buf.store<uint32_t>(i, (i / contention) * padding);
                } else if (thread_dist == "random_access") {
                    local_strat_buf.store<uint32_t>(i, distribution(gen));
                }
            }
            uint32_t rmw_iters = 128;
            while(1) {
                result_buf.clear();
                rmw_iters_buf.store<uint32_t>(0, rmw_iters);
                vector<Buffer> buffers = {result_buf, rmw_iters_buf, strat_buf};

                if (thread_dist == "branched") buffers.emplace_back(branch_buf);
                else if (thread_dist == "random_access") buffers.emplace_back(random_access_buf);
                
                if (test_name == "atomic_fa_relaxed_out") buffers.emplace_back(out_buf);
                else if (test_name == "local_atomic_fa_relaxed") buffers.emplace_back(local_strat_buf);
                else if (test_name == "mixed_operations") buffers.emplace_back(mixed_buf);

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
                if ((total_duration/test_iters) > 1000000.0) {
                    benchmark_data << total_rate/test_iters << ")" << endl;
                    break;
                }
                rmw_iters *= 2;
            }

            result_buf.teardown();
            random_access_buf.teardown();
            rmw_iters_buf.teardown();
            branch_buf.teardown();
            mixed_buf.teardown();
            local_strat_buf.teardown();
            strat_buf.teardown();
            out_buf.teardown();

            loading_counter++;
            if (thread_dist == "random_access") {
                cout << "\r" << thread_dist << ", " << test_name << ": "
                << int(((float)loading_counter / (test_values.size())) * 100.0) << "% ";
            } else if (test_name == "local_atomic_fa_relaxed") {
                cout << "\r" << thread_dist << ", " << test_name << ": "
                << int(((float)loading_counter / (test_values.size() * 4)) * 100.0) << "% ";
            } else {
                cout << "\r" << thread_dist << ", " << test_name << ": " 
                << int(((float)loading_counter / (test_values.size() * test_values.size())) * 100.0) << "% ";
            }
            cout.flush();
        }
    }

    benchmark_data.close();
    return;
}

extern "C" void rmw_benchmark_suite(easyvk::Device device, const vector<string> &thread_dist, const vector<string> &atomic_rmws) {  
    uint32_t test_iters = 3;
    uint32_t workgroup_size = device.properties.limits.maxComputeWorkGroupInvocations;
    uint32_t workgroups = occupancy_discovery(device, workgroup_size, 256, get_spv_code("occupancy_discovery.cinit"), 16, 1024);
    cout << "Workgroups: (" << workgroup_size << ", 1) x " << workgroups << endl;
    for (const string& strategy : thread_dist) {
        for (const string& rmw : atomic_rmws) {
            vector<uint32_t> spv_code = get_spv_code(strategy + "/" + rmw + ".cinit");
            if (rmw == "local_atomic_fa_relaxed") rmw_microbenchmark(device, workgroups, 256, test_iters, strategy, spv_code, rmw);
            else rmw_microbenchmark(device, workgroups, workgroup_size, test_iters, strategy, spv_code, rmw);
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
        "branched", 
        "cross_warp",
        "contiguous_access",
        "random_access"
    };
    vector<string> atomic_rmw_options = {
        "atomic_fa_relaxed",
        "atomic_fa_relaxed_out",
        "local_atomic_fa_relaxed",
        "atomic_cas_succeed_store",
        "atomic_cas_succeed_no_store",
        "atomic_fetch_min",
        "atomic_fetch_max",
        "atomic_exchange",
        "mixed_operations"
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
