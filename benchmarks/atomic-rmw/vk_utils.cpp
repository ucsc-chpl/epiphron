#include "vk_utils.h"
#include <iostream>
#include <sstream>
#include <random>
using namespace std;
using easyvk::Instance;
using easyvk::Device;
using easyvk::Buffer;
using easyvk::Program;
using easyvk::vkDeviceType;

#ifdef __ANDROID__
#define APPNAME "GPURmwTests"
#endif

// Validated correctness of thread access pattern results
uint32_t validate_output(easyvk::Buffer resultBuf, uint32_t rmw_iters, uint32_t test_iters, uint32_t contention, uint32_t padding, uint32_t size, string thread_dist, string test_name) {
   
    uint32_t error_count = 0;
    if (test_name == "atomic_fa_relaxed" && thread_dist != "random_access") {
        for (int access = 0; access < size; access += padding) {
            uint32_t observed_output;
            resultBuf.load(&observed_output, sizeof(uint32_t));
            uint32_t expected_output = rmw_iters * test_iters * contention;
            if (observed_output != expected_output) {
                error_count++;
            }
    }
    }
    return error_count;
}

uint32_t occupancy_discovery(easyvk::Device device, uint32_t workgroup_size, uint32_t workgroups, vector<uint32_t> spv_code, 
                             uint32_t test_iters, uint32_t rmw_iters, uint32_t bucket_size, uint32_t thread_count, uint32_t local_mem) {
        int maxOccupancyBound = 0;
        for (int i = 0; i < test_iters; i++) {

            Buffer result_buf = Buffer(device, bucket_size * sizeof(uint32_t), true);
            result_buf.clear();

            Buffer size_buf = Buffer(device, sizeof(uint32_t), true);
            size_buf.store(&bucket_size, sizeof(uint32_t));

            Buffer rmw_iters_buf = Buffer(device, sizeof(uint32_t), true);
            rmw_iters_buf.store(&rmw_iters, sizeof(uint32_t));
            
            uint64_t global_work_size = workgroup_size * workgroups;
            Buffer strat_buf = Buffer(device, global_work_size * sizeof(uint32_t), true); 
            Buffer local_strat_buf = Buffer(device, workgroup_size * sizeof(uint32_t), true);

            Buffer thread_buf = Buffer(device, sizeof(uint32_t), true);
            thread_buf.store(&thread_count, sizeof(uint32_t));

            Buffer branch_buf = Buffer(device, global_work_size * sizeof(uint32_t), true); 

            random_device rd;
            mt19937 gen(rd()); 
            uniform_int_distribution<> distribution(0, bucket_size-1);

            vector<uint32_t> strat_buf_host, local_strat_buf_host, branch_buf_host;
            for (int i = 0; i < global_work_size; i++) {
                strat_buf_host.push_back(distribution(gen));
                branch_buf_host.push_back((i % 2));
            }
            for (int i = 0; i < workgroup_size; i++) {
                local_strat_buf_host.push_back((i / thread_count) * (bucket_size));
            }

            if (strat_buf_host.size() > 0)
                strat_buf.store(strat_buf_host.data(), strat_buf_host.size() * sizeof(uint32_t));
            if (local_strat_buf_host.size() > 0)
                local_strat_buf.store(local_strat_buf_host.data(), local_strat_buf_host.size() * sizeof(uint32_t));
            if (branch_buf_host.size() > 0)
                branch_buf.store(branch_buf_host.data(), branch_buf_host.size() * sizeof(uint32_t));
            
            
            Buffer count_buf = Buffer(device, sizeof(uint32_t), true);
            uint32_t zero = 0; // need to figure out the right way to pass an rvalue to a void* but i'm lazy
            count_buf.store(&zero, sizeof(uint32_t));
            Buffer poll_open_buf = Buffer(device, sizeof(uint32_t), true);
            uint32_t poll_init_val = 1;
            poll_open_buf.store(&poll_init_val, sizeof(uint32_t)); // Poll is initially open.
            Buffer M_buf = Buffer(device, workgroups * sizeof(uint32_t), true);
            Buffer now_serving_buf = Buffer(device, sizeof(uint32_t));
            now_serving_buf.store(&zero, sizeof(uint32_t));
            Buffer next_ticket_buf = Buffer(device, sizeof(uint32_t), true);
            next_ticket_buf.store(&zero, sizeof(uint32_t));
            Buffer local_mem_buf = Buffer(device, sizeof(uint32_t), true);
            local_mem_buf.store(&local_mem, sizeof(uint32_t));
            vector<Buffer> kernelInputs = {             result_buf, rmw_iters_buf, strat_buf, size_buf, local_strat_buf, branch_buf,//thread_buf,
                                                        count_buf, 
                                                        poll_open_buf,
                                                        M_buf,
                                                        now_serving_buf,
                                                        next_ticket_buf,
                                                        local_mem_buf};
            auto program = Program(device, spv_code, kernelInputs);
            program.setWorkgroups(workgroups);
            program.setWorkgroupSize(workgroup_size);
            program.initialize("occupancy_discovery");
            program.run();
            uint32_t measured;
            count_buf.load(&measured, sizeof(uint32_t));
            if (measured > maxOccupancyBound) {
                maxOccupancyBound = measured;
            }
            program.teardown();
            result_buf.teardown();
            rmw_iters_buf.teardown();
            strat_buf.teardown();
            size_buf.teardown();
            local_strat_buf.teardown();
            thread_buf.teardown();
            branch_buf.teardown();
            count_buf.teardown();
            poll_open_buf.teardown();
            M_buf.teardown();
            next_ticket_buf.teardown();
            now_serving_buf.teardown();
        }
        return (uint32_t) maxOccupancyBound;
}

void modifyLocalMemSize(vector<uint32_t>& spirv, uint32_t newValue, const uint32_t LOCAL_MEM_SIZE) {
    if(spirv.size() < 5) {
        cerr << "Invalid SPIR-V binary." << endl;
        return;
    }
    if(spirv[0] != 0x07230203) {
        cerr << "Not a SPIR-V binary." << endl;
        return;
    }

    // Iterate through SPIR-V instructions
    // https://github.com/KhronosGroup/SPIRV-Guide/blob/master/chapters/parsing_instructions.md
    size_t i = 5;  // skip SPIR-V header
    while(i < spirv.size()) {
        uint32_t instruction = spirv[i];
        uint32_t length = instruction >> 16;
        uint32_t opcode = instruction & 0x0ffffu;

        // Opcode source: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConstant
        if(opcode == 43) { // OpConstant
            uint32_t resultType = spirv[i+1];
            uint32_t resultId = spirv[i+2];
            uint32_t constantValue = spirv[i+3];

            // This is a simplistic check
            // Doesn't verify the type and name (through debug info)
            if(constantValue == LOCAL_MEM_SIZE) {
                spirv[i+3] = newValue;
                return;
            }
        }
        i += length; // move to next instruction
    }

    cerr << "Did not modify any instructions when parsing the SPIR-V module!\n";
}

vector<int> select_configurations(vector<string> options, string prompt) {
    cout << prompt << endl;
    for (size_t i = 0; i < options.size(); ++i) {
        cout << i + 1 << ". " << options[i] << endl;
    }
    vector<int> selections;
    string input;
    cout << "Enter choices separated by space: ";
    getline(cin, input); 
    stringstream ss(input);
    int choice;
    while (ss >> choice) {
        if (choice >= 1 && choice <= options.size()) {
            selections.push_back(choice - 1);
        }
    }
    return selections;
}

vector<uint32_t> get_spv_code(const string& filename) {
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

uint32_t get_params(string prompt) {
    int value;
    string input;
    while (true) {
        cout << prompt;
        getline(cin, input);
        stringstream ss(input);
        if (ss >> value && ss.eof() && value > 0) {
            break;
        }
        cout << "Invalid input" << endl;
    }

    return value;
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
