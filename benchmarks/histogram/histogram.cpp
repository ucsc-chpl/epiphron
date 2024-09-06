#include "histogram.h"

namespace histogram {

    void modifyLocalMemSize(std::vector<uint32_t>& spirv, uint32_t newValue) {
        // SPIR-V magic number to verify the binary
        const uint32_t SPIRV_MAGIC = 0x07230203;
        // NOTE: This function modifies the first OpConstant instruction it finds with a
        // value of LOCAL_MEM_SIZE (defined below). It does no semantic check of type or 
        // variable name, so ensure the constant you want to modify doens't conflict with
        // any previously defined constant (i.e ensure that the first #define constant w/ value
        // 1024 is at the top of the OpenCL file).
        const uint32_t LOCAL_MEM_SIZE = 8192; 
        if(spirv.size() < 5) {
            std::cerr << "Invalid SPIR-V binary." << std::endl;
            return;
        }
        if(spirv[0] != SPIRV_MAGIC) {
            std::cerr << "Not a SPIR-V binary." << std::endl;
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
                // This is a simplistic check.
                // Doesn't verify the type and name (through debug info)
                if(constantValue == LOCAL_MEM_SIZE) {
                    spirv[i+3] = newValue;
                    return;
                }
            }
            i += length; // move to next instruction
        }
        std::cerr << "Did not modify any instructions when parsing the SPIR-V module!\n";
    }
    
    std::vector<uint32_t> read_spirv(const char *filename) {
        auto fin = std::ifstream(filename, std::ios::binary | std::ios::ate);
        if (!fin.is_open())
        {
        throw std::runtime_error(std::string("failed opening file ") + filename + " for reading");
        }
        const auto stream_size = unsigned(fin.tellg());
        fin.seekg(0);

        auto ret = std::vector<std::uint32_t>((stream_size + 3) / 4, 0);
        std::copy(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>(), reinterpret_cast<char *>(ret.data()));
        return ret;
    }

    float cpuHistogram(std::vector<uint64_t>& bins, uint32_t* data, uint64_t len, uint32_t num_bins) {
        auto start = std::chrono::high_resolution_clock::now();
        bins.resize(num_bins);
        for (uint64_t i = 0; i < len; i++) {
            uint32_t bin = data[i] % num_bins;
            bins[bin]++;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> runtime = (end - start);
        return runtime.count();
    }

    Histogram::Histogram(easyvk::Device device, uint32_t* data, uint64_t len, uint32_t num_bins, enum Implementation impl) {
        if (impl == CPU) {
            float runtime = cpuHistogram(bins, data, len, num_bins);
            printf("Ran in %f ms.\n", runtime);
            return;
        }
        if (impl == MULTILEVEL) {
            fprintf(stderr, "Not implemented!\n");
            return;
        }

        easyvk::Buffer d_data = easyvk::Buffer(device, len * sizeof(uint32_t), true);
        uint64_t data_size = len * sizeof(uint32_t);
        // If data > 1GB, copy it over in 1GB intervals so as to not overflow memory with staging buffer size
        if (data_size > (1llu << 30)) {
            for (uint64_t i = 0; i < data_size; i += (1llu << 30)) {
                uint64_t chunk_size = (1llu << 30);
                if ((data_size - i) < chunk_size)
                    chunk_size = data_size - i;
                d_data.store(data, chunk_size, i, i);
            }
        } else {
            d_data.store(data, data_size);
        }
        
        std::vector<uint32_t> spvCode;
        size_t binType_size = 0;
        uint32_t phys_bins = num_bins; // Number of "physical" bins, for <32-bit bins packed into 32-bit atomics
        switch (impl) {
            case GLOBAL:
                binType_size = sizeof(uint32_t);
                spvCode = read_spirv("shaders/histogram-global.spv");
                break;
            case SHARED_UINT8:
                binType_size = sizeof(uint8_t);
                phys_bins /= 4;
                spvCode = read_spirv("shaders/histogram-shared-uint8.spv");
                break;
            case SHARED_UINT16:
                binType_size = sizeof(uint16_t);
                phys_bins /= 2;
                spvCode = read_spirv("shaders/histogram-shared-uint16.spv");
                break;
            case SHARED_UINT32:
                binType_size = sizeof(uint32_t);
                spvCode = read_spirv("shaders/histogram-shared-uint32.spv");
                break;
            case SHARED_UINT64:
                binType_size = sizeof(uint64_t);
                spvCode = read_spirv("shaders/histogram-shared-uint64.spv");
                break;
            case MULTILEVEL:
                binType_size = sizeof(uint32_t);
                spvCode = read_spirv("shaders/histogram-multilevel.spv");
                break;
        }
        easyvk::Buffer d_bins = easyvk::Buffer(device, phys_bins * binType_size, true);
        d_bins.fill(0);

        if (impl == SHARED_UINT8 || impl == SHARED_UINT16 || impl == SHARED_UINT32 || impl == SHARED_UINT64) {
            if (phys_bins * binType_size > device.properties.limits.maxComputeSharedMemorySize)
                fprintf(stderr, "WARNING: Allocated local memory size '%zu' bytes exceeds maximum size of '%u' bytes.\n", phys_bins * binType_size, device.properties.limits.maxComputeSharedMemorySize);
            modifyLocalMemSize(spvCode, phys_bins * binType_size);
        }

        easyvk::Buffer d_meta = easyvk::Buffer(device, 3 * sizeof(uint32_t), true);
        d_meta.store(&len, sizeof(uint32_t));
        d_meta.store(&num_bins, sizeof(uint32_t), 0, sizeof(uint32_t));

        std::vector<easyvk::Buffer> bufs = {d_data, d_bins, d_meta};
        easyvk::Program program = easyvk::Program(device, spvCode, bufs);
        
        // Still need to do occupancy discovery here
        // (glsl requires workgroup size to bet set in-shader, need to write code to modify spv if it needs to be changed at runtime)
        program.setWorkgroups(10);
        program.initialize("main");
        float runtime = program.runWithDispatchTiming();
        printf("Ran in %f ms.\n", runtime / 1000000.0);

        void* _bins = (void*)malloc(phys_bins * binType_size);
        d_bins.load(_bins, phys_bins * binType_size);
        bins.resize(num_bins);
        for (int i = 0; i < num_bins; i++) {
            switch(impl) {
                case GLOBAL:
                    bins[i] = ((uint32_t*)_bins)[i];
                    break;
                case SHARED_UINT8:
                    bins[i] = ((uint8_t*)_bins)[i];
                    break;
                case SHARED_UINT16:
                    bins[i] = ((uint16_t*)_bins)[i];
                    break;
                case SHARED_UINT32:
                    bins[i] = ((uint32_t*)_bins)[i];
                    break;
                case SHARED_UINT64:
                    bins[i] = ((uint64_t*)_bins)[i];
                    break;
                case MULTILEVEL:
                    bins[i] = ((uint32_t*)_bins)[i];
                    break;
            }
        }
        free(_bins);

        program.teardown();
        d_data.teardown();  
        d_bins.teardown();
        d_meta.teardown();
    }

}