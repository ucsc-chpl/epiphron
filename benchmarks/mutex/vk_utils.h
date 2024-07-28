// vk_utils.h
#ifndef VK_UTILS_H
#define VK_UTILS_H
#include "easyvk.h"

extern uint32_t occupancy_discovery(easyvk::Device device, uint32_t workgroup_size, 
                                    uint32_t a, std::vector<uint32_t> spv_code, uint32_t b);
std::vector<int> select_configurations(std::vector<std::string> options, std::string prompt);
std::vector<uint32_t> get_spv_code(const std::string &filename);
void log(const char* fmt, ...);
const char* os_name();

#endif // VK_UTILS_H