#include "common.h"

typedef struct warp_mem_t {
    // TODO: What if we also store curr?
    uint costs[CHUNK_SIZE];
    uint vertices[CHUNK_SIZE + 1];
} warp_mem_t;

// TODO: The explicit addresss space in the args may not be needed. See:
// https://man.opencl.org/genericAddressSpace.html
static void memcpy_global_to_local(uint size, __local uint *dest, __global uint *src) {
    for (uint idx = get_local_id(0); idx < size; idx += get_local_size(0)) {
        dest[idx] = src[idx];
    }
    // TODO: Which type of fence to use?
    work_group_barrier(CLK_LOCAL_MEM_FENCE); 
}

// static void expand_bfs(uint cnt,
//                        __global uint *edges,
//                        __global uint *costs,
//                        uint curr,
//                        __global bool *finished) {
//     for (uint idx = get_local_id(0); idx < cnt; idx += get_local_size(0)) {
//         uint v = edges[idx];
//         if (costs[v] == UINT_MAX) {
//             // Vertex hasn't been visited yet.
//             costs[v] = curr + 1;
//             *finished = false;
//         }
//     }
//     work_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 
// }

__kernel void warp_bfs(__global uint *num_vertices,
                      __global uint *_curr,
                      __global uint *costs,
                      __global uint *vertices,
                      __global uint *edges,
                      __global bool *finished) {
    // Copy my work to local
    __local warp_mem_t warp_mem;
    uint curr = *_curr;
    uint work_start = get_group_id(0) * CHUNK_SIZE; 
    if (work_start + CHUNK_SIZE >= *num_vertices) {
        // Truncate work to avoid segfault.
        memcpy_global_to_local(*num_vertices - work_start, warp_mem.costs, &costs[work_start]);
        memcpy_global_to_local(*num_vertices - work_start + 1, warp_mem.vertices, &vertices[work_start]);
    } else {
        memcpy_global_to_local(CHUNK_SIZE, warp_mem.costs, &costs[work_start]);
        memcpy_global_to_local(CHUNK_SIZE + 1, warp_mem.vertices, &vertices[work_start]);
    }

    // Iterate over my work.
    for (uint v = 0; v < CHUNK_SIZE; v++) {
        if (warp_mem.costs[v] == curr) { // Vertex is part of frontier to be expanded.

            uint num_nbr = warp_mem.vertices[v+1] - warp_mem.vertices[v];
            uint edge_index = warp_mem.vertices[v]; // Starting index of neighbors

            // Threads within workgroup expand neighbors of this vertex in parallel.
            for (uint idx = get_local_id(0); idx < num_nbr; idx += get_local_size(0)) {
                uint w = edges[edge_index + idx]; // neigbor of v
                if (costs[w] == UINT_MAX) {
                    // Vertex hasn't been visited yet.
                    costs[w] = curr + 1;
                    *finished = false;
                }
            }

            // Re-align threads before continuing.
            work_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 
        }
    }
}
