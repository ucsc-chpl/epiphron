#define LOCAL_BUF_SIZE 1024 // Don't change me! I'm modified by host code.
#define PARTICIPATING 1
#define NON_PARTICIPATING 0

// Ticket lock 
static void lock(__global atomic_uint* next_ticket, __global atomic_uint* now_serving) {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_relaxed, memory_scope_device);
    uint my_ticket = atomic_fetch_add(next_ticket, 1);
    while (atomic_load(now_serving) != my_ticket) {}
}

static void unlock(__global atomic_uint* now_serving) {
    atomic_store(now_serving, atomic_load(now_serving) + 1);
}

static uint get_occupancy(__global uint *count, 
                          __global uint *poll_open,
                          __global uint *M,
                          __global atomic_uint *now_serving,
                          __global atomic_uint *next_ticket) {
    lock(next_ticket, now_serving);
    // Polling Phase
    if (*poll_open) { 
        M[get_group_id(0)] = *count;
        *count = *count + 1;
        unlock(now_serving);
    } else {
        // Poll is no longer open. Workgroup is not participating.
        unlock(now_serving);
        return NON_PARTICIPATING;
    }

    // Closing Phase
    lock(next_ticket, now_serving);
    if (*poll_open) {
        // First workgroup to reach this point closes the poll.
        *poll_open = 0;  
    }
    unlock(now_serving);
    return PARTICIPATING;
}

// We only need to run occupancy discovery on this kernel. 
// Occupancy will be exactly the same for each kernel.
kernel void noBarrier(global uchar *buf,
                      global uint *buf_size,
                      global uint *num_iters,
                      global uint *count,
                      global uint *poll_open,
                      global uint *M,
                      global atomic_uint *now_serving,
                      global atomic_uint *next_ticket) {
    // Single represesentative thread from each workgroups runs the occupancy_discovery protocol
    if (get_local_id(0) == 0) {
        get_occupancy(count, poll_open, M, now_serving, next_ticket);
    }

    work_group_barrier(CLK_GLOBAL_MEM_FENCE);

    // Workgroup-local memory.
    local uchar local_buf[LOCAL_BUF_SIZE]; 
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];
    }
}

// kernel void localSubgroupBarrier(global uint *buf, global uint *buf_size, global uint *num_iters) {
//     // Workgroup-local memory.
//     local uint local_buf[LOCAL_BUF_SIZE]; 
//     for (uint i = 0; i < *num_iters; i++) {
//         // Modify local memory.
//         uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
//         local_buf[local_idx] += 1;

//         // Modify global memory.
//         buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

//         sub_group_barrier(CLK_LOCAL_MEM_FENCE);
//     }
// }

// kernel void globalSubgroupBarrier(global uint *buf, global uint *buf_size, global uint *num_iters) {
//     local uint local_buf[LOCAL_BUF_SIZE]; 
//     for (uint i = 0; i < *num_iters; i++) {
//         // Modify local memory.
//         uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
//         local_buf[local_idx] += 1;

//         // Modify global memory.
//         buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

//         sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
//     }
// }

// kernel void localWorkgroupBarrier(global uint *buf, global uint *buf_size, global uint *num_iters) {
//     local uint local_buf[LOCAL_BUF_SIZE]; 
//     for (uint i = 0; i < *num_iters; i++) {
//         // Modify local memory.
//         uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
//         local_buf[local_idx] += 1;

//         // Modify global memory.
//         buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

//         work_group_barrier(CLK_LOCAL_MEM_FENCE);
//     }
// }

// kernel void globalWorkgroupBarrier(global uint *buf, global uint *buf_size, global uint *num_iters) {
//     local uint local_buf[LOCAL_BUF_SIZE];
//     for (uint i = 0; i < *num_iters; i++) {
//         // Modify local memory.
//         uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
//         local_buf[local_idx] += 1;

//         // Modify global memory.
//         buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

//         work_group_barrier(CLK_GLOBAL_MEM_FENCE);
//     }
// }

