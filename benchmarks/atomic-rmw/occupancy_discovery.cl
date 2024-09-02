#define PARTICIPATING 1
#define NON_PARTICIPATING 0
#define LOCAL_MEM_SIZE 8192


// Ticket lock 
static void lock(__global atomic_uint* next_ticket, __global atomic_uint* now_serving) {
    uint my_ticket;
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_relaxed, memory_scope_device);
    my_ticket = atomic_fetch_add(next_ticket, 1);
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

__kernel void occupancy_discovery(__global atomic_uint* global_histogram, 
                                  __global uint* iters, 
                                  __global uint* seed,
                                  __global uint* bucket_size, 
                                  __global uint* local_mapping,
                                  __global uint *count, 
                                  __global uint *poll_open,
                                  __global uint *M,
                                  __global atomic_uint *now_serving,
                                  __global atomic_uint *next_ticket) {
    
    __local atomic_uint local_histogram[LOCAL_MEM_SIZE];

    uint prev = seed[get_global_id(0)];
    uint index = 0, offset = 0;
    uint atomic_location = local_mapping[get_local_id(0)];

    // Single represesentative thread from each workgroups runs the occupancy_discovery protocol
    if (get_local_id(0) == 0) {
        get_occupancy(count, poll_open, M, now_serving, next_ticket);
    }

    for (uint i = 0; i < (*iters); i++) {
        index = ((prev * 8121) + 28411) % (*bucket_size);
        offset = atomic_location + index;
        atomic_fetch_add_explicit(&local_histogram[offset], 1, memory_order_relaxed);
        prev = index;
    }

    if (get_local_id(0) == 0) {
        atomic_fetch_add_explicit(&global_histogram[index], atomic_load(&local_histogram[offset]), memory_order_relaxed);
    }

    // atomic_fetch_add(&local_histogram[get_global_id(0) % LOCAL_MEM_SIZE], 1);

    return;
}
