#define PARTICIPATING 1
#define NON_PARTICIPATING 0

// Ticket lock --------------------------------------------------------------------------
static void lock(__global atomic_uint *next_ticket, __global atomic_uint *now_serving) {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_relaxed, memory_scope_device);
    uint my_ticket = atomic_fetch_add(next_ticket, 1);
    while (atomic_load(now_serving) != my_ticket) {}
}

static void unlock(__global atomic_uint *now_serving) {
    atomic_store(now_serving, atomic_load(now_serving) + 1);
}

// Occupancy Discovery ------------------------------------------------------------------
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


// Occupancy-Bound Execution Environment ------------------------------------------------
static uint p_get_num_groups(__global uint *count) {
    return *count;
}

static uint p_get_group_id(__global uint *M) {
    return M[get_group_id(0)];
}

static uint p_get_global_id(__global uint *M) {
    return (p_get_group_id(M) * get_local_size(0)) + get_local_id(0);
}

static uint p_get_global_size(__global uint *count) {
    return (*count) * get_local_size(0);
}

__kernel void global_barrier(__global uint *count, 
                             __global uint *poll_open,
                             __global uint *M,
                             __global atomic_uint *now_serving,
                             __global atomic_uint *next_ticket,
                             __global atomic_uint *flag,
                             __global uint *output_buf,
                             __global uint *num_iters) {
    // Single represesentative thread from each workgroups runs the occupancy_discovery protocol
    __local uint participating[1];
    if (get_local_id(0) == 0) {
        participating[0] = get_occupancy(count, poll_open, M, now_serving, next_ticket);
    }

    // Wait for representative thread to finish.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Workgroups found to not be participating immediately exit.
    if (participating[0] == 0) {
        return;
    }

    // Participating workgroups continue with kernel computation. 
    // From here we can assume fair scheduling of workgroups.    
    // Occupancy-bound execution environment variables.
    uint p_group_id = p_get_group_id(M);
    uint p_num_groups = p_get_num_groups(count);

    for (int i = 0; i < *num_iters; i++) {
        // Each workgroup will write to a location in a global buffer, offset each
        // iteration. A safe and correct barrier is necessary to ensure that there 
        // are no data races and every iteration each workgroup is writing to it's 
        // own memory location.
        // NOTE: This is assuming that there are at least as many threads per 
        // workgroup as there are participating workgroups!.
        if (get_local_id(0) == 0) {
            // Only one thread within the workgroup does the work.
            output_buf[(p_group_id + i) % p_num_groups]++;
        } 

        // Global Barrier -----------------------------------------------------
        if (p_group_id == 0) {
            // Controller workgroup
            if (get_local_id(0) + 1 < p_num_groups) {
                for (int j = get_local_id(0) + 1; j < p_num_groups; j += get_local_size(0)) {
                    // Each thread is responsible for waiting for some subset of the workgroups
                    // to arrive to the barrier.
                    while (atomic_load(&flag[j]) == 0);
                }
            }

            // Wait for all threads to have their follower workgroup arrive.
            barrier(CLK_GLOBAL_MEM_FENCE);

            if (get_local_id(0) + 1 < p_num_groups) {
                for (int j = get_local_id(0) + 1; j < p_num_groups; j += get_local_size(0)) {
                    // Release workgroups from barrier.
                    atomic_store(&flag[j], 0);
                }
            }
        } else {
            barrier(CLK_GLOBAL_MEM_FENCE);

            // Follower workgroups
            if (get_local_id(0) == 0) {
                // Update flag to signal arrival to barrier.
                atomic_store(&flag[p_group_id], 1);
                while (atomic_load(&flag[p_group_id]) == 1);
            }

            // Wait for all flags to be updated before proceeding.
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
}
