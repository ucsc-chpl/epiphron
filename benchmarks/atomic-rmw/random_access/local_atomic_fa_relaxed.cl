#define LOCAL_SIZE 8192

// global_histogram              atomic_buffer: bucket_size
// iters                         rmw_iters
// seed                          initial random number
// bucket_size                   256
// local_mapping                 (local_id / thread count) * bucket_size
// local_histogram               atomic_buffer: (workgroup_size / thread_count) * bucket_size

__kernel void rmw_test( __global atomic_uint* global_histogram, global uint* iters, 
                        global uint* seed, global uint* bucket_size, 
                        global uint* local_mapping, global uint* thread_count) {
    
    __local atomic_uint local_histogram[LOCAL_SIZE];

    uint prev = seed[get_global_id(0)];
    uint index = 0, offset = 0;
    uint atomic_location = local_mapping[get_local_id(0)];

    for (uint i = 0; i < (*iters); i++) {
        index = ((prev * 75) + 74) % (*bucket_size);
        offset = atomic_location + index;
        atomic_fetch_add_explicit(&local_histogram[offset], 1, memory_order_relaxed);
        prev = index;
    }
    
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0; i < (*bucket_size); i++) {
        if (get_local_id(0) % (*thread_count) == 0) {
            atomic_fetch_add_explicit(&global_histogram[i], atomic_load(&local_histogram[atomic_location+i]), memory_order_relaxed);
        }
    }
}
