#define LOCAL_MEM_SIZE 8192

__kernel void rmw_test( __global atomic_uint* global_histogram, global uint* iters, 
                        global uint* seed, global uint* bucket_size, 
                        global uint* local_mapping) {
    
    __local atomic_uint local_histogram[LOCAL_MEM_SIZE];

    uint prev = seed[get_global_id(0)];
    uint index = 0, offset = 0;
    uint atomic_location = local_mapping[get_local_id(0)];

    for (uint i = 0; i < (*iters); i++) {
        index = ((prev * 8121) + 28411) % (*bucket_size);
        offset = atomic_location + index;
        atomic_fetch_add_explicit(&local_histogram[offset], 1, memory_order_relaxed);
        prev = index;
    }
    
    if (get_local_id(0) == 0) {
        atomic_fetch_add_explicit(&global_histogram[index], atomic_load(&local_histogram[offset]), memory_order_relaxed);
    }
}
