#define LOCAL_SIZE 8192

__kernel void rmw_test( __global atomic_uint* res, global uint* iters, global uint* seed, 
                        global uint* atomics, global uint* local_mapping) {
    
    __local atomic_uint local_res[LOCAL_SIZE];
    
    uint prev = seed[get_global_id(0)];
    uint index = 0, offset = 0;
    uint atomic_location = local_mapping[get_local_id(0)];
    for (uint i = 0; i < *iters; i++) {
        index = ((prev * 75) + 74) % (*atomics);
        offset = atomic_location + index;
        atomic_fetch_add_explicit(&local_res[offset], 1, memory_order_relaxed);
        prev = index;
    }
    atomic_fetch_add_explicit(&res[offset], atomic_load(&local_res[offset]), memory_order_relaxed);
}
