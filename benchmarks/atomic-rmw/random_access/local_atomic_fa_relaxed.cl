#define LOCAL_SIZE 2048

__kernel void rmw_test( __global atomic_uint* res, global uint* iters, global uint* global_mapping, global uint* buf_size) {
    
    __local atomic_uint local_res[LOCAL_SIZE];
    
    uint prev = global_mapping[get_global_id(0)];
    uint index = 0;
    for (uint i = 0; i < *iters; i++) {
        index = ((prev * 1664525) + 1013904223) % (*buf_size);
        atomic_fetch_add_explicit(&local_res[index], 1, memory_order_relaxed);
        prev = index;
    }
    atomic_fetch_add_explicit(&res[global_mapping[get_global_id(0)]], atomic_load(&local_res[index]), memory_order_relaxed);
}
