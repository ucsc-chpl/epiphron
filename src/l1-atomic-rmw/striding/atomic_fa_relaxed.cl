// Atomic Fetch Add Relaxed
__kernel void rmw_test(__global atomic_uint* res, global uint* iters, global uint* padding_size, global uint* buffer_size) {
    uint index = get_global_id(0) * (*padding_size) % (*buffer_size); // striding                                      
    for (uint i = 0; i < *iters; i++) {
        atomic_fetch_add_explicit(&res[index], 1, memory_order_relaxed);
    }
}
