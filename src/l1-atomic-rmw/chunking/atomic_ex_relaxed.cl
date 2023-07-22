// Atomic Exchange Relaxed
__kernel void rmw_test(__global atomic_uint* res, global uint* iters, global uint* padding_size, global uint* contention_size) {
    uint index = (get_global_id(0) / (*contention_size)) * (*padding_size); // chunking
    for (uint i = 0; i < *iters; i++) {
        uint previous_val = atomic_load_explicit(&res[index], memory_order_relaxed);
        uint new_val = previous_val + 1;
        atomic_exchange_explicit(&res[index], new_val, memory_order_relaxed);
    }
}
