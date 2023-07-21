// Atomic Compare Exchange Strong Succeed No Store
__kernel void rmw_test(__global atomic_uint* res, global uint* iters, global uint* padding_size, global uint* contention_size) {
    uint index = (get_global_id(0) / (*contention_size)) * (*padding_size); // chunking
    for (uint i = 0; i < *iters; i++) {
        uint expected = atomic_load_explicit(&res[index], memory_order_relaxed);
        atomic_compare_exchange_strong_explicit(&res[index], &expected, expected, memory_order_relaxed, memory_order_relaxed);
    }
}
