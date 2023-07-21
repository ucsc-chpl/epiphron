// Atomic Compare Exchange Strong Succeed Store
__kernel void rmw_test(__global atomic_uint* res, global uint* iters, global uint* padding_size, global uint* contention_size) {
    uint index = (get_global_id(0) / (*contention_size)) * (*padding_size); // chunking
    for (uint i = 0; i < *iters; i++) {
        uint expected = atomic_load_explicit(&res[index], memory_order_acquire);
        atomic_compare_exchange_strong_explicit(&res[index], &expected, expected + 1, memory_order_acq_rel, memory_order_acquire);
    }
}
