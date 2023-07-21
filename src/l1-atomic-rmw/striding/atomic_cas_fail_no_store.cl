// Atomic Compare Exchange Strong Fail No Store
__kernel void rmw_test(__global atomic_uint* res, global uint* iters, global uint* padding_size, global uint* buffer_size) {
    uint index = get_global_id(0) * (*padding_size) % (*buffer_size); // striding
    for (uint i = 0; i < *iters; i++) {
        atomic_compare_exchange_strong_explicit(&res[index], padding_size, (*padding_size), memory_order_relaxed, memory_order_relaxed);
    }
}
