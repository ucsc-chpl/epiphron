// Atomic Fetch Add Relaxed
__kernel void rmw_test( __global atomic_uint* res, global uint* iters, 
                        global uint* indexes, global uint* branch) {

    uint index = indexes[get_global_id(0)]; // chunking
    for (uint i = 0; i < *iters; i++) {
        if (branch[get_global_id(0)]) {
            uint expected = atomic_load_explicit(&res[index], memory_order_relaxed);
            while (!atomic_compare_exchange_strong_explicit(&res[index], &expected, expected + 1, memory_order_relaxed, memory_order_relaxed)) {
                expected = atomic_load_explicit(&res[index], memory_order_relaxed);
            }
        } else {
            uint expected = atomic_load_explicit(&res[index], memory_order_relaxed);
            while (!atomic_compare_exchange_strong_explicit(&res[index], &expected, expected + 1, memory_order_relaxed, memory_order_relaxed)) {
                expected = atomic_load_explicit(&res[index], memory_order_relaxed);
            }
        }
    }
}
