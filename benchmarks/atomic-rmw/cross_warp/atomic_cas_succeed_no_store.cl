// Atomic Fetch Add Relaxed
__kernel void rmw_test( __global atomic_uint* res, global uint* iters, 
                        global uint* indexes) {

    uint index = indexes[get_global_id(0)]; // chunking
    uint expected = atomic_load_explicit(&res[index], memory_order_relaxed);
    for (uint i = 0; i < *iters; i++) {
        atomic_compare_exchange_strong_explicit(&res[index], &expected, expected, memory_order_relaxed, memory_order_relaxed);
    }
}
