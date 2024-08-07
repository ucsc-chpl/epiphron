__kernel void rmw_test( __global atomic_uint* res, global uint* iters, global uint* indexes, global uint* buf_size) {
    uint prev = indexes[get_global_id(0)];
    uint index = 0;
    uint expected = atomic_load_explicit(&res[((prev * 75) + 74) % (*buf_size)], memory_order_relaxed);
    for (uint i = 0; i < *iters; i++) {
        index = ((prev * 1664525) + 1013904223) % (*buf_size);
        atomic_compare_exchange_strong_explicit(&res[index], &expected, expected, memory_order_relaxed, memory_order_relaxed);
        prev = index;
    }
} 
