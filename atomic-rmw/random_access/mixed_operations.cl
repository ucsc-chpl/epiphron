__kernel void rmw_test( __global atomic_uint* res, global uint* iters, global uint* indexes, global uint* buf_size, global uint* mixed) {
    uint prev = indexes[get_global_id(0)];
    uint index = 0;
    for (uint i = 0; i < *iters; i++) {
        index = ((prev * 1664525) + 1013904223) % (*buf_size);
        if (mixed[get_global_id(0)]) {
            atomic_fetch_add_explicit(&res[index], 1, memory_order_relaxed);
        } else {
            atomic_fetch_max_explicit(&res[index], prev, memory_order_relaxed);
        }
        prev = index;
    }
} 
