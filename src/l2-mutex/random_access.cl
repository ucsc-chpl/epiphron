__kernel void mutex_test(global atomic_uint* res, global uint* iters, global uint* buf_size, global uint* indexes) {
    uint prev = indexes[get_global_id(0)]; // initialize as seed
    uint index;
    for (uint i = 0; i < *iters; i++) {
        index = ((prev * 1664525) + 1013904223) % (*buf_size);
        atomic_fetch_add_explicit(&res[index], 1, memory_order_relaxed);
        prev = index;
    }
} 
