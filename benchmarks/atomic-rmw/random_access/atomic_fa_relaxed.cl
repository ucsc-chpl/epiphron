__kernel void rmw_test( __global atomic_uint* res, global uint* iters, global uint* seed, global uint* size, global uint* padding) {
    uint prev = seed[get_global_id(0)];
    uint index = 0, offset = 0;
    for (uint i = 0; i < *iters; i++) {
        index = (((prev * 75) + 74) % (*size));
        offset = index * (*padding);
        atomic_fetch_add_explicit(&res[offset], 1, memory_order_relaxed);
        prev = index;
    }
} 
