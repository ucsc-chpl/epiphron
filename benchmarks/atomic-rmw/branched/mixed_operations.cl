// Atomic Fetch Add Relaxed
__kernel void rmw_test( __global atomic_uint* res, global uint* iters, 
                        global uint* indexes, global uint* branch, global uint* mixed) {
    
    uint index = indexes[get_global_id(0)];                             
    for (uint i = 0; i < *iters; i++) {
        if (branch[get_global_id(0)]) {
            if (mixed[get_global_id(0)]) {
                atomic_fetch_add_explicit(&res[index], 1, memory_order_relaxed);
            } else {
                atomic_fetch_max_explicit(&res[index], i + get_global_id(0), memory_order_relaxed);
            }
        }
    }
}
