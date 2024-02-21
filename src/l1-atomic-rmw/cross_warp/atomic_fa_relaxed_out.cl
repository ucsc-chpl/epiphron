// Atomic Fetch Add Relaxed
__kernel void rmw_test( __global atomic_uint* res, global uint* iters, 
                        global uint* indexes, global uint* output) {
    
    uint index = indexes[get_global_id(0)];      
    uint tmp = 0;
    for (uint i = 0; i < *iters; i++) {
        tmp = atomic_fetch_add_explicit(&res[index], 1, memory_order_relaxed);
    }
    output[get_global_id(0)] = tmp;
}
