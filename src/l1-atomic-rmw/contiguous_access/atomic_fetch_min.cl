// Atomic Fetch Add Relaxed
__kernel void rmw_test( __global atomic_uint* res, global uint* iters, 
                        global uint* indexes) {

    uint index = indexes[get_global_id(0)]; // chunking
    atomic_store_explicit(&res[index], (*iters) + 1, memory_order_relaxed);
    for (uint i = 0; i < *iters; i++) {
        atomic_fetch_min_explicit(&res[index], (*iters) - i, memory_order_relaxed);
    }
}
