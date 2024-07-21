#define LOCAL_SIZE 2048

// Atomic Fetch Add Relaxed
__kernel void rmw_test(__global atomic_uint* res, global uint* iters, 
                        global uint* global_mapping, global uint* local_mapping) {

    __local atomic_uint local_res[LOCAL_SIZE];

    uint global_index = global_mapping[get_global_id(0)];
    uint local_index = local_mapping[get_local_id(0)];

    for (uint i = 0; i < *iters; i++) {
        atomic_fetch_add_explicit(&local_res[local_index], 1, memory_order_relaxed);
    }

    atomic_fetch_add_explicit(&res[global_index], atomic_load(&local_res[local_index]), memory_order_relaxed);
}
