#define LOCAL_SIZE 2048

// Atomic Fetch Add Relaxed
__kernel void rmw_test(__global atomic_uint* res, global uint* iters, 
                        global uint* indexes, global uint* padding, global uint* size) {

    __local atomic_uint local_res[LOCAL_SIZE];

    uint global_index = indexes[get_global_id(0)];
    uint local_index = get_local_id(0) * *padding % *size;

    for (uint i = 0; i < *iters; i++) {
        atomic_fetch_add_explicit(&local_res[local_index], 1, memory_order_relaxed);
    }

    atomic_store(&res[global_index], atomic_load(&local_res[local_index]));
}
