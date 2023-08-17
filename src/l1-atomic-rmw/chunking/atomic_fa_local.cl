#define LOCAL_SIZE 128 * 128

// Atomic Fetch Add Relaxed - Allocate global memory that simulates local memory
__kernel void rmw_test(__global atomic_uint* res, global uint* iters, global uint* padding_size, global uint* contention_size) {
    __local atomic_uint local_res[LOCAL_SIZE];

    uint global_index = (get_global_id(0) / (*contention_size)) * (*padding_size); // chunking
    uint local_index = (get_local_id(0) / (*contention_size)) * (*padding_size);

    for (uint i = 0; i < *iters; i++) {
        atomic_fetch_add_explicit(&local_res[local_index], 1, memory_order_acq_rel);
    }

    atomic_store(&res[global_index], atomic_load(&local_res[local_index]));
}
