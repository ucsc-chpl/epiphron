// Atomic Exchange
__kernel void rmw_test(__global atomic_uint* res, global uint* iters, global uint* padding_size, global uint* buffer_size) {
    uint index =  get_global_id(0) * (*padding_size) % (*buffer_size);
    uint previous_val, new_val;
    for (uint i = 0; i < *iters; i++) {
        previous_val = atomic_load_explicit(&res[index], memory_order_acquire);
        new_val = previous_val + 1;
        atomic_exchange_explicit(&res[index], new_val, memory_order_acq_rel);
    }
}
