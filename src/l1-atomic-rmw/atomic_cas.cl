// Atomic Compare Exchange Strong
__kernel void rmw_test(__global atomic_uint* res, global uint* iters, global uint* padding_size, global uint* buffer_size) {
    uint index =  get_global_id(0) * (*padding_size) % (*buffer_size);

    for (uint i = 0; i < *iters; i++) {
        uint expected = atomic_load(&res[index]);
        uint desired = expected + 1;
        atomic_compare_exchange_strong(&res[index], &expected, desired);
    }
}
