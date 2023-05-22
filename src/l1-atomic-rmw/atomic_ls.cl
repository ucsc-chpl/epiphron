// Atomic load() and store()
// Cannot guarantee the increment is performed on the most recent state
// -> will compute incorrect results
__kernel void rmw_test(__global atomic_uint* res, global uint* iters, global uint* padding_size, global uint* buffer_size) {
    uint index =  get_global_id(0) * (*padding_size) % (*buffer_size);
    uint x;
    for (uint i = 0; i < *iters; i++) {
        x = atomic_load_explicit(&res[index], memory_order_acquire);
        x++;
        atomic_store_explicit(&res[index], x, memory_order_release);
    }
}
