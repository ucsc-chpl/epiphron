static void lock(global atomic_uint* l, uint index) {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_relaxed, memory_scope_device);
    uint e;
    atomic_uint acq;
    e = 0;
    atomic_store(&acq, 0);
    while (atomic_load(&acq) == 0) {
        atomic_store(&acq, atomic_compare_exchange_strong(&l[index], &e, 1));
        e = 0;
    }
}

static void unlock(global atomic_uint* l, uint index) {
    atomic_store(&l[index], 0);
}

//Striding
__kernel void mutex_test(__global atomic_uint* l, global uint* res, global uint* iters, global uint* padding_size, global uint* buffer_size) {
    uint index = get_global_id(0) * (*padding_size) % (*buffer_size); // striding
    uint x;
    for (uint i = 0; i < *iters; i++) {
        lock(l, index);
        x = res[index];
        x++;
        res[index] = x;
        unlock(l, index);
    }
} 
