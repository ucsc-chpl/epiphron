static void lock(global atomic_uint* l, uint index) {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_relaxed, memory_scope_device);
    uint e, acq;
    e = 0;
    acq = 0;
    while (acq == 0) {
        while(atomic_load_explicit(&l[index], memory_order_relaxed) == 1);
        e = 0;
        acq = atomic_compare_exchange_strong(&l[index], &e, 1);
    }
}

static uint try_lock(global atomic_uint* l, uint index) {
    uint e = 0;
    return atomic_compare_exchange_strong(&l[index], &e, 1);
}

static void unlock(global atomic_uint* l, uint index) {
    atomic_store(&l[index], 0);
}

__kernel void mutex_test(__global atomic_uint* l, global uint* res, global uint* iters, global uint* contention_size) {
    uint index = get_global_id(0) / (*contention_size);
    uint x;
    for (uint i = 0; i < *iters; i++) {
        if (try_lock(l, index)); 
        else lock(l, index);
        x = res[index];
        x++;
        res[index] = x;
        unlock(l, index);
    }
} 
