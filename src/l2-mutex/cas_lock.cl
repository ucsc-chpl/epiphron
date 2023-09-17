static void lock(global atomic_uint* l) {
    //atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_relaxed, memory_scope_device);
    uint e, acq;
    e = 0;
    acq = 0;
    while (acq == 0) {
        acq = atomic_compare_exchange_strong(l, &e, 1);
        e = 0;
    }
}

static void unlock(global atomic_uint* l) {
    atomic_store(l, 0);
}

__kernel void mutex_test(__global atomic_uint* l, global uint* res, global uint* iters) {
    uint x;
    for (uint i = 0; i < *iters; i++) {
        lock(l);
        x = *res;
        x++;
        *res = x;
        unlock(l);
    }
} 
