static void lock(global atomic_uint* l, uint index) {
    //atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_relaxed, memory_scope_device);
    uint e, acq;
    e = 0;
    acq = 0;
    while (acq == 0) {
        acq = atomic_compare_exchange_strong(&l[index], &e, 1);
        e = 0;
    }
}

static void unlock(global atomic_uint* l, uint index) {
    atomic_store(&l[index], 0);
}

__kernel void mutex_test(__global atomic_uint* l, global uint* res, global uint* iters, global uint* indexes, global uint* buf_size) {
    uint prev = indexes[get_global_id(0)];
    uint index = 0;
    for (uint i = 0; i < *iters; i++) {
        index = ((prev * 1664525) + 1013904223) % (*buf_size);
        lock(l, index);
        res[index]++;
        unlock(l, index);
        prev = index;
    }
} 
