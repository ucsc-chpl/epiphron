static void lock(__global atomic_uint* next_ticket, __global atomic_uint* now_serving, uint index) {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_relaxed, memory_scope_device);

    uint my_ticket = atomic_fetch_add(&next_ticket[index], 1);

    while (atomic_load(&now_serving[index]) != my_ticket) {}
}

static void unlock(__global atomic_uint* now_serving, uint index) {
    atomic_store(&now_serving[index], atomic_load(&now_serving[index]) + 1);
}

__kernel void mutex_test(__global atomic_uint* now_serving, global uint* res, global uint* iters, global uint* contention_size, global atomic_uint* next_ticket) {
    uint index = get_global_id(0) / (*contention_size);
    uint x;
    for (uint i = 0; i < *iters; i++) {
        lock(next_ticket, now_serving, index);
        x = res[index];
        x++;
        res[index] = x;
        unlock(now_serving, index);
    }
} 
