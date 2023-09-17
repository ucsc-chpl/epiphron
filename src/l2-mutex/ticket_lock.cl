static void lock(__global atomic_uint* next_ticket, __global atomic_uint* now_serving) {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_relaxed, memory_scope_device);

    uint my_ticket = atomic_fetch_add(next_ticket, 1);

    while (atomic_load(now_serving) != my_ticket) {}
}

static void unlock(__global atomic_uint* now_serving) {
    atomic_store(now_serving, atomic_load(now_serving) + 1);
}

__kernel void mutex_test(__global atomic_uint* now_serving, global uint* res, global uint* iters, global atomic_uint* next_ticket) {
    uint x;
    for (uint i = 0; i < *iters; i++) {
        lock(next_ticket, now_serving);
        x = *res;
        x++;
        *res = x;
        unlock(now_serving);
    }
} 
