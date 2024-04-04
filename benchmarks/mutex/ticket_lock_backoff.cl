static void lock(__global atomic_uint* next_ticket, __global atomic_uint* now_serving, volatile global uchar* sleep) {
    //atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_relaxed, memory_scope_device);

    uint my_ticket = atomic_fetch_add(next_ticket, 1);

    uint index = get_global_id(0) * 32;
    while (atomic_load_explicit(now_serving, memory_order_relaxed) != my_ticket) {
        uint delay = (my_ticket - atomic_load_explicit(now_serving, memory_order_relaxed)) * 1000000;
        while (delay--) {
            uchar temp = sleep[index];
        } //inc
    }
} // do FA(mem, 0)? make sleep atomic?, maybe backoff not beneficial on this GPU?

static void unlock(__global atomic_uint* now_serving) {
    atomic_store_explicit(now_serving, atomic_load(now_serving) + 1, memory_order_relaxed);
}

__kernel void mutex_test(__global atomic_uint* now_serving, global uint* res, global uint* iters, global atomic_uint* next_ticket, volatile global uchar* sleep) {
    uint x;
    for (uint i = 0; i < *iters; i++) {
        lock(next_ticket, now_serving, sleep);
        x = *res;
        x++;
        *res = x;
        unlock(now_serving);
    }
} 
