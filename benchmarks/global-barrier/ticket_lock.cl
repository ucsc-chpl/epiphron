static void lock(__global atomic_uint* next_ticket, __global atomic_uint* now_serving) {
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_relaxed, memory_scope_device);
    uint my_ticket = atomic_fetch_add(next_ticket, 1);
    while (atomic_load(now_serving) != my_ticket) {}
}

static void unlock(__global atomic_uint* now_serving) {
    atomic_store(now_serving, atomic_load(now_serving) + 1);
}

__kernel void ticket_lock_test(__global atomic_uint* next_ticket, 
                               __global atomic_uint* now_serving, 
                               __global uint* counter,
                               __global uint* histogram) {
    for (int i = 0; i < 256; i++) {
        lock(next_ticket, now_serving);

        // Increment global counter.
        (*counter)++;

        // Update histogram to measure fairness.
        histogram[get_global_id(0)]++;

        unlock(now_serving);
    }
}
