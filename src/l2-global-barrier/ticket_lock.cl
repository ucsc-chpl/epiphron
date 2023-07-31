static void unlock(__global atomic_uint* now_serving) {
    atomic_store(now_serving, atomic_load(now_serving) + 1);
}

static void lock(__global atomic_uint* next_ticket, __global atomic_uint* now_serving) {
    uint my_ticket = atomic_fetch_add(next_ticket, 1);
    while (atomic_load(now_serving) != my_ticket) {}
}

__kernel void ticket_lock_test(__global atomic_uint* next_ticket, 
                               __global atomic_uint* now_serving, 
                               __global uint* histogram,
                               __global uint* counter) {
    for (int i = 0; i < 256; i++) {
        lock(next_ticket, now_serving);

        // Increment global counter.
        (*counter)++;

        // Update histogram.
        histogram[get_global_id(0)]++;

        unlock(now_serving);
    }
}
