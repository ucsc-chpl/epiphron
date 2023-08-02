static void lock(__global atomic_uint* next_ticket, __global atomic_uint* now_serving) {
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
    uint x;
    for (int i = 0; i < 256; i++) {
        lock(next_ticket, now_serving);

        // Increment global counter.
        x = *counter;
        x++;
        *counter = x;

        // Update histogram to measure fairness.
        histogram[get_global_id(0)]++;

        unlock(now_serving);
    }
}
