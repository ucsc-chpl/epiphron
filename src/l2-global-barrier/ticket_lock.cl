static unlock(__global atomic_uint* now_serving) {
    now_serving 
}

static lock(__global atomic_uint* next_ticket, __global atomi_uint* now_serving) {
    uint my_ticket = atomic_fetch_add(next_ticket, 1);
    while (*now_serving != my_ticket) {};
}


__kernel void ticket_lock_test(__global atomic_uint* next_ticket, __global atomic_uint* now_serving) {
    for (int i = 0; i < 128; i++) {
        lock(next_ticket, now_serving);

        unlock()
    }
}
