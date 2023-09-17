static void lock(global atomic_uint* l) {
    uint e, acq;
    e = 0;
    acq = 0;
    while (acq == 0) {
        while(atomic_load_explicit(l, memory_order_relaxed) == 1);
        e = 0;
        acq = atomic_compare_exchange_strong(l, &e, 1);
    }
}

static uint try_lock(global atomic_uint* l) {
    uint e = 0;
    return atomic_compare_exchange_strong(l, &e, 1);
}

static void unlock(global atomic_uint* l) {
    atomic_store(l, 0);
}

__kernel void mutex_test(__global atomic_uint* l, global uint* res, global uint* iters) {
    uint x;
    for (uint i = 0; i < *iters; i++) {
        if (try_lock(l)); 
        else lock(l);
        x = *res;
        x++;
        *res = x;
        unlock(l);
    }
} 
