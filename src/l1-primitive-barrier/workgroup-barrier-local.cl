kernel void benchmark(global uint *buf, global uint *buf_size, global uint *num_iters) {
    uint id = get_global_id(0);
    for (uint i = 0; i < *num_iters; i++) {
        uint idx = (id + i) % *buf_size;
        buf[idx] += 1;
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
}
