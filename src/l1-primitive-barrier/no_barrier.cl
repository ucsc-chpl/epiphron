#define LOCAL_BUF_SIZE 256 * 1

kernel void benchmark(global uint *buf, global uint *buf_size, global uint *num_iters) {
    // Workgroup-local memory.
    local uint local_buf[LOCAL_BUF_SIZE]; 
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

    }
}
