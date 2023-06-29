#define LOCAL_BUF_SIZE 1024 * 1

kernel void benchmark(global uint *buf, global uint *buf_size, global uint *num_iters) {
    // Workgroup-local memory.
    local uint local_buf[LOCAL_BUF_SIZE]; 
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        // Changes should be flused and visible to workgroup.
        uint local_idx = (get_local_id(0) + 1) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        // Won't be flushed?
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
}
