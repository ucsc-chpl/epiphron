#define LOCAL_BUF_SIZE 256 * 1

kernel void noBarrier(global uint *buf, global uint *buf_size, global uint *num_iters) {
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

kernel void localSubgroupBarrier(global uint *buf, global uint *buf_size, global uint *num_iters) {
    // Workgroup-local memory.
    local uint local_buf[LOCAL_BUF_SIZE]; 
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void globalSubgroupBarrier(global uint *buf, global uint *buf_size, global uint *num_iters) {
    local uint local_buf[LOCAL_BUF_SIZE]; 
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

        sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

kernel void localWorkgroupBarrier(global uint *buf, global uint *buf_size, global uint *num_iters) {
    local uint local_buf[LOCAL_BUF_SIZE]; 
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void globalWorkgroupBarrier(global uint *buf, global uint *buf_size, global uint *num_iters) {
    local uint local_buf[LOCAL_BUF_SIZE];
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
