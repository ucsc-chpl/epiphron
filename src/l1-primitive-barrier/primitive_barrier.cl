#define LOCAL_BUF_SIZE 1024 // Don't change me! I'm modified by host code.

kernel void noBarrier(global uchar *buf, global uint *buf_size, global uint *num_iters) {
    // Workgroup-local memory.
    local uchar local_buf[LOCAL_BUF_SIZE]; 
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];
    }
}

kernel void localSubgroupBarrier(global uchar *buf, global uint *buf_size, global uint *num_iters) {
    // Workgroup-local memory.
    local uchar local_buf[LOCAL_BUF_SIZE]; 
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void globalSubgroupBarrier(global uchar *buf, global uint *buf_size, global uint *num_iters) {
    local uchar local_buf[LOCAL_BUF_SIZE]; 
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

        sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

kernel void localWorkgroupBarrier(global uchar *buf, global uint *buf_size, global uint *num_iters) {
    local uchar local_buf[LOCAL_BUF_SIZE]; 
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void globalWorkgroupBarrier(global uchar *buf, global uint *buf_size, global uint *num_iters) {
    local uchar local_buf[LOCAL_BUF_SIZE];
    for (uint i = 0; i < *num_iters; i++) {
        // Modify local memory.
        uint local_idx = (get_local_id(0) + i) % LOCAL_BUF_SIZE;
        local_buf[local_idx] += 1;

        // Modify global memory.
        buf[(get_global_id(0) + i) % *buf_size] = local_buf[local_idx];

        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

