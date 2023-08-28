__kernel void kernel_barrier(__global uint *output_buf,
                            __global uint *i,
                            __global uint *num_workgroups) {
    if (get_local_id(0) == 0) {
        uint idx = (get_group_id(0) + (*i)) % *num_workgroups;
        output_buf[idx]++;
    }
}
