__kernel void kernel_barrier(__global uint *output_buf,
                            __global uint *i) {
    if (get_local_id(0) == 0) {
        output_buf[(get_group_id(0) + i) % get_num_groups(0)]++;
    }
}