__kernel void render(__global uint *image_buffer,
                     __global uint *image_buf_width,
                     __global uint *image_buf_height) {

    uint idx = get_global_id(0);
    if (idx >= image_buf_width[0] * image_buf_height[0]) {
        return;
    }

    image_buffer[idx] |= 0xFF0000FF; 
}
