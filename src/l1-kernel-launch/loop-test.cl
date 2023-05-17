__kernel void litmus_test(__global uint *numLoops, __global uint *input, __global uint *output) {
    size_t id = get_global_id(0);
    uint i;
    float result = (float) input[id];
    for (i = 0; i < numLoops[0]; i += 1) {
        result = result * result + 1.0f;
    }
    output[id] = (uint) result;
}
