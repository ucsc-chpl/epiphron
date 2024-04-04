__kernel void test(__global uint *a, __global uint *b, __global uint *c, __global uint *dispatchSizeBuf, __global uint *vecSize) {
    for (uint i = get_global_id(0); i < vecSize[0]; i += dispatchSizeBuf[0]) {
        c[i] = a[i] + b[i];
    }
}