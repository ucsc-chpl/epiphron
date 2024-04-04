
__kernel void baselineBfs(__global uint *num_vertices,
                          __global uint *curr,
                          __global uint *costs,
                          __global uint *vertices,
                          __global uint *edges,
                          __global bool *finished) {
    uint v = get_global_id(0);

    if (v < *num_vertices && costs[v] == *curr) {
        // Iterate over neighbors of v.
        uint num_nbr = vertices[v+1] - vertices[v];
        uint edge_idx = vertices[v];
        for (uint i = 0; i < num_nbr; i++) {
            uint w = edges[edge_idx];
            if (costs[w] == UINT_MAX) { // if not visited yet
                *finished = false;
                costs[w] = *curr + 1;
            }
            edge_idx++;
        }
    }

    return;
}
