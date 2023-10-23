#include "bvh.h"
#include <stdlib.h>
#include <iostream>


BVH::BVH(Mesh *triangle_mesh) {
    mesh = triangle_mesh;
    // For a mesh with N triangles, the corresponding BVH will never have more than 2N - 1 nodes.
    nodes = (BVHNode *) malloc(sizeof(BVHNode) * mesh->triCount * 2 - 1);
    tri_indices = new uint32_t[mesh->triCount];
    nodes_used = 0;

    // Populate triangle index array.
    for (int i = 0; i < mesh->triCount; i++) {
        tri_indices[i] = i;
    }

    // Calculate triangle centroids for partitioning.
    for (int i = 0; i < mesh->triCount; i++) {
        Tri tri = mesh->tri[i];
        float cx = (tri.v0x + tri.v1x + tri.v2x) * 0.3333f;
        float cy = (tri.v0y + tri.v1y + tri.v2y) * 0.3333f;
        float cz = (tri.v0z + tri.v1z + tri.v2z) * 0.3333f;
        mesh->tri[i].cx = cx;
        mesh->tri[i].cy = cy;
        mesh->tri[i].cz = cz;
    }

    BVHNode *root = &nodes[0];
    nodes_used++;
    root->left_first = 0;
    root->tri_count = mesh->triCount;
    update_node_bounds(0);
    subdivide(0);
}

void BVH::update_node_bounds(uint32_t node_idx) {
    BVHNode *node = &nodes[node_idx];
    // Initialize aabb bounds.
    node->minx = 1e30f; node->miny = 1e30f; node->minz = 1e30f;
    node->maxx = -1e30f; node->maxy = -1e30f; node->maxz = -1e30f;
    for (int first = node->left_first, i = 0; i < node->tri_count; i++) {
        uint32_t leaf_tri_idx = tri_indices[first + i];
        Tri &tri = mesh->tri[leaf_tri_idx];
        // Initialize min and max variables with first vertex values
        float minX = tri.v0x, maxX = tri.v0x;
        float minY = tri.v0y, maxY = tri.v0y;
        float minZ = tri.v0z, maxZ = tri.v0z;

        // Check against second vertex
        if (tri.v1x < minX) minX = tri.v1x;
        if (tri.v1x > maxX) maxX = tri.v1x;
        if (tri.v1y < minY) minY = tri.v1y;
        if (tri.v1y > maxY) maxY = tri.v1y;
        if (tri.v1z < minZ) minZ = tri.v1z;
        if (tri.v1z > maxZ) maxZ = tri.v1z;

        // Check against third vertex
        if (tri.v2x < minX) minX = tri.v2x;
        if (tri.v2x > maxX) maxX = tri.v2x;
        if (tri.v2y < minY) minY = tri.v2y;
        if (tri.v2y > maxY) maxY = tri.v2y;
        if (tri.v2z < minZ) minZ = tri.v2z;
        if (tri.v2z > maxZ) maxZ = tri.v2z;

        // Now minX, maxX, minY, maxY, minZ, maxZ contain the AABB bounds
        node->minx = minX; node->miny = minY; node->minz = minZ;
        node->maxx = maxX; node->maxy = maxY; node->maxz = maxZ;
    }
}

void BVH::subdivide(uint32_t node_idx) {
    std::cout << "\nSubdividing node " << node_idx << "\n";
    BVHNode& node = nodes[node_idx];
    std::cout << "tri count: " << node.tri_count << "\n";
    if (node.tri_count <= 2) return; // terminate recursion

    // Determine split axis and position
    float3 extent = {
        node.maxx - node.minx, 
        node.maxy - node.miny,
        node.maxz - node.minz,
        0
    };
    std::cout << "aabb extent: " << extent.x << " " << extent.y << " " << extent.z << "\n";
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    std::cout << "Split axis: " << axis << "\n";
    float split_pos;
    if (axis == 0) split_pos = node.minx + extent.x * 0.5f;
    else if (axis == 1) split_pos = node.miny + extent.y * 0.5f;
    else if (axis == 2) split_pos = node.minz + extent.z * 0.5f;

    // In-place partition
    int i = node.left_first;
    int j = i + node.tri_count - 1;
    while (i <= j) {
        Tri tri = mesh->tri[tri_indices[i]];
        float3 centroid = {tri.cx, tri.cy, tri.cz, 0};
        if (centroid[axis] < split_pos) {
            i++;
        } else {
            std::swap(tri_indices[i], tri_indices[j--]);
        }
    }

    std::cout << "Finished in-place partitioning of triangles\n";

    // Abort split if one of the sides is empty
    int left_count = i - node.left_first;
    if (left_count == 0 || left_count == node.tri_count) return;


    // Create child nodes
    std::cout << "Creating child nodes\n";
    int left_child_idx = nodes_used;
    nodes_used++;
    int right_child_idx = nodes_used;
    nodes_used++;
    std::cout << "l child: " << left_child_idx << " r child: " << right_child_idx << "\n";
    nodes[left_child_idx].left_first = node.left_first;
    nodes[left_child_idx].tri_count = left_count;
    nodes[right_child_idx].left_first = i;
    nodes[right_child_idx].tri_count = node.tri_count - left_count;
    node.left_first = left_child_idx;
    node.tri_count = 0; // set to 0 since it's not a leaf.
    update_node_bounds(left_child_idx);
    update_node_bounds(right_child_idx);
    // recurse
    subdivide(left_child_idx);
    subdivide(right_child_idx);
}
