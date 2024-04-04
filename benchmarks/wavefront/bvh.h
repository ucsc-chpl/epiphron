#pragma once

#include "util.h"
#include "mesh.h"


// 32 bytes
typedef struct BVHNode {
	float minx, miny, minz;
	float maxx, maxy, maxz;
	int tri_count;
  // If node has triangles, it MUST be a leaf, thus it has no child nodes.
  // Therefore, we need to only store a child node index OR the index of the first triangle.
  // If tri_count is 0, left_first contains the index of the l child node (r child = l child + 1).
  // Otherwise, it contains the index of the first triangle index.
	int left_first; 
} BVHNode;


class BVH {
  public:
    BVH(class Mesh *mesh);
    void build();
    int nodes_used = 0;

  private:
    void update_node_bounds(uint32_t node_idx);
    void subdivide(uint32_t node_idx);

    BVHNode *nodes;
    Mesh *mesh;

    // BVHNode leaves reference slice of indices which point to triangles stored by the mesh.
    uint32_t *tri_indices;

};