#pragma once 

#include "util.h"
#include "bvh.h"

class Mesh {
public:
	Mesh() = default;
	// Mesh(uint32_t primCount);
	Mesh(const char* objFile, const char* texFile);
	Tri* tri = 0;			// triangle data for intersection
	TriEx* triEx = 0;		// triangle data for shading
	int triCount = 0;
	class BVH* bvh;
	float3* P = 0, * N = 0;
};