#include "mesh.h"

#include <cstring>
#include <iostream>

#define MAX_TRIS 1024 * 32

// Mesh::Mesh(const uint32_t primCount) {
// 	// basic constructor, for top-down TLAS construction
// 	tri = (Tri*)_aligned_malloc( primCount * sizeof( Tri ), 64 );
// 	memset( tri, 0, primCount * sizeof( Tri ) );
// 	triEx = (TriEx*)_aligned_malloc( primCount * sizeof( TriEx ), 64 );
// 	memset( triEx, 0, primCount * sizeof( TriEx ) );
// 	triCount = primCount;
// }

// Bare-bones obj file loader; only supports very basic meshes
Mesh::Mesh( const char* objFile, const char* texFile ) {
	// Allocate memory for triangles and extended triangle information.
	tri = new Tri[MAX_TRIS];
	triEx = new TriEx[MAX_TRIS];

	// Allocate memory for UV coords, normals, and vertices.
	float2* UV = new float2[MAX_TRIS * 3];
	N = new float3[MAX_TRIS * 3];
	P = new float3[MAX_TRIS * 3];

	// Variables to keep track of UVs, normals, and vertices.
	int UVs = 0, Ns = 0, Ps = 0;

	// Variables to store indices from the OBJ file.
	int a, b, c, d, e, f, g, h, i;

	FILE* file = fopen( objFile, "r" );
	if (!file) {
		std::cerr << "Failed to open obj file!\n";
		return;
	}
	while (!feof( file )) {
		char line[512] = { 0 };
		fgets( line, 511, file );
		if (line == strstr( line, "vt " )) {
			// Get uv coords.
			sscanf( line + 3, "%f %f", &UV[UVs].x, &UV[UVs].y ), UVs++;
		} else if (line == strstr( line, "vn " )) {
			sscanf( line + 3, "%f %f %f", &N[Ns].x, &N[Ns].y, &N[Ns].z ), Ns++;
		} else if (line[0] == 'v') {
			sscanf( line + 2, "%f %f %f", &P[Ps].x, &P[Ps].y, &P[Ps].z ), Ps++;
		}	
		if (line[0] != 'f') {
			continue; 
		} else {
			sscanf( line + 2, "%i/%i/%i %i/%i/%i %i/%i/%i",
				&a, &b, &c, &d, &e, &f, &g, &h, &i );
		}

		// Set vertex coordinates for the triangle
		tri[triCount].v0x = P[a - 1].x, tri[triCount].v0y = P[a - 1].y, tri[triCount].v0z = P[a - 1].z;
		tri[triCount].v1x = P[d - 1].x, tri[triCount].v1y = P[d - 1].y, tri[triCount].v1z = P[d - 1].z;
		tri[triCount].v2x = P[g - 1].x, tri[triCount].v2y = P[g - 1].y, tri[triCount].v2z = P[g - 1].z;

		// Set normal coordinates for the triangle
		triEx[triCount].N0x = N[c - 1].x, triEx[triCount].N0y = N[c - 1].y, triEx[triCount].N0z = N[c - 1].z;
		triEx[triCount].N1x = N[f - 1].x, triEx[triCount].N1y = N[f - 1].y, triEx[triCount].N1z = N[f - 1].z;
		triEx[triCount].N2x = N[i - 1].x, triEx[triCount].N2y = N[i - 1].y, triEx[triCount].N2z = N[i - 1].z;

		// Set UV coordinates for the triangle
		triEx[triCount].uv0 = UV[b - 1], triEx[triCount].uv1 = UV[e - 1], triEx[triCount].uv2 = UV[h - 1];

		// Increment the triangle count
		triCount++;

		if (triCount >= MAX_TRIS) {
			std::cerr << "Max triangle limit reached while creating mesh!\n";
		}
	}
	fclose( file );
	bvh = new BVH(this);
}