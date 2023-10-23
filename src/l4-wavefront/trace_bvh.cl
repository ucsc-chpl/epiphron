// Credit for the BVH code: https://jacco.ompf2.com/2022/06/03/how-to-build-a-bvh-part-9a-to-the-gpu/
#define MIN_T 0.01f
#define MAX_T 1e30f
#define FOV_DEGREES 90.0f
#define PI 3.14159265359f

typedef struct Intersection {
    float t;       // Intersection distance along ray
    float u, v;    // Barycentric coordinates of the intersection.
    uint inst_prim; // Instance index (12 bit) and primitve index (20 bit).
} Intersection;

typedef struct Ray {
    float3 O, D, rD; // In OpenCL, each of these will be padded to 16 bytes
    Intersection hit; // Total ray size: 64 bytes.
} Ray;

typedef struct Tri {
	float v0x, v0y, v0z, dummy1;
	float v1x, v1y, v1z, dummy2;
	float v2x, v2y, v2z, dummy3;
	float cx, cy, cz, dummy4;
} Tri;

typedef struct TriEx {
	float2 uv0, uv1, uv2;
	float N0x, N0y, N0z;
	float N1x, N1y, N1z;
	float N2x, N2y, N2z;
	float dummy;
} TriEx;

// 32 bytes
typedef struct BVHNode {
	float minx, miny, minz;
	float maxx, maxy, maxz;
	int left_first;
	int tri_count;
} BVHNode;

typedef struct TLASNode {
	float minx, miny, minz;
	uint left_right; // 2x16 bits
	float maxx, maxy, maxz;
	uint BLAS;
} TLASNode;

typedef struct BVHInstance {
	float16 transform;
	float16 invTransform; // inverse transform
	uint dummy[16];
} BVHInstance;

static uint2 indexToPosition(uint i, uint width) {
    uint x = i % width;
    uint y = i / width;
    return (uint2) (x, y);
}

static void write_color(uint index, __global uint *image, float3 col) {
    // Color is layed out in memory as ABGR, so MSB -> alpha and LSB-> red.
    // A float3 represents a normalized rgb color col.xyz -> rgb.
    uint alpha = 255;
    uint blue = clamp(convert_uint(col.z * 255.0f), 0u, 255u);
    uint green = clamp(convert_uint(col.y * 255.0f), 0u, 255u);
    uint red = clamp(convert_uint(col.x * 255.0f), 0u, 255u);
    image[index] = red | (green << 8) | (blue << 16) | (alpha << 24);
}

static void intersect_tri(Ray* ray, __global Tri* tri, const uint inst_prim) {
	float3 v0 = (float3)(tri->v0x, tri->v0y, tri->v0z);
	float3 v1 = (float3)(tri->v1x, tri->v1y, tri->v1z);
	float3 v2 = (float3)(tri->v2x, tri->v2y, tri->v2z);
	float3 edge1 = v1 - v0, edge2 = v2 - v0;
	float3 h = cross( ray->D, edge2 );
	float a = dot( edge1, h );
	if (fabs( a ) < 0.00001f) return; // ray parallel to triangle
	float f = 1 / a;
	float3 s = ray->O - v0;
	float u = f * dot( s, h );
	if (u < 0 | u > 1) return;
	const float3 q = cross( s, edge1 );
	const float v = f * dot( ray->D, q );
	if (v < 0 | u + v > 1) return;
	const float t = f * dot( edge2, q );
	if (t > 0.0001f && t < ray->hit.t) {
		ray->hit.t = t, ray->hit.u = u,
		ray->hit.v = v, ray->hit.inst_prim = inst_prim;
    }
}

static float intersect_AABB(Ray* ray, __global BVHNode* node) {
	float tx1 = (node->minx - ray->O.x) * ray->rD.x, tx2 = (node->maxx - ray->O.x) * ray->rD.x;
	float tmin = min( tx1, tx2 ), tmax = max( tx1, tx2 );
	float ty1 = (node->miny - ray->O.y) * ray->rD.y, ty2 = (node->maxy - ray->O.y) * ray->rD.y;
	tmin = max( tmin, min( ty1, ty2 ) ), tmax = min( tmax, max( ty1, ty2 ) );
	float tz1 = (node->minz - ray->O.z) * ray->rD.z, tz2 = (node->maxz - ray->O.z) * ray->rD.z;
	tmin = max( tmin, min( tz1, tz2 ) ), tmax = min( tmax, max( tz1, tz2 ) );
	if (tmax >= tmin && tmin < ray->hit.t && tmax > 0) return tmin; else return 1e30f;
}

static void BVH_intersect(Ray *ray,     
                          uint instance_idx,
                          __global Tri *tri, 
                          __global BVHNode *bvh_node, 
                          __global uint *tri_idx) {
    // Root of the BVH is at index 0.
    uint node_idx = 0; 
    // Initialize a stack to keep track of BVH nodes that still need to be traversed.
    uint stack[64];
    uint stackPtr = 0;
    while (1) {
        // Get the node pointer using the index.
        __global BVHNode *node = &bvh_node[node_idx];
        if (node->tri_count > 0) {
            // Node is a leaf.
            // Loop through every triangle at this leaf.
            for (uint i = 0; i < node->tri_count; i++) {
                // Compute the unique identifier for this instance + primitive combination.
                uint inst_prim = (instance_idx << 20) + tri_idx[node->left_first + i];
                intersect_tri( ray, &tri[inst_prim & 0xfffff /* 20 bits */], inst_prim);
            }
            // Pop the next node to examine from the stack, or break if stack is empty.
            if (stackPtr == 0) {
                break;
             } else{  
                node_idx = stack[--stackPtr];
            }
            continue;
        }
         // The node is an internal node, not a leaf. Get indices of the left and right children.
        uint child1_idx = node->left_first;
        uint child2_idx = node->left_first + 1;

        float dist1 = intersect_AABB( ray, &bvh_node[child1_idx]);
        float dist2 = intersect_AABB( ray, &bvh_node[child2_idx]);
        if (dist1 > dist2) { 
            float d = dist1; 
            dist1 = dist2; 
            dist2 = d;
            uint c = child1_idx; 
            child1_idx = child2_idx; 
            child2_idx = c; 
        }
        if (dist1 == 1e30f) {
            // If closest node is too far away, traverse up the tree.
            if (stackPtr == 0) { 
                break;
            } else { 
                node_idx = stack[--stackPtr]; 
            }
        } else {  
            // Traverse to closest child.
            node_idx = child1_idx;
            // If the second closest child is also within reach, push it onto the stack for later.
            if (dist2 != 1e30f) { 
                stack[stackPtr++] = child2_idx;
            }
        }
    }
}


static float3 trace(Ray *ray, __global Tri *tri_data, __global BVHNode *bvh_node_data, __global uint *idx_data) {
    BVH_intersect(ray, 0, tri_data, bvh_node_data, idx_data);
    if (ray->hit.t < MAX_T) {
        return (float3)(1, 1, 1);
    } 
    return (float3)(0, 0, 0);
}

__kernel void render(__global uint *image_buffer,
                     __global uint *image_buf_width,
                     __global uint *image_buf_height,
                     __global Tri *tri_data,
                     __global BVHNode *bvh_node_data,
                     __global uint *idx_data) {

    uint idx = get_global_id(0);
    if (idx >= image_buf_width[0] * image_buf_height[0]) {
        return;
    }

    // Get the pixel (x, y) integer coordinate into the image buffer that this thread is responsible for.
    // Convert that to a (u, v) normalized coordinate, where (0, 0) is bottom left and (1, 1) is top right.
    uint2 pixel_coord = indexToPosition(idx, image_buf_width[0]);
    float2 uv = convert_float2(pixel_coord) / (float2)(convert_float(image_buf_width[0]), convert_float(image_buf_height[0]));
    uv.y = 1.0f - uv.y;

    // Create primary ray.
    float focal_length = 1.0f / tan(FOV_DEGREES * 0.5f * PI / 180.0f);
    Ray ray;
    ray.O = (float3) (0.f, 0.f, 0.f); // Camera pos
    // calculate coordinates of the ray target on the imaginary pixel plane.
    // -1 to +1 on x,y axis. 1 unit away on the z axis
    float3 ray_target = (float3) (uv * 2.0f - 1.0f, focal_length);
    // Correct for aspect ratio.
    float aspectRatio = convert_float(image_buf_width[0]) / convert_float(image_buf_height[0]);
    ray_target.y /= aspectRatio;
    ray.D = normalize(ray_target - ray.O); 

    // Tracep primary ray.
    float3 col = trace(&ray, tri_data, bvh_node_data, idx_data);
    write_color(idx, image_buffer, col);
}

