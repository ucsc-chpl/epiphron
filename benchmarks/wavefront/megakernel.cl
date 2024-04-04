#define MIN_T 0.01f
#define MAX_T 10000.0f
#define FOV_DEGREES 90.0f
#define PI 3.14159265359f


typedef struct Sphere {
    float3 pos;
    float r;
} Sphere;

typedef struct Triangle {
    float pos[3];
    float normal[3];
} Triangle;

typedef struct Scene {
    __global Sphere *spheres;
    uint num_spheres; 
    __global Triangle *triangles;
    uint num_triangles;
} Scene;

typedef struct HitInfo {
    float3 normal;
    float t;
} HitInfo;


static uint2 indexToPosition(uint i, uint width, uint height) {
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
 

static bool intersectSphere(float3 ray_pos, 
                            float3 ray_dir, HitInfo *hit, 
                            const __global Sphere *sphere) {
    // Get the vector from the center of the sphere to the ray origin.
    float3 m = ray_pos - sphere->pos;
    float b = dot(m, ray_dir);
    float c = dot(m, m) - sphere->r * sphere->r;

    // No hit if the ray's origin is outside the sphere (c > 0) and 
    // the ray is pointing away from the sphere (b > 0)
    if (c > 0.0f && b > 0.0f) {
        return false;
    }

    // A negative discriminant corresponds to the ray missing the sphere.
    float discr = b * b - c;
    if (discr < 0.0f) {
        return false;
    }

    // Ray intersects the sphere, compute smallest t value of intersection.
    bool from_inside = false;
    float dist = -b - sqrt(discr);
    if (dist < 0.0f) {
        from_inside = true;
        dist = -b + sqrt(discr);
    }

    if (dist > MIN_T && dist < hit->t) {
        hit->t = dist;
        hit->normal = normalize((ray_pos + ray_dir * dist) - sphere->pos) * (from_inside ? -1.0f : 1.0f);
        return true;
    }

    return false;
}


static float3 trace_ray(float3 ray_pos, float3 ray_dir, Scene *scene) {
    HitInfo closest_hit;
    closest_hit.t = MAX_T;

    float3 col = (float3) (0.0f, 0.0f, 0.0f);

    bool hit = false; 
    for (uint i = 0; i < scene->num_spheres; i++) {
        hit |= intersectSphere(ray_pos, ray_dir, &closest_hit, &scene->spheres[i]);
    }

    if (hit) {
        col = (float3) (1.0f, 0.0f, 0.0f);
    }
    
    return col;
}


__kernel void render(__global uint *image_buffer,
                     __global uint *image_buf_width,
                     __global uint *image_buf_height,
                     __global Sphere *spheres,
                     __global uint *num_spheres,
                     __global Triangle *triangles,
                     __global uint *num_triangles) {

    uint idx = get_global_id(0);
    if (idx >= image_buf_width[0] * image_buf_height[0]) {
        return;
    }

    // Get the pixel (x, y) integer coordinate into the image buffer that this thread is responsible for.
    // Convert that to a (u, v) normalized coordinate, where (0, 0) is bottom left and (1, 1) is top right.
    uint2 pixel_coord = indexToPosition(idx, image_buf_width[0], image_buf_height[0]);
    float2 uv = convert_float2(pixel_coord) / (float2)(convert_float(image_buf_width[0]), convert_float(image_buf_height[0]));
    uv.y = 1.0f - uv.y;

    float focal_length = 1.0f / tan(FOV_DEGREES * 0.5f * PI / 180.0f);

    // The initial view ray starts from the camera position (origin).
    float3 ray_origin = (float3) (0.0f, 0.0f, 0.0f);
    // calculate coordinates of the ray target on the imaginary pixel plane.
    // -1 to +1 on x,y axis. 1 unit away on the z axis
    float3 ray_target = (float3) (uv * 2.0f - 1.0f, focal_length);

    // Correct for aspect ratio.
    float aspectRatio = convert_float(image_buf_width[0]) / convert_float(image_buf_height[0]);
    ray_target.y /= aspectRatio;

    // The direction of the view ray from the camera origin to the image plane.
    float3 ray_dir = normalize(ray_target - ray_origin);

    // Init scene.
    Scene scene;
    scene.spheres = spheres;
    scene.num_spheres = *num_spheres;
    scene.triangles = triangles;
    scene.num_triangles = *num_triangles;

    float3 col = trace_ray(ray_origin, ray_dir, &scene);
    write_color(idx, image_buffer, col);
}
