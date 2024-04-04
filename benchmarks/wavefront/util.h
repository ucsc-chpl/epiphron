#pragma once

#include <stdint.h>
#include <algorithm>

// aligned memory allocations
#ifdef _MSC_VER
#define ALIGN( x ) __declspec( align( x ) )
#define MALLOC64( x ) ( ( x ) == 0 ? 0 : _aligned_malloc( ( x ), 64 ) )
#define FREE64( x ) _aligned_free( x )
#else
#define ALIGN( x ) __attribute__( ( aligned( x ) ) )
#define MALLOC64( x ) ( ( x ) == 0 ? 0 : aligned_alloc( 64, ( x ) ) )
#define FREE64( x ) free( x )
#endif
#if defined(__GNUC__) && (__GNUC__ >= 4)
#define CHECK_RESULT __attribute__ ((warn_unused_result))
#elif defined(_MSC_VER) && (_MSC_VER >= 1700)
#define CHECK_RESULT _Check_return_
#else
#define CHECK_RESULT
#endif


typedef struct float2 {
	float x;
	float y; 
} float2;

typedef struct alignas(16) float3 {
	union {
		struct {
			float x;
			float y;
			float z;
			float padding; // Padding to match 16-byte alignment.
		};
		float components[4]; // Array for indexed access.
	};
	float& operator [] (const int n ) { return components[n]; }
} float3;

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