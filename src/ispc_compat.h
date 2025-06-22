#ifndef ISPC_COMPAT_H
#define ISPC_COMPAT_H

// ISPC compatibility header for Zig
// Provides definitions needed for ISPC headers to work with Zig C import

#include <stdint.h>
#include <stdbool.h>

// Define ISPC alignment macros for Zig compatibility
#ifndef __ISPC_ALIGN__
#if defined(__clang__) || !defined(_MSC_VER)
// Clang, GCC, ICC
#define __ISPC_ALIGN__(s) __attribute__((aligned(s)))
#define __ISPC_ALIGNED_STRUCT__(s) struct __attribute__((aligned(s)))
#else
// Visual Studio
#define __ISPC_ALIGN__(s) __declspec(align(s))
#define __ISPC_ALIGNED_STRUCT__(s) __declspec(align(s)) struct
#endif
#endif

// Provide struct___ISPC_ALIGN__ macro for Zig compatibility
#define struct___ISPC_ALIGN__(s) struct __attribute__((aligned(s)))

// Common ISPC types for Zig
typedef struct {
    float x, y, z, w;
} ispc_float4;

typedef struct {
    int32_t x, y, z, w;
} ispc_int4;

#endif // ISPC_COMPAT_H