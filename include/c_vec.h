#ifndef __C_VEC_H__
#define __C_VEC_H__

#include <cuda_runtime.h>

/* C limited vectors */

#ifdef __cplusplus
extern "C" {
#endif

/* Define functions as valid for host and device if on .cu file */
#ifdef __CUDACC__
  #define CUDA_TAG __host__ __device__ 
#else 
  #define CUDA_TAG 
#endif

/* C limited headers */
#include "headers.h"
#include "definitions.h"

typedef struct {
  int data[3];
} vec3_int;

typedef struct {
  float data[3];
} vec3_float;

/**
 * @brief Malloc with built in error handling 
 * @param size Size in uint64_t 
 */
static inline void *safe_malloc(size_t size) {
  void *ptr = malloc(size);
  if (ptr == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  // Return uncasted ptr 
  return ptr;
}

/**
 * @brief adds two type int 3D vectors
 */
CUDA_TAG
static inline vec3_int addVec3_int(vec3_int a, vec3_int b)
{
  return (vec3_int) {
    .data = {
      a.data[0] + b.data[0],
      a.data[1] + b.data[1],
      a.data[2] + b.data[2]
    }
  };
}

/**
 * @brief adds two type int 3D vectors
 */
CUDA_TAG
static inline vec3_int subtractVec3_int(vec3_int a, vec3_int b) {
  return (vec3_int) {
    .data = {
      a.data[0] - b.data[0],
      a.data[1] - b.data[1],
      a.data[2] - b.data[2]
    }
  };
}

/**
 * @brief adds two type float 3D vectors
 */
CUDA_TAG
static inline vec3_float addVec3_float(vec3_float a, vec3_float b) {
  return (vec3_float) {
    .data = {
      a.data[0] + b.data[0],
      a.data[1] + b.data[1],
      a.data[2] + b.data[2]
    }
  };
}

/**
 * @brief subtracts two type float 3D vectors
 */
CUDA_TAG
static inline vec3_float subtractVec3_float(vec3_float a, vec3_float b) {
  return (vec3_float) {
    .data = {
      a.data[0] - b.data[0],
      a.data[1] - b.data[1],
      a.data[2] - b.data[2]
    }
  };
}

/**
 * @brief scales type float 3D vector by scalar
 */
CUDA_TAG
inline vec3_float scaleVec3_float(vec3_float a, float scalar) {
 return (vec3_float) {
    .data = {
      a.data[0] * scalar,
      a.data[1] * scalar,
      a.data[2] * scalar
    }
  };
}

/**
 * @brief scales type int 3D vector by integer scalar 
 */
CUDA_TAG
inline vec3_int scaleVec3_int(vec3_int a, int scalar) {
 return (vec3_int) {
    .data = {
      a.data[0] * scalar,
      a.data[1] * scalar,
      a.data[2] * scalar
    }
  };
}

/**
 * @brief finds magnitude of type float vector 
 */
CUDA_TAG
inline float magnitudeVec3_float(vec3_float a) {
  return sqrt(a.data[0] * a.data[0] + a.data[1] * a.data[1] + a.data[2] * a.data[2]);
}

/* End of C limited code*/

#ifdef __cplusplus 
} 

/* Function overloads for c typed structs vec3_int and vec3_float */

/**
 * @brief Overload for gt operator for vec3_int
 */
CUDA_TAG
inline bool operator>(const vec3_int& a, const int value) {
  return (a.data[0] > value) || (a.data[1] > value) || (a.data[2] > value);
}

/**
 * @brief Overload for lt operator for vec3_int
 */
CUDA_TAG
inline bool operator<(const vec3_float& a, const int value) {
  return (a.data[0] < value) || (a.data[1] < value) || (a.data[2] < value);
}

CUDA_TAG
inline bool operator>(const vec3_float& a, const int value) {
  return (a.data[0] > value) || (a.data[1] > value) || (a.data[2] > value);
}

/**
 * @brief Overload for lt operator for vec3_int
 */
CUDA_TAG
inline bool operator<(const vec3_int& a, const int value) {
  return (a.data[0] < value) || (a.data[1] < value) || (a.data[2] < value);
}

/**
 * @brief Overload for ge operator for vec3_int
 */
CUDA_TAG
inline bool operator>=(const vec3_int& a, const int value) {
  return (a.data[0] >= value) || (a.data[1] >= value) || (a.data[2] >= value);
}

/**
 * @brief Overload for le operator for vec3_int
 */
CUDA_TAG
inline bool operator<=(const vec3_int& a, const int value) {
  return (a.data[0] <= value) || (a.data[1] <= value) || (a.data[2] <= value);
}

/**
 * @brief Overload for addition operator for vec3_float type
 */
CUDA_TAG
inline vec3_float operator+(const vec3_float& a, const vec3_float& b) {
  return (vec3_float) {
    .data = {a.data[0] + b.data[0], a.data[1] + b.data[1], a.data[2] + b.data[2]}
  };
}

/**
 * @brief Overload for subtration operator for vec3_float type 
 */
CUDA_TAG
inline vec3_float operator-(const vec3_float& a, const vec3_float& b) {
  return (vec3_float) {
    .data = {a.data[0] - b.data[0], a.data[1] - b.data[1], a.data[2] - b.data[2]}
  };
}

/**
 * @brief Overload for -= operator for vec3_float type
 */
CUDA_TAG
inline vec3_float operator-=(vec3_float& a, const vec3_float& b) {
  for (int i = 0; i < 3; ++i) {
    a.data[i] -= b.data[i];
  } 
  return a;
}

/**
 * @brief Overload for += operator for vec3_float type
 */
CUDA_TAG
inline vec3_float operator+=(vec3_float& a, const vec3_float& b) {
  for (int i = 0; i < 3; ++i) {
    a.data[i] += b.data[i];
  } 
  return a;
}

/**
 * @brief Overload for multiplication operator for vec3_float type 
 */
CUDA_TAG
inline vec3_float operator*(const vec3_float& a, const float value) {
  return (vec3_float) {
    .data = {a.data[0] * value, a.data[1] * value, a.data[2] * value}
  };
}

/**
 * @brief Overload for multiplication operator for vec3_float type 
 */
CUDA_TAG
inline vec3_float operator/(const vec3_float& a, const float value) {
  return (vec3_float) {
    .data = {a.data[0] / value, a.data[1] / value, a.data[2] / value}
  };
}

/**
 * @brief Overload for multiplication operator for vec3_float type 
 */
CUDA_TAG
inline vec3_float operator/=(vec3_float& a, const float value) {
 for (int i = 0; i < 3; ++i) {
    a.data[i] /= value;
  } 
  return a;
}
#endif

#endif // __C_VEC_H__
