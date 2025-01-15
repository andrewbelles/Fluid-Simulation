#ifndef __HASH_H__
#define __HASH_H__

#include "headers.h"
#include "definitions.h"
#include "c_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Define functions as valid for host and device if on .cu file */
#ifdef __CUDACC__
  #define CUDA_TAG __host__ __device__ 
#else 
  #define CUDA_TAG 
#endif

/**
 * @brief Converts integer cell coordinates to hash value
 * @param position 3D position in simulation 
 * @param partition_count Number of grid cells 
 */
static inline CUDA_TAG uint32_t
hashFunction(vec3_int cell_coords, uint32_t partition_count)
{
 // Scale by large relatively co-prime values
  uint32_t x_value = cell_coords.data[0] * 73856093;
  uint32_t y_value = cell_coords.data[1] * 19349663;
  uint32_t z_value = cell_coords.data[2] * 83492791;

  // Bitwise XOR and modulus to wrap to table size
  uint32_t hash = (x_value ^ y_value ^ z_value); 
  // printf("partition_count: %u\n", partition_count);
  hash %= partition_count;
  // printf("hash: %u\n", hash);
  return hash;
}

/**
 * @brief Converts float position to cell coordinate position
 */
static inline CUDA_TAG vec3_int
positionToID(vec3_float position, float partition_length)
{
  return (vec3_int) {
    .data = {
      (int)floor(position.data[0] / partition_length),
      (int)floor(position.data[1] / partition_length),
      (int)floor(position.data[2] / partition_length)
    }
  };
}

#ifdef __cplusplus
}
#endif

#endif // __HASH_H__