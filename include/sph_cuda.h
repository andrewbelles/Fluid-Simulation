#ifndef __SPH_CUDA_H__
#define __SPH_CUDA_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "c_vec.h"
#include "hash.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Functions related to creating lookup table and its logic 
 */

__host__ void makeSpatialLookup();
__host__ void bitonicMerge();

/**
 * @brief Kernel Function families available in simulation
 */

__host__ __device__ float cubicSpline(float distance, float smoothing_radius);
__host__ __device__ float gradSpline(float distance, float smoothing_radius);
__host__ __device__ float laplaceSpline(float distance, float smoothing_radius);

/**
 * @brief Functions related to iterating over volume of cube
 * 
 * neighborSearch is a 3x3x3 grid search
 * enforceBounds is a threaded simultaneous check
 */

__host__ void neighborSearch();
__host__ void enforceBounds();

/**
 * @brief Copies a position array back to the host
 */


#ifdef __cplusplus
}
#endif

#endif // __SPH_CUDA_H__