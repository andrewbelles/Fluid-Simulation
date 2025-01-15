#ifndef __SPATIAL_LOOKUP_H__
#define __SPATIAL_LOOKUP_H__

// C++ check already in headers 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "c_vec.h"
#include "hash.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Headers */
#include "headers.h"
#include "definitions.h"
#include "fluid_structs.h"

/**
 * @brief Function pointer to verlet integration 
 */
typedef void (*verletFunc_) (integrateArgs_t*);

/**
 * @brief Copies constant values, necessary tables, and kernel functions to device
 */
void copyConstantsToDevice(Constants consts);

/**
 * @brief cubicSpline family of kernels for approximation
 */
//
__device__ float cubicSpline(float distance, float  smoothing_radius);
__device__ float gradCubicSpline(float  distance, float  smoothing_radius);
__device__ float laplacianCubicSpline(float  distance, float  smoothing_radius);

/**
 * @brief Host function to call GPU kernel to for efficient sorting of table
 */
void bitonicSort(uint32_table_t *table, uint32_t size);
__global__ void sortPairs(uint32_table_t *table, int j, int i, uint32_t size);

/**
 * @brief Host function to call GPU kernel to perform an efficient neighbor search
 */
void neighborSearch(searchArgs_t *h_args);
__global__ void processParticleSearch(searchArgs_t *args);

/**
 * @brief Performs threaded verlet integration to quickly update state of particles
 */
void verletIntegration(integrateArgs_t *h_args, verletFunc_ processParticleIntegrate);
__global__ void process1stParticleIntegrate(integrateArgs_t *args);
__global__ void process2ndParticleIntegrate(integrateArgs_t *args);

#ifdef __cplusplus
}
#endif 

#endif // __SPATIAL_LOOKUP_H__