#ifndef __FLUID_STRUCTS_H__
#define __FLUID_STRUCTS_H__

#include "headers.h"
#include "c_vec.h"

/**
 * @brief Holds parameter values to be accessed by simulation
 */
typedef struct {
  int axis_count;
  uint32_t particle_count;
  uint32_t partition_count;
  float partition_length;
  float cube_size;
  float h;
} Constants;

/**
 * @brief Simple table structure to hold particle id and cell key 
 */
typedef struct {
  uint32_t particle_id;
  uint32_t cell_key;
} uint32_table_t;

/**
 * @brief Fluid Particle which stores information on each particle's neighbors,
 * density, pressure, position, and velocity
 */
typedef struct {
  vec3_float position;
  vec3_float velocity;   
  double mass;
  double density;
  double pressure;
  vec3_float pressure_force;
  vec3_float viscosity_force;
  vec3_int wall;
} fparticle_t;              

typedef struct {
  fparticle_t *particles;
  uint32_table_t *table;
  uint32_t *start_cell;
  uint32_t *end_cell;
  uint32_t size;
} searchArgs_t;

typedef struct {
  fparticle_t *particles;
  uint32_t size;
} integrateArgs_t;

#endif // __FLUID_STRUCTS_H__