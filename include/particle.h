#ifndef __PARTICLE_H__
#define __PARTICLE_H__


#include "headers.h"
#include "definitions.h"
#include "fluid_structs.h"

#include "c_vec.h"

/**
 * @brief Function that generateas a random value in a given range 
 * 
 * Credit to chat-GPT o1 model 
 */
static inline float randomInRange(float min, float max) {
  // Generate a float between 0.0 and 1.0
  float scale = rand() / (float)RAND_MAX;
  // Scale and shift to the desired range
  return min + scale * (max - min);
}

/**
 * @brief Creates a new initialized particle
 */
static inline fparticle_t setParticle(float min, float max) {
  fparticle_t new_particle;
  vec3_float random_vec[2], zero = (vec3_float) {{0.0, 0.0, 0.0}};


  for (int i = 0; i < 3; ++i) {
    random_vec[0].data[i] = randomInRange(min, max);
    random_vec[1].data[i] = randomInRange(0.0, 1.0);
  }

  // Set each individual particles initial state 
  new_particle.position        = random_vec[0];
  new_particle.velocity        = random_vec[1];
  new_particle.pressure_force  = zero;
  new_particle.viscosity_force = zero;
  new_particle.density         = 1.0;
  new_particle.pressure        = 0.0;
  new_particle.mass            = 0.25;
  new_particle.wall            = (vec3_int) {.data = {0,0,0}};

  return new_particle;
}

#endif // __PARTICLE_