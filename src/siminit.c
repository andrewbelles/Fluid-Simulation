#include "siminit.h"

// Global declaration of particles and system constants 
fparticle_t *particles;
Constants consts;

/**
 * @brief Function to initialize particles array and set system params
 * @param cube_size Size of cube at time of initialization
 * @param particle_count Number of simulated particles
 * @param h Smoothing radius for kernel 
 */
void startSim(float cube_size, const uint32_t particle_count, float h) {
  // Local Vars 
  float partition_length = h, max = 0.0, min = 0.0;
  int axis_count = (int)floor(cube_size / partition_length);
  // printf("Axis Count: %d\n", axis_count);

  // Set the parameter values for system 
  consts.cube_size        = cube_size; 
  consts.h                = h;
  consts.particle_count   = particle_count;
  consts.axis_count       = axis_count;
  consts.partition_count  = axis_count * axis_count * axis_count;
  consts.partition_length = partition_length;

  printf("partition_count: %u\n", consts.partition_count);

  // printf("Partition Count: %u\n", consts.partition_count);

  // Copy consts to device 
  (void)copyConstantsToDevice(consts);

  // Initialize particle array 
  max = cube_size - (cube_size / 8.0);
  min = (cube_size / 8.0);
  particles = (fparticle_t*)safe_malloc(particle_count * sizeof(fparticle_t));
  for (uint32_t particle = 0; particle < particle_count; ++particle) {
    particles[particle] = setParticle(min, max);
  }
}