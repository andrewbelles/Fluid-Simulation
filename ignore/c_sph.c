#include "c_sph.h"

/**
 * @brief Global struct of constant simulation values
 */
static Constants consts;
fparticle_t *particles;

/**
 * @brief Non-threaded C function to initialize the fluid particles positions and etc. 
 * @param All constant values needed for simulation to run
 */
fparticle_t *initializeFluidSimulation_(const double cube_size, const int particle_count, const double dt, const double h) {
  particles = (fparticle_t*)safe_malloc(particle_count * sizeof(fparticle_t));
  double partition_length;
  int axis_size;

  // Set constant values according to simulation design parameters 
  consts.cube_size = cube_size;
  consts.particle_count = particle_count;
  consts.k = 3000.0;
  consts.rho0 = 1000.0;
  consts.beta = 0.5;
  consts.viscosity = 0.5;
  consts.dt = dt;
  consts.particle_spacing = consts.cube_size / cbrt(consts.particle_count);
  consts.h = h;;
  consts.standard = consts.particle_spacing * consts.beta;
  consts.mean = consts.cube_size / 2.0;

  // Find number of partitions to start simulation with
  partition_length = 2.0 * consts.h;
  axis_size = floor(consts.cube_size / partition_length);
  axis_size = (axis_size < 8) ? 8 : axis_size;
  consts.axis_count = (axis_size % 2) ? axis_size - 1 : axis_size;
  consts.partition_count = pow(consts.axis_count, 3);

  for (int particle = 0; particle < particle_count; ++particle) {
    particles[particle].wall = (vec3_int) {
      .data = {0, 0, 0}
    };
    particles[particle].density = 0.0; 
    particles[particle].mass = 1.0; 
    particles[particle].pressure = 0.0;

    
  }
}