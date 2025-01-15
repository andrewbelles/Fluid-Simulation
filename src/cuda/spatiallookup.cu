
#include "spatiallookup.h"

/**
 * @brief Quick offset reference table to avoid triple nested loop in 3x3x3 iterator
 */
static vec3_int offset_table[27] = {
  {{-1, -1, -1}},
  {{-1, -1,  0}},
  {{-1, -1,  1}},
  {{-1,  0, -1}},
  {{-1,  0,  0}},
  {{-1,  0,  1}},
  {{-1,  1, -1}},
  {{-1,  1,  0}},
  {{-1,  1,  1}},
  {{ 0, -1, -1}},
  {{ 0, -1,  0}},
  {{ 0, -1,  1}},
  {{ 0,  0, -1}},
  {{ 0,  0,  0}},  
  {{ 0,  0,  1}},
  {{ 0,  1, -1}},
  {{ 0,  1,  0}},
  {{ 0,  1,  1}},
  {{ 1, -1, -1}},
  {{ 1, -1,  0}},
  {{ 1, -1,  1}},
  {{ 1,  0, -1}},
  {{ 1,  0,  0}},
  {{ 1,  0,  1}},
  {{ 1,  1, -1}},
  {{ 1,  1,  0}},
  {{ 1,  1,  1}},
};

/**
 * Externally defined parameters describing system
 */
extern Constants consts;
__constant__ Constants d_consts; 
__constant__ vec3_int d_offset_table[27];

/**
 * @brief Cubic Spline Kernel
 * @param distance Magnitude of displacement vector between two particles
 * @param smoothing_radius Constant smoothing radius to determine influence between particles
 */
__device__ float cubicSpline(float distance, float  smoothing_radius) {
  // Constant values
  const float  q = distance / smoothing_radius;
  const float  a3 = 1.0 / (M_PI * smoothing_radius * smoothing_radius * smoothing_radius);
  float  value = a3;

  // Calcuate value of kernel over the smoothing radius
  if (q >= 0 && q < 1) {
    value *= (1.0 - (1.5 * q * q) + 0.75 * q * q * q);
  } else if (q >= 1 && q < 2) {
    value *= (0.25 * (2.0 - q) * (2.0 - q) * (2.0 - q)); 
  // Outside influence
  } else if (q >= 2) {
    value = 0;
  }
  return value;
}

/**
 * @brief Gradient of the Cubic Spline Kernel
 * @param distance Magnitude of displacement vector between two particles
 * @param smoothing_radius Constant smoothing radius to determine influence between particles
 */
__device__ float gradCubicSpline(float  distance, float  smoothing_radius) {
  const float  q = distance / smoothing_radius;
  const float  a3 = 1.0 / (M_PI * smoothing_radius * smoothing_radius * smoothing_radius);
  float  value = a3;

  // Calculate the gradient of the kernel over the smoothing radius
  if (q >= 0 && q < 1) {
    value *= (-3.0 * q + 2.25 * q * q);
  } else if (q >= 1 && q < 2) {
    value *= (-0.75 * (2.0 - q) * (2.0 - q));
  // Outside influence
  } else if (q >= 2) {
    value = 0;
  }
  return value;
}

/**
 * @brief The Laplacian of the Cubic Spline Kernel
 * @param distance Magnitude of displacement vector between two particles
 * @param smoothing_radius Constant smoothing radius to determine influence between particles
 */
__device__ float laplacianCubicSpline(float  distance, float  smoothing_radius) {
  const float  q = distance / smoothing_radius;
  const float  a3 = 1.0 / (M_PI * smoothing_radius * smoothing_radius * smoothing_radius);
  float  value = a3;

  // Calculate the laplacian of the kernel over the smoothing radius
  if (q >= 0 && q < 1) {
    value *= (-3.0 + 4.5 * q);
  } else if (q >= 1 && q < 2) {
    value *= (1.5 * (2.0 - q));
  // Outside incluence
  } else if (q >= 2) {
    value = 0;
  }
  return value;
}

/**
 * @brief Copies consts struct and offset table to GPU memory 
 */
void copyConstantsToDevice(Constants consts) {
  cudaMemcpyToSymbol(d_consts, &consts, sizeof(Constants));
  cudaMemcpyToSymbol(d_offset_table, offset_table, 27 * sizeof(vec3_int));
}

/**
 * @brief Array is sorted according to Bitonic Merge Sort algorithm with O(log2(n))
 * 
 * @param table Unsorted, Padded Array
 * @param size Padded size of array; some value 2^n
 */
__host__ void bitonicSort(uint32_table_t *table, uint32_t size) {
  // Determine the number of threads 
  int threadPerBlock = 512;
  int blocks = (size + threadPerBlock - 1) / threadPerBlock;

  // Create memory for table
  uint32_table_t *d_table;
  cudaMalloc(&d_table, size * sizeof(uint32_table_t));

  // Copy data to gpu array to be sorted 
  cudaMemcpy(d_table, table, size * sizeof(uint32_table_t), cudaMemcpyHostToDevice);

  // Iterator to ensure pairs are correctly sized 
  for (uint32_t i = 2; i <= size; i <<= 1) {
    for (uint32_t j = i >> 1; j > 0; j >>= 1) {
      // Call GPU kernel
      sortPairs<<<blocks, threadPerBlock>>>(d_table, j, i, size);
    }
  }

  // Copy sorted array onto src array
  cudaMemcpy(table, d_table, size * sizeof(uint32_table_t), cudaMemcpyDeviceToHost);

  // Free gpu sorted array
  cudaFree(d_table);
}

/**
 * @brief Kernel Function to sort lookup table by cell_key value. 
 * 
 * Credit to github user rgba for the relative structure of pair selectioon
 */
__global__ void sortPairs(uint32_table_t *table, int j, int i, uint32_t size) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int idxj = idx ^ j;
  uint32_table_t temp;
  if (idx >= size) return;

  if (idxj > idx) {
    if ((idx & i) == 0) {
      if (table[idx].cell_key > table[idxj].cell_key) {
        temp = table[idx];
        table[idx] = table[idxj];
        table[idxj] = temp;
      }
    } else {
      if (table[idx].cell_key < table[idxj].cell_key) {
        temp = table[idx];
        table[idx] = table[idxj];
        table[idxj] = temp;
      }
    }
  }
}

/**
 * @brief Allocate memory on device for the arguments for search
 */
__host__ static searchArgs_t *searchArgAllocate(searchArgs_t *h_args) {
  searchArgs_t *d_args = nullptr, temp;
  fparticle_t *d_particles;
  uint32_table_t *d_table;
  uint32_t *d_start_cell;
  uint32_t *d_end_cell;

  cudaMalloc(&d_args, sizeof(searchArgs_t));

  /* Allocate and copy individual memory for each pntr in searchArgs_t */

  cudaMalloc(&d_particles, h_args->size * sizeof(fparticle_t));
  cudaMemcpy(d_particles, h_args->particles, h_args->size * sizeof(fparticle_t), cudaMemcpyHostToDevice);

  cudaMalloc(&d_table, h_args->size * sizeof(uint32_table_t));
  cudaMemcpy(d_table, h_args->table, h_args->size * sizeof(uint32_table_t), cudaMemcpyHostToDevice);

  cudaMalloc(&d_start_cell, h_args->size * sizeof(uint32_t));
  cudaMemcpy(d_start_cell, h_args->start_cell, h_args->size * sizeof(uint32_t), cudaMemcpyHostToDevice);

  cudaMalloc(&d_end_cell, h_args->size * sizeof(uint32_t));
  cudaMemcpy(d_end_cell, h_args->end_cell, h_args->size * sizeof(uint32_t), cudaMemcpyHostToDevice);

  // Set arguments 
  temp.particles = d_particles;
  temp.table = d_table;
  temp.start_cell = d_start_cell;
  temp.end_cell = d_end_cell; 
  temp.size = h_args->size;

  // Copy temp argument holder to device arguments 
  cudaMemcpy(d_args, &temp, sizeof(searchArgs_t), cudaMemcpyHostToDevice);

  return d_args;
}

/**
 * @brief Allocate memory on the device for the arguments to integrate 
 */
__host__ static integrateArgs_t *integrateArgAllocate(integrateArgs_t *h_args) {
  integrateArgs_t *d_args= nullptr, temp;
  fparticle_t *d_particles;

  // Allocate Memory for argument pntr 
  cudaMalloc(&d_args, sizeof(integrateArgs_t));

  // Allocate memory for particles and copy from host 
  cudaMalloc(&d_particles, h_args->size * sizeof(fparticle_t));
  cudaMemcpy(d_particles, h_args->particles, h_args->size * sizeof(fparticle_t), cudaMemcpyHostToDevice);

  // Set arguments 
  temp.particles = d_particles;
  temp.size = h_args->size;

  // Copy temp argument holder to device arguments 
  cudaMemcpy(d_args, &temp, sizeof(integrateArgs_t), cudaMemcpyHostToDevice);

  // Return pointer
  return d_args;
}

/**
 * @brief CPU call to set the thread pool for the kernel
 */
__host__  void neighborSearch(searchArgs_t *h_args) {
  // Particle Hard Cap 2^19 ~= 520k particles
  int threadPerBlock = 512; 
  int gridSize = 2048;

  // Allocate memory in GPU for argument list 
  searchArgs_t *d_args = searchArgAllocate(h_args), result_ptr;

  // printf("searchArgs Set on Device");

  // Calls the GPU kernel to specified dimensions. Passes device args
  processParticleSearch<<<gridSize, threadPerBlock>>>(d_args);
  
  // printf("Particle Search Complete\n");

  // Capture modified array in d_args->particles to a temp struct on host size 
  cudaMemcpy(&result_ptr, d_args, sizeof(searchArgs_t), cudaMemcpyDeviceToHost);

  // Copy from temp struct to original host array
  cudaMemcpy(h_args->particles, result_ptr.particles, h_args->size * sizeof(fparticle_t), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(result_ptr.particles);
  cudaFree(result_ptr.table);
  cudaFree(result_ptr.start_cell);
  cudaFree(result_ptr.end_cell);
  cudaFree(d_args);
}

/**
 * @brief Kernel to calculate the density, pressure, and forces for each particle based on its neighbors
 */
__global__ void processParticleSearch(searchArgs_t *args) {
  // Set base args for id check 
  const uint32_t size = args->size; 

  // Thread id check
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= size) return;  // More threads than particles are launched.

  // Particle array to be acted upon 
  fparticle_t *particles = args->particles;
  const uint32_table_t *table = args->table;
  const uint32_t *start_cell = args->start_cell;
  const uint32_t *end_cell = args->end_cell;

  // uint and integer variables 
  uint32_t relative_hash, start, end, rel;
  vec3_int cell_coord, relative_coord;
  // Float variables
  vec3_float displacement, velocity_difference, intermediate_value;
  float distance = 0.0;

  cell_coord = positionToID(particles[idx].position, d_consts.h);

  for (int i = 0; i < 27; ++i) {
    // Set the relative coordinate by looking up the offset from the table
    relative_coord = addVec3_int(cell_coord, d_offset_table[i]);

    // If the relative coordinate is out of bounds continue to next iteration
    if (relative_coord < 0 || relative_coord >= d_consts.axis_count) {
      continue;
    }

    // Find relative hash value
    relative_hash = hashFunction(
      relative_coord,
      d_consts.partition_count
    );
    
    // Find start and stop indexes
    start = start_cell[relative_hash];
    end   = end_cell[relative_hash];

    // Skip empty bucket (UINT32_MAX is stored in empty buckets)
    if (start > d_consts.particle_count || end > d_consts.particle_count) {
      continue;
    }

    // Iterate over bucket
    for (int i = start; i < end; ++i) {
      rel = table[i].particle_id; 

      // Skip self if offset is center
      if (rel == idx) {
        continue;
      }

      /* SPH Value calculation for particle and its neighbor */

      // Find displacement vector and true distance between particles 
      displacement = particles[idx].position - particles[rel].position;
      distance = magnitudeVec3_float(displacement);

      // Check if relative partice is influencing src particle
      if (distance < (2.0 * d_consts.h)) {

        // Find vector describing velocity difference
        velocity_difference = particles[idx].velocity - particles[rel].velocity;

        // Find density and sum; find pressure from density at current value 
        particles[idx].density += particles[rel].mass * cubicSpline(distance, d_consts.h);
        particles[idx].pressure = static_cast<float>(k * (particles[idx].density - rho0));
        
        // Find pressure force and sum
        intermediate_value = displacement * (particles[rel].mass
          * (particles[idx].density + particles[rel].density) 
          / (2.0 * particles[idx].density * particles[rel].density));
        particles[idx].pressure_force -= (intermediate_value * gradCubicSpline(distance, d_consts.h));

        // Find viscosity (or 'liquid friction') force and sum
        intermediate_value = velocity_difference * (viscosity * particles[rel].mass * laplacianCubicSpline(distance, d_consts.h));
        particles[idx].viscosity_force += intermediate_value;
      }
    }
  }

  // Set particles array from args  
  args->particles = particles;
}

/**
 * @brief Completes verlet integration for one iteration
 */
__host__ void verletIntegration(integrateArgs_t *h_args, verletFunc_ processParticleIntegrate) {
  int threadPerBlock = 512;
  int gridSize = 2048;

  // Create memory on device for passing arguments 
  integrateArgs_t *d_args = integrateArgAllocate(h_args), result_ptr;

  // Kernel call to handle integration 
  processParticleIntegrate<<<gridSize, threadPerBlock>>>(d_args);

  // Copy modified d_args to temp holder
  cudaMemcpy(&result_ptr, d_args, sizeof(integrateArgs_t), cudaMemcpyDeviceToHost);

  // Copy from temp struct to original host array
  cudaMemcpy(h_args->particles, result_ptr.particles, h_args->size * sizeof(fparticle_t), cudaMemcpyDeviceToHost);

  // printf("Modified particle array copied back\n");

  // Free memory on device 
  cudaFree(result_ptr.particles);
  cudaFree(d_args);
}

/**
 * @brief First position and velocity update to half values 
 */
__global__ void process1stParticleIntegrate(integrateArgs_t *args) {
  // Set base args for id check 
  const uint32_t size = args->size;   

  // Thread id check
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= size) return;  // More threads than particles are launched => Return excess.

  // Copy particles locally
  fparticle_t *particles = args->particles;

  // Standard Velocity Verlet Integration Scheme
  vec3_float force_sum = {{0.0, -9.81, 0.0}};

  /* Integration */ 

  force_sum += (particles[idx].viscosity_force + particles[idx].pressure_force);

  // Update State
  vec3_float new_velocity, new_position;
  new_velocity = particles[idx].velocity + (force_sum * static_cast<float>(0.5 * dt));
  new_position = particles[idx].position + (new_velocity * static_cast<float>(dt));

  particles[idx].velocity = new_velocity;
  particles[idx].position = new_position;

  // Modify argument particles to pass back 
  args->particles = particles;
}

/**
 * @brief Second position and velocity update after new forces have been calculated 
 */
__global__ void process2ndParticleIntegrate(integrateArgs_t *args) {
  // Set base args for id check 
  const uint32_t size = args->size;   

  // Thread id check
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= size) return;  // More threads than particles are launched => Return excess.

  // Copy particles locally
  fparticle_t *particles = args->particles;

  // Standard Velocity Verlet Integration Scheme
  vec3_float force_sum = (vec3_float) {{0.0, -9.81, 0.0}};

  /* Integration */
  vec3_float new_velocity;

  // Sum forces
  particles[idx].pressure_force  /= particles[idx].mass;
  particles[idx].viscosity_force /= particles[idx].mass;

  force_sum += (particles[idx].pressure_force + particles[idx].viscosity_force);

  // Find next velocity
  new_velocity = particles[idx].velocity + (force_sum * static_cast<float>(0.5 * dt));
  particles[idx].velocity = new_velocity;

  // Modify argument particles to pass back 
  args->particles = particles;
}