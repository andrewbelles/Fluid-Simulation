#include "iterator.h"

/**
 * @brief Constant values and fluid particle array are externally defined 
 */
extern Constants consts;
extern fparticle_t *particles;

/**
 * @brief Hash table size while be globally avaliable in this source file
 */
static uint32_table_t *lookup_table;
static uint32_t *start_cell;
static uint32_t *end_cell;
static uint32_t axis_count;

/**
 * @brief Find the next power of two greater than the input value
 */
static uint32_t findSquare(uint32_t value) {
  // Base case (If particle_count was zero for some reason)
  if (value == 0) {
    return 1;
  }

  // Maniputate bits to be 2^(n) - 1
  value--;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;

  // Re-add 1 and return result
  return value + 1;
}

/**
 * @brief Allocates memory for tables if they are uninitialized 
 */
static void allocateTables() {
  lookup_table = (uint32_table_t*)safe_malloc(consts.particle_count * sizeof(uint32_table_t));
  start_cell = (uint32_t*)safe_malloc(consts.partition_count * sizeof(uint32_t));
  end_cell = (uint32_t*)safe_malloc(consts.partition_count * sizeof(uint32_t));
}

/**
 * @brief Creates an unsorted spatial lookup table based on the positions of the particles positions
 */
static void makeSpatialLookup() {
  // Local struct
  uint32_table_t *array_toSort = NULL;

  // hash_index is limited to consts.particle_count 
  uint32_t hash_index = 0, prev_index, padded_size = 0; 
  vec3_int cell_coordinate;

  // Initialize all to sentinel value
  for (uint32_t idx = 0; idx < consts.partition_count; ++idx) {
    start_cell[idx] = UINT32_MAX;
    end_cell[idx]   = UINT32_MAX;
    
    // Reset lookup table values 
    if (idx < consts.particle_count) {
      lookup_table[idx].cell_key    = UINT32_MAX;
      lookup_table[idx].particle_id = UINT32_MAX;
    }
  }

  for (uint32_t particle = 0; particle < consts.particle_count; ++particle) {
    
    // Reset Accumulators  
    particles[particle].pressure_force  = (vec3_float) {.data = {0.0, 0.0, 0.0}};
    particles[particle].viscosity_force = (vec3_float) {.data = {0.0, 0.0, 0.0}};
    particles[particle].wall            = (vec3_int) { .data = {0, 0, 0}};
    particles[particle].density         = 1.0;
    particles[particle].pressure        = 0.0;
    
    // Hash the bucket value
    cell_coordinate = positionToID(
      particles[particle].position,
      consts.partition_length
    );

    // Hash cell coordinate 
    hash_index = hashFunction(
      cell_coordinate,
      consts.partition_count
    );

    // Fill lookup tble with particle id, cell key pair 
    lookup_table[particle] = (uint32_table_t) {
      .particle_id = particle,
      .cell_key    = hash_index
    };
  }

  // Pad array to next size of form 2^n to send to GPU 
  padded_size = findSquare(consts.particle_count);
  array_toSort = (uint32_table_t*)safe_malloc(padded_size * sizeof(uint32_table_t));

  // Add padded values sentinel values to 2^n 
  for (uint32_t idx = 0; idx < padded_size; ++idx) {
    if (idx < consts.particle_count) {
      array_toSort[idx].cell_key    = lookup_table[idx].cell_key;
      array_toSort[idx].particle_id = lookup_table[idx].particle_id; 
    } else {
      array_toSort[idx].cell_key    = UINT32_MAX;
      array_toSort[idx].particle_id = UINT32_MAX; 
    }
  }

  // Kernel function call to sort array 
  (void)bitonicSort(array_toSort, padded_size);

  // Copy back to normal array
  for (uint32_t idx = 0; idx < consts.particle_count; ++idx) {
    // Copy sorted spatial array over 
    lookup_table[idx].cell_key    = array_toSort[idx].cell_key;
    lookup_table[idx].particle_id = array_toSort[idx].particle_id; 
  }

  // Free padded array
  free(array_toSort);

  // Handle 0th particle separately
  hash_index = lookup_table[0].cell_key;
  start_cell[hash_index] = 0;
  prev_index = hash_index;

  // Create the start index table 
  for (uint32_t particle = 1; particle < consts.particle_count; ++particle) {
    // Find hash_index and prev_index for ith value
    hash_index = lookup_table[particle].cell_key;
    prev_index = lookup_table[particle - 1].cell_key;

    // If the hash_index doesn't match the previous index then place the next start index and the current end index
    if (hash_index != prev_index) {
      end_cell[prev_index] = particle;
      start_cell[hash_index] = particle; 
    }
  }
}

/**
 * @brief Simple function to return false if the coordinate does not lie on an edge 
 * @param cell_coordinate True cell position that particle lies in
 */
static bool onEdge(vec3_int cell_coordinate, vec3_int *bad_coord) {
  for (int i = 0; i < 3; ++i) {
    if (cell_coordinate.data[i] == 0 || cell_coordinate.data[i] == (consts.axis_count - 1)) {
      bad_coord->data[i] = 1;
      return true;
    }
  }
  return false;
}

/**
 * @brief Calculates the amount of overlap between a boundary condition and the particle
 * @param current_overlap summation of overlap if overlaping in mulitple directions
 * @param position particle's position 
 * @param radius particle's true radius
 * @param axis x,y,z axis that is being overlapped
 * 
 * @result new 3D position for the particle to assume
 */
static vec3_float overlapCalc(const vec3_float current_overlap, const vec3_float position, const float radius, const int axis) {
  vec3_float overlap = current_overlap;

  // Find the overlap in the upper and lower bound directions
  float upper = consts.cube_size - position.data[axis] - radius;
  float lower = position.data[axis] - radius;

  // Set to zero if it isn't actually intersecting with wall 
  upper = (upper < tol) ? abs(upper) + tol : 0.0;
  lower = (lower < tol) ? abs(lower) + tol : 0.0;

  // Set overlap for axis to sum and return
  overlap.data[axis] += -upper + lower;
  return overlap;
}

/**
 * @brief Processes a collision between the wall and a fluid particle
 * @param *particle Modifying specific particle by reference
 * @param bad_coord Out of bounds index
 */
static void processWall(vec3_int bad_coord, uint32_t particle_id) {
  const float restitution = 0.8;
  // Sets overlap 
  vec3_float overlap = (vec3_float) {{0.0, 0.0, 0.0}};

  // Iterates over x,y,z axis 
  for (int axis = 0; axis < 3; ++axis){
    // Skip if not the edge axis
    if (bad_coord.data[axis] == 0) continue;
    // Skip if already adjusted for wall this iteration
    if (particles[particle_id].wall.data[axis] == 1) continue;

    // Reset velocity and find the overlap in the axis direction
    overlap = overlapCalc(overlap, particles[particle_id].position, consts.h, axis);
    particles[particle_id].velocity.data[axis] *= -restitution;
    particles[particle_id].wall.data[axis] = 1;
  }
  // Update the position based on overlap
  particles[particle_id].position = subtractVec3_float(particles[particle_id].position, overlap);
}

/**
 * @brief Calls the iterator to check for boundary conditions and enforce them
 * @param None
 */
static void enforceBoundaryConditions() {
  // Local vars
  vec3_int cell_coord, bad_coord;
  uint32_t particle, hash, start, end;

  // Iterates over 3D cell coordinates
  for (int z = 0; z < consts.axis_count; ++z) {
    for (int y = 0; y < consts.axis_count; ++y) {
      for (int x = 0; x < consts.axis_count; ++x) {
        // Cell coord for iteration 
        cell_coord = (vec3_int) {{x, y, z}};
        bad_coord  = (vec3_int) {{0, 0, 0}};

        // Skip if cell isn't close enough to the edge to have an out of bounds particle
        if (!onEdge(cell_coord, &bad_coord)) {
          continue;
        }

        // Find hash for edge coordinate 
        hash = hashFunction(
          cell_coord,
          consts.partition_count
        );

        // Find start and end values for edge bucket 
        start = start_cell[hash];
        end   = end_cell[hash];

        // Iterate over number of particles in bucket and enforce boundary condition
        for (uint32_t i = start; i < end; ++i) {
          // Find the particle id from the lookup table
          particle = lookup_table[i].particle_id;
          if (particle > consts.particle_count) {
            fprintf(stderr, "Particle ID exceeds valid value: %u\n", particle);
            exit(EXIT_FAILURE);
          }

          // Process potential out of bounds behavior 
          processWall(bad_coord, particle);
        }
      }
    }
  }
}

/**
 * @brief Main iteratoring function that calculates lookup table each iteration
 * 
 * Not worried about threading until made as compute shader (Is this something I'd make a shader for?)
 */
void particleIterator(uint32_t particle_count) {
  /**
   * Partition cube into consts.h sections rounding down to ensure cubes are >= consts.h
   */
  float partition_length = consts.h;
  float max = consts.cube_size - (consts.cube_size / 8.0), min = (consts.cube_size / 8.0);;
  
  // Set Arguments for neighborSearch and integrator
  searchArgs_t search_args;
  integrateArgs_t integrate_args;

  // Allocate memories for table if they do not exist
  if (lookup_table == NULL || start_cell == NULL || end_cell == NULL) allocateTables();

  // Set most args now
  search_args.start_cell = start_cell;
  search_args.end_cell   = end_cell;
  search_args.size       = consts.particle_count;
  integrate_args.size    = consts.particle_count;

  // Find potentially new axis count for iteration
  axis_count = (int)floor(consts.cube_size / partition_length);

  // Update values if necessary 
  if (axis_count != consts.axis_count) {
    consts.partition_count = axis_count * axis_count * axis_count;
    consts.axis_count = axis_count;
    // Copy to device if value has shifted 
    (void)copyConstantsToDevice(consts);

    // Resize start and end cell arrays
    start_cell   = (uint32_t *)realloc(start_cell, consts.partition_count * sizeof(uint32_t));
    end_cell     = (uint32_t *)realloc(end_cell, consts.partition_count * sizeof(uint32_t));
  }

  // If particle count increases/decreases 
  if (particle_count != consts.particle_count) {
    // Realloc all arrays
    particles    = (fparticle_t *)realloc(particles, particle_count * sizeof(fparticle_t));
    lookup_table = (uint32_table_t *)realloc(lookup_table, particle_count * sizeof(uint32_table_t));

    // Create new particles to fill new uninitialized space if array increases in size
    if (particle_count > consts.particle_count) {
      for (uint32_t idx = consts.particle_count; idx < particle_count - consts.particle_count; ++idx) {
        particles[idx] = setParticle(min, max);
      }
    }

    // Set new particle count as global
    consts.particle_count = particle_count;
  }

  // Generate spacial lookup for iteration
  makeSpatialLookup();
  search_args.table = lookup_table;

  // Enfore boundary conditions in the space defined above
  enforceBoundaryConditions();

  // Set particle array for integrate args 
  integrate_args.particles = particles;

  // Start integrating through the current acceleration
  verletIntegration(&integrate_args, process1stParticleIntegrate);
  
  // Set particle array in search args 
  search_args.particles  = particles;

  // Kernel call to neighborSearch
  neighborSearch(&search_args);

  // Set particle array in integrate args 

  // Finish the process with the updated acceleration
  verletIntegration(&integrate_args, process2ndParticleIntegrate);
}

/**
 * @brief Frees memory allocated to tables
 */
void freeTables() {
  free(lookup_table);
  free(start_cell);
  free(end_cell);
}