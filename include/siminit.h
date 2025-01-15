#ifndef __SIMINIT_H__
#define __SIMINIT_H__

#include "spatiallookup.h"
#include "c_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "headers.h"
#include "definitions.h"
#include "fluid_structs.h"

#include "particle.h"

void startSim(float cube_size, const uint32_t particle_count, float h);

#ifdef __cplusplus
}
#endif

#endif // __SIMINIT_H__