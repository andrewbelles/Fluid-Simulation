#ifndef __ITERATOR_H__
#define __ITERATOR_H__

/* compiler neutral headers */
#include "spatiallookup.h"
#include "c_vec.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "headers.h"
#include "definitions.h"
#include "fluid_structs.h"

#include "particle.h"

void particleIterator(uint32_t particle_count);
void freeTables();

#ifdef __cplusplus
}
#endif

#endif // __ITERATOR_H__
