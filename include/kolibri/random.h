#ifndef KOLIBRI_RANDOM_H
#define KOLIBRI_RANDOM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t state;
} KolibriRng;

void k_rng_seed(KolibriRng *rng, uint64_t seed);
uint64_t k_rng_next(KolibriRng *rng);
double k_rng_next_double(KolibriRng *rng);

#ifdef __cplusplus
}
#endif

#endif /* KOLIBRI_RANDOM_H */
