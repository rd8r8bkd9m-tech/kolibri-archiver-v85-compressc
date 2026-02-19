/*
 * Copyright (c) 2025 Кочуров Владислав Евгеньевич
 */

#include "kolibri/random.h"

#define K_RNG_MULTIPLIER 6364136223846793005ULL
#define K_RNG_INCREMENT 1442695040888963407ULL

void k_rng_seed(KolibriRng *rng, uint64_t seed) {
  if (!rng) {
    return;
  }
  rng->state = seed + K_RNG_INCREMENT;
  (void)k_rng_next(rng);
}

uint64_t k_rng_next(KolibriRng *rng) {
  if (!rng) {
    return 0;
  }
  rng->state = rng->state * K_RNG_MULTIPLIER + K_RNG_INCREMENT;
  uint64_t xorshifted = ((rng->state >> 18U) ^ rng->state) >> 27U;
  uint64_t rot = rng->state >> 59U;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

double k_rng_next_double(KolibriRng *rng) {
  const uint64_t value = k_rng_next(rng);
  return (value >> 11) * (1.0 / 9007199254740992.0);
}
