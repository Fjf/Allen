/*!
 *  \brief     find_permutation sorting tool.
 *  \author    Daniel Hugo Campora Perez
 *  \author    Dorothea vom Bruch
 *  \date      2018
 */

#pragma once

#include "CudaCommon.h"
#include <cassert>

/**
 * @brief Sort by var stored in sorting_vars, store index in hit_permutations
 */
template<class T>
__host__ __device__ void find_permutation(
  const unsigned hit_start,
  const unsigned hit_permutations_start,
  const unsigned number_of_hits,
  unsigned* hit_permutations,
  const T& sort_function)
{
  FOR_STATEMENT(unsigned, i, number_of_hits)
  {
    const unsigned hit_index = hit_start + i;

    // Find out local position
    unsigned position = 0;
    for (unsigned j = 0; j < number_of_hits; ++j) {
      const unsigned other_hit_index = hit_start + j;
      const int sort_result = sort_function(hit_index, other_hit_index);
      // Stable sort
      position += sort_result > 0 || (sort_result == 0 && i > j);
    }
    assert(position < number_of_hits);

    // Store it in hit_permutations
    hit_permutations[hit_permutations_start + position] = hit_permutations_start + i;
  }
}
