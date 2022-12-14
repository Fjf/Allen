/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
/*!
 *  \brief     apply_permutation sorting tool.
 *  \author    Daniel Hugo Campora Perez
 *  \author    Dorothea vom Bruch
 *  \date      2018
 */

#pragma once

#include "BackendCommon.h"
#include <cassert>

/**
 * @brief Apply permutation from prev container to new container
 */
template<class T>
__host__ __device__ void apply_permutation(
  unsigned* permutation,
  const unsigned hit_start,
  const unsigned number_of_hits,
  T* prev_container,
  T* new_container)
{
  // Apply permutation across all hits
  FOR_STATEMENT(unsigned, i, number_of_hits)
  {
    const auto hit_index_global = permutation[hit_start + i];
    new_container[hit_start + i] = prev_container[hit_index_global];
  }
}
