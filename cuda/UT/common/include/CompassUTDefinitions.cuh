#pragma once

#include "UTDefinitions.cuh"
#include "SystemOfUnits.h"

namespace CompassUT {

  constexpr uint num_sectors = 5;
  constexpr uint num_elems = num_sectors * 2;

} // namespace CompassUT

//=========================================================================
// Point to correct position for windows pointers
//=========================================================================
struct TrackCandidates {
  const short* m_base_pointer;

  __device__ TrackCandidates(const short* base_pointer) : m_base_pointer(base_pointer) {}

  __device__ inline short get_from(int layer, int sector) const
  {
    return m_base_pointer
      [sector * UT::Constants::n_layers * UT::Constants::num_thr_compassut + layer * UT::Constants::num_thr_compassut +
       threadIdx.x];
  }

  __device__ inline short get_size(int layer, int sector) const
  {
    return m_base_pointer
      [(sector + (CompassUT::num_elems / 2)) * UT::Constants::n_layers * UT::Constants::num_thr_compassut +
       layer * UT::Constants::num_thr_compassut + threadIdx.x];
  }
};

//=========================================================================
// Save the best q/p, chi2 and number of hits
//=========================================================================
struct BestParams {
  float qp;
  float chi2UT;
  int n_hits;
  float x, z, tx;

  __device__ BestParams()
  {
    qp = 0.0f;
    chi2UT = UT::Constants::maxPseudoChi2;
    n_hits = 0;
    x = -10000;
    tx = -10000;
    z = -10000;
  }
};
