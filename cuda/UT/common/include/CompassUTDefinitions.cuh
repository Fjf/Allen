#pragma once

#include "UTDefinitions.cuh"

namespace CompassUT {

constexpr uint max_considered_before_found = 4;
constexpr uint num_threads = 32;

}

//=========================================================================
// Point to correct position for windows pointers
//=========================================================================
struct LayerCandidates {
  int from0;
  int size0;
  int from1;
  int size1;
  int from2;
  int size2;
};

struct TrackCandidates {
  LayerCandidates layer[UT::Constants::n_layers];
};

struct WindowIndicator {
  const int* m_base_pointer;
  __host__ __device__ WindowIndicator(const int* base_pointer) : m_base_pointer(base_pointer) {}

  __host__ __device__ const TrackCandidates* get_track_candidates(const int i_track)
  {
    return reinterpret_cast<const TrackCandidates*>(m_base_pointer + (6 * UT::Constants::n_layers * i_track));
  }
};

//=========================================================================
// Save the best q/p, chi2 and number of hits
//=========================================================================
struct BestParams {
  float qp;
  float chi2UT;
  int n_hits;

  __host__ __device__ BestParams () 
  {
    qp = 0.0f;
    chi2UT = UT::Constants::maxPseudoChi2;
    n_hits = 0;
  }
};