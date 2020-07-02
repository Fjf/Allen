#pragma once

#include "VeloEventModel.cuh"
#include "BinarySearch.cuh"
#include "VeloTools.cuh"
#include <tuple>

__device__ void track_seeding(
  Velo::ConstClusters& velo_cluster_container,
  const Velo::ModulePair* module_pair_data,
  bool* hit_used,
  Velo::TrackletHits* tracklets,
  unsigned* tracks_to_follow,
  unsigned short* h1_rel_indices,
  unsigned* dev_shifted_atomics_velo,
  const float max_scatter,
  const int16_t* hit_phi,
  const int16_t phi_tolerance);

__device__ void track_forwarding(
  Velo::ConstClusters& velo_cluster_container,
  const int16_t* hit_phi,
  bool* hit_used,
  const Velo::ModulePair* module_pair_data,
  const unsigned diff_ttf,
  unsigned* tracks_to_follow,
  Velo::TrackletHits* weak_tracks,
  const unsigned prev_ttf,
  Velo::TrackletHits* tracklets,
  Velo::TrackHits* tracks,
  unsigned* dev_atomics_velo,
  unsigned* dev_number_of_velo_tracks,
  const int16_t phi_tolerance,
  const float max_scatter,
  const unsigned max_skipped_modules);

/**
 * @brief Returns the first possible candidate, according to
 *        extrapolation of the track to phi minus the tolerance.
 *        Returns the candidate, and the extrapolated phi value.
 */
__device__ inline std::tuple<int, int16_t> find_forward_candidate(
  const Velo::ModulePair& module_pair,
  const int16_t* hit_Phis,
  const Velo::HitBase& h0,
  const float tx,
  const float ty,
  const float dz,
  const int16_t phi_tolerance)
{
  const auto predx = tx * dz;
  const auto predy = ty * dz;
  const auto x_prediction = h0.x + predx;
  const auto y_prediction = h0.y + predy;
  const auto track_extrapolation_phi = hit_phi_16(x_prediction, y_prediction);

  return {binary_search_leftmost(
            hit_Phis + module_pair.hit_start,
            module_pair.hit_num,
            static_cast<int16_t>(track_extrapolation_phi - phi_tolerance)),
          track_extrapolation_phi};
}
