#pragma once

#include "VeloEventModel.cuh"
#include <cassert>

__device__ void process_modules(
  Velo::Module* module_data,
  bool* hit_used,
  const short* h0_candidates,
  const short* h2_candidates,
  const uint* module_hitStarts,
  const uint* module_hitNums,
  Velo::ConstClusters& velo_cluster_container,
  const float* hit_phi,
  uint* tracks_to_follow,
  Velo::TrackletHits* weak_tracks,
  Velo::TrackletHits* tracklets,
  Velo::TrackHits* tracks,
  const uint number_of_hits,
  unsigned short* h1_rel_indices,
  const uint hit_offset,
  const float* dev_velo_module_zs,
  uint* dev_atomics_velo,
  uint* dev_number_of_velo_tracks);
