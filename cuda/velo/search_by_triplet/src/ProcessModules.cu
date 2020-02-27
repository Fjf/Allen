#include "ProcessModules.cuh"
#include "TrackSeeding.cuh"
#include "TrackForwarding.cuh"
#include "ClusteringDefinitions.cuh"
#include "SearchByTriplet.cuh"

/**
 * @brief Processes modules in decreasing order with some stride
 */
__device__ void process_modules(
  Velo::Module* module_data,
  bool* hit_used,
  const short* h0_candidates,
  const short* h2_candidates,
  const uint* module_hitStarts,
  const uint* module_hitNums,
  Velo::ConstClusters& velo_cluster_container,
  const half_t* hit_phi,
  uint* tracks_to_follow,
  Velo::TrackletHits* weak_tracks,
  Velo::TrackletHits* tracklets,
  Velo::TrackHits* tracks,
  unsigned short* h1_rel_indices,
  const uint hit_offset,
  const float* dev_velo_module_zs,
  uint* dev_atomics_velo,
  uint* dev_number_of_velo_tracks,
  const int ttf_modulo_mask,
  const float max_scatter_seeding,
  const uint ttf_modulo,
  const float max_scatter_forwarding,
  const uint max_skipped_modules,
  const float forward_phi_tolerance)
{
  auto first_module = VP::NModules - 1;

  // Prepare the first seeding iteration
  // Load shared module information
  for (uint i = threadIdx.x; i < 4; i += blockDim.x) {
    const auto module_number = first_module - i - 2;
    module_data[i].hitStart = module_hitStarts[module_number] - hit_offset;
    module_data[i].hitNums = module_hitNums[module_number];
    module_data[i].z = dev_velo_module_zs[module_number];
  }

  // Due to shared module data loading
  __syncthreads();

  // Do first track seeding
  track_seeding(
    velo_cluster_container,
    module_data,
    h0_candidates,
    h2_candidates,
    hit_used,
    tracklets,
    tracks_to_follow,
    h1_rel_indices,
    dev_atomics_velo,
    max_scatter_seeding,
    ttf_modulo_mask);

  // Prepare forwarding - seeding loop
  uint last_ttf = 0;
  first_module -= 2;

  while (first_module > 4) {

    // Due to WAR between trackSeedingFirst and the code below
    __syncthreads();

    // Iterate in modules
    // Load in shared
    for (int i = threadIdx.x; i < 4; i += blockDim.x) {
      const auto module_number = first_module - i - 2;
      module_data[i].hitStart = module_hitStarts[module_number] - hit_offset;
      module_data[i].hitNums = module_hitNums[module_number];
      module_data[i].z = dev_velo_module_zs[module_number];
    }

    const auto prev_ttf = last_ttf;
    last_ttf = dev_atomics_velo[2];
    const auto diff_ttf = last_ttf - prev_ttf;

    // Reset atomics
    // Note: local_number_of_hits
    dev_atomics_velo[3] = 0;

    // Due to module data loading
    __syncthreads();

    // Track Forwarding
    track_forwarding(
      velo_cluster_container,
      hit_phi,
      hit_used,
      module_data,
      diff_ttf,
      tracks_to_follow,
      weak_tracks,
      prev_ttf,
      tracklets,
      tracks,
      dev_atomics_velo,
      dev_number_of_velo_tracks,
      forward_phi_tolerance,
      ttf_modulo_mask,
      ttf_modulo,
      max_scatter_forwarding,
      max_skipped_modules);

    // Due to ttf_insert_pointer
    __syncthreads();

    // Seeding
    track_seeding(
      velo_cluster_container,
      module_data,
      h0_candidates,
      h2_candidates,
      hit_used,
      tracklets,
      tracks_to_follow,
      h1_rel_indices,
      dev_atomics_velo,
      max_scatter_seeding,
      ttf_modulo_mask);

    first_module -= 2;
  }

  // Due to last seeding ttf_insert_pointer
  __syncthreads();

  const auto prev_ttf = last_ttf;
  last_ttf = dev_atomics_velo[2];
  const auto diff_ttf = last_ttf - prev_ttf;

  // Process the last bunch of track_to_follows
  for (uint ttf_element = threadIdx.x; ttf_element < diff_ttf; ttf_element += blockDim.x) {
    const int fulltrackno =
      tracks_to_follow[(prev_ttf + ttf_element) & ttf_modulo_mask];
    const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
    const int trackno = fulltrackno & 0x0FFFFFFF;

    // Here we are only interested in three-hit tracks,
    // to mark them as "doubtful"
    if (track_flag) {
      const auto weakP = atomicAdd(dev_atomics_velo, 1);
      weak_tracks[weakP] = tracklets[trackno];
    }
  }
}
