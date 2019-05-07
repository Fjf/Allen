#include "ProcessModules.cuh"
#include "TrackSeeding.cuh"
#include "TrackForwarding.cuh"
#include "ClusteringDefinitions.cuh"

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
  const float* dev_velo_cluster_container,
  uint* tracks_to_follow,
  Velo::TrackletHits* weak_tracks,
  Velo::TrackletHits* tracklets,
  Velo::TrackHits* tracks,
  const uint number_of_hits,
  unsigned short* h1_rel_indices,
  const uint hit_offset,
  const float* dev_velo_module_zs,
  int* dev_atomics_velo)
{
  const int ip_shift = gridDim.x + blockIdx.x * (Velo::num_atomics - 1);
  auto first_module = VP::NModules - 1;

  // Prepare the first seeding iteration
  // Load shared module information
  if (threadIdx.x < 6) {
    const auto module_number = first_module - threadIdx.x;
    module_data[threadIdx.x].hitStart = module_hitStarts[module_number] - hit_offset;
    module_data[threadIdx.x].hitNums = module_hitNums[module_number];
    module_data[threadIdx.x].z = dev_velo_module_zs[module_number];
  }

  // Due to shared module data loading
  __syncthreads();

  // Do first track seeding
  track_seeding(
    dev_velo_cluster_container,
    number_of_hits,
    module_data,
    h0_candidates,
    h2_candidates,
    hit_used,
    tracklets,
    tracks_to_follow,
    h1_rel_indices,
    dev_atomics_velo,
    ip_shift);

  // Prepare forwarding - seeding loop
  uint last_ttf = 0;
  first_module -= 2;

  while (first_module >= 4) {

    // Due to WAR between trackSeedingFirst and the code below
    __syncthreads();

    // Iterate in modules
    // Load in shared
    if (threadIdx.x < 6) {
      const auto module_number = first_module - threadIdx.x;
      module_data[threadIdx.x].hitStart = module_hitStarts[module_number] - hit_offset;
      module_data[threadIdx.x].hitNums = module_hitNums[module_number];
      module_data[threadIdx.x].z = dev_velo_module_zs[module_number];
    }

    const auto prev_ttf = last_ttf;
    last_ttf = dev_atomics_velo[ip_shift + 2];
    const auto diff_ttf = last_ttf - prev_ttf;

    // Reset atomics
    // Note: local_number_of_hits
    dev_atomics_velo[ip_shift + 3] = 0;

    // Due to module data loading
    __syncthreads();

    // Track Forwarding
    track_forwarding(
      dev_velo_cluster_container,
      hit_used,
      module_data,
      diff_ttf,
      tracks_to_follow,
      weak_tracks,
      prev_ttf,
      tracklets,
      tracks,
      number_of_hits,
      dev_atomics_velo,
      ip_shift);

    // Due to ttf_insert_pointer
    __syncthreads();

    // Seeding
    track_seeding(
      dev_velo_cluster_container,
      number_of_hits,
      module_data,
      h0_candidates,
      h2_candidates,
      hit_used,
      tracklets,
      tracks_to_follow,
      h1_rel_indices,
      dev_atomics_velo,
      ip_shift);

    first_module -= 2;
  }

  // Due to last seeding ttf_insert_pointer
  __syncthreads();

  const auto prev_ttf = last_ttf;
  last_ttf = dev_atomics_velo[ip_shift + 2];
  const auto diff_ttf = last_ttf - prev_ttf;

  // Process the last bunch of track_to_follows
  for (int i = 0; i < (diff_ttf + blockDim.x - 1) / blockDim.x; ++i) {
    const auto ttf_element = blockDim.x * i + threadIdx.x;

    if (ttf_element < diff_ttf) {
      const int fulltrackno = tracks_to_follow[(prev_ttf + ttf_element) % Velo::Tracking::ttf_modulo];
      const bool track_flag = (fulltrackno & 0x80000000) == 0x80000000;
      const int trackno = fulltrackno & 0x0FFFFFFF;

      // Here we are only interested in three-hit tracks,
      // to mark them as "doubtful"
      if (track_flag) {
        const auto weakP = atomicAdd(dev_atomics_velo + ip_shift, 1);
        assert(weakP < number_of_hits);
        weak_tracks[weakP] = tracklets[trackno];
      }
    }
  }
}
