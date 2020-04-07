#include "TrackForwarding.cuh"
#include "ClusteringDefinitions.cuh"
#include "SearchByTriplet.cuh"
#include "VeloTools.cuh"
#include <cstdio>

using namespace Velo::Tracking;

/**
 * @brief Track forwarding algorithm based on triplet finding.
 *
 * @detail Search by triplet is a parallel local track forwarding algorithm, whose main building blocks are two steps:
 *         track seeding and track forwarding. These two steps are applied iteratively
 *         throughout the VELO detector. The forward end of the detector is used as the start of the search,
 *         and the detector is traversed in the backwards direction, in groups of two modules at a time:
 *
 *         i-3    i-2   [i-1   i   i+1]
 *                      =============== Track seeding of triplet of modules {i-1, i, i+1}
 *
 *         i-3   [i-2] [i-1   i   i+1]
 *               =====                  Track forwarding to module i-2
 *
 *         i-3   [i-2   i-1   i]  i+1
 *               ===============        Track seeding of triplet of modules {i-2, i-1, i}
 *
 *         [i-3] [i-2   i-1   i   i+1]
 *         =====                        Track forwarding to module i-3
 *
 *         [i-3   i-2   i-1]  i   i+1
 *         =================            Track seeding of triplet of modules {i-3, i-2, i-1}
 *
 *         * Track seeding: Triplets of hits in consecutive modules on the same side are sought.
 *         The three hits composing a track seed must be on the same side - empirically it was found that
 *         no physics efficiency is gained if allowing for triplets to be on both sides.
 *         Incoming VELO cluster data is expected to be sorted by phi previously. This fact allows for several
 *         optimizations in the triplet seed search. First, the closest hit in the previous module is sought with
 *         a binary search. The closest n candidates in memory are found with a pendulum-like search (more details
 *         in track_seeding). The doublet is extrapolated to the third module, and a triplet is formed. Hits used
 *         to form triplets must be "not used".
 *
 *         * Track forwarding: Triplet track seeds and tracks with more than three hits are
 *         extended to modules by extrapolating the last two hits into the next layer and finding the
 *         best hits. Again, a binary search in phi is used to speed up the search. If hits are found,
 *         the track is extended and all hits found are marked as "used".
 *
 *         Both for track seeding and for track forwarding, a "max_scatter" function is used to determine the best hit.
 *         This function simply minimizes dx^2 + dy^2 in the detector plane.
 *
 *         The "hit used" array imposes a Read-After-Write dependency from every seeding stage to every forwarding
 *         stage, and a Write-After-Read dependency from every forwarding stage to every seeding stage. Hence, execution
 *         of these two stages is separated with control flow barriers.
 *
 *         For more details see:
 *         * https://ieeexplore.ieee.org/document/8778210
 */
__global__ void velo_search_by_triplet::velo_search_by_triplet(
  velo_search_by_triplet::Parameters parameters,
  const VeloGeometry* dev_velo_geometry)
{
  // Initialize event number and number of events based on kernel invoking parameters
  const uint event_number = blockIdx.x;
  const uint number_of_events = gridDim.x;

  // Pointers to data within the event
  const uint tracks_offset = event_number * Velo::Constants::max_tracks;
  const uint total_estimated_number_of_clusters =
    parameters.dev_offsets_estimated_input_size[Velo::Constants::n_modules * number_of_events];
  const uint* module_hitStarts =
    parameters.dev_offsets_estimated_input_size + event_number * Velo::Constants::n_modules;
  const uint* module_hitNums = parameters.dev_module_cluster_num + event_number * Velo::Constants::n_modules;
  const uint hit_offset = module_hitStarts[0];

  const auto velo_cluster_container =
    Velo::ConstClusters {parameters.dev_sorted_velo_cluster_container, total_estimated_number_of_clusters, hit_offset};

  const auto hit_phi = parameters.dev_hit_phi + hit_offset;

  Velo::TrackHits* tracks = parameters.dev_tracks + tracks_offset;
  bool* hit_used = parameters.dev_hit_used + hit_offset;

  uint* tracks_to_follow = parameters.dev_tracks_to_follow + event_number * parameters.max_tracks_to_follow;
  Velo::TrackletHits* three_hit_tracks = parameters.dev_three_hit_tracks + event_number * parameters.max_weak_tracks;
  Velo::TrackletHits* tracklets = parameters.dev_tracklets + event_number * parameters.max_tracks_to_follow;
  unsigned short* h1_rel_indices = parameters.dev_rel_indices + event_number * Velo::Constants::max_numhits_in_module;

  // Shared memory size is constantly fixed, enough to fit information about six modules
  // (three on each side).
  __shared__ float module_data[18];

  process_modules(
    (Velo::Module*) &module_data[0],
    hit_used,
    module_hitStarts,
    module_hitNums,
    velo_cluster_container,
    hit_phi,
    tracks_to_follow,
    three_hit_tracks,
    tracklets,
    tracks,
    h1_rel_indices,
    hit_offset,
    dev_velo_geometry->module_zs,
    parameters.dev_atomics_velo + blockIdx.x * Velo::num_atomics,
    parameters.dev_number_of_velo_tracks,
    parameters.max_scatter_seeding,
    parameters.max_tracks_to_follow,
    parameters.max_scatter_forwarding,
    parameters.max_skipped_modules,
    parameters.forward_phi_tolerance);
}

/**
 * @brief Processes modules in decreasing order.
 */
__device__ void process_modules(
  Velo::Module* module_data,
  bool* hit_used,
  const uint* module_hitStarts,
  const uint* module_hitNums,
  Velo::ConstClusters& velo_cluster_container,
  const int16_t* hit_phi,
  uint* tracks_to_follow,
  Velo::TrackletHits* three_hit_tracks,
  Velo::TrackletHits* tracklets,
  Velo::TrackHits* tracks,
  unsigned short* h1_rel_indices,
  const uint hit_offset,
  const float* dev_velo_module_zs,
  uint* dev_atomics_velo,
  uint* dev_number_of_velo_tracks,
  const float max_scatter_seeding,
  const uint max_tracks_to_follow,
  const float max_scatter_forwarding,
  const uint max_skipped_modules,
  const float forward_phi_tolerance)
{
  auto first_module = VP::NModules - 1;

  // Prepare the first seeding iteration
  // Load shared module information
  for (uint i = threadIdx.x; i < 6; i += blockDim.x) {
    const auto module_number = first_module - i;
    module_data[i].hitStart = module_hitStarts[module_number] - hit_offset;
    module_data[i].hitNums = module_hitNums[module_number];
    module_data[i].z = dev_velo_module_zs[module_number];
  }

  // Due to shared module data initialization
  __syncthreads();

  // Do first track seeding
  track_seeding(
    velo_cluster_container,
    module_data,
    hit_used,
    tracklets,
    tracks_to_follow,
    h1_rel_indices,
    dev_atomics_velo,
    max_scatter_seeding,
    max_tracks_to_follow,
    hit_phi);

  // Prepare forwarding - seeding loop
  // For an explanation on ttf, see below
  uint last_ttf = 0;
  first_module -= 2;

  while (first_module > 4) {

    // Due to WAR between track_seeding and population of shared memory.
    __syncthreads();

    // Iterate in modules
    // Load in shared
    for (int i = threadIdx.x; i < 6; i += blockDim.x) {
      const auto module_number = first_module - i;
      module_data[i].hitStart = module_hitStarts[module_number] - hit_offset;
      module_data[i].hitNums = module_hitNums[module_number];
      module_data[i].z = dev_velo_module_zs[module_number];
    }

    // ttf stands for "tracks to forward"
    // The tracks to forward are stored in a circular buffer.
    const auto prev_ttf = last_ttf;
    last_ttf = dev_atomics_velo[atomics::tracks_to_follow];
    const auto diff_ttf = last_ttf - prev_ttf;

    // Reset local number of hits
    dev_atomics_velo[atomics::local_number_of_hits] = 0;

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
      three_hit_tracks,
      prev_ttf,
      tracklets,
      tracks,
      dev_atomics_velo,
      dev_number_of_velo_tracks,
      forward_phi_tolerance,
      max_tracks_to_follow,
      max_scatter_forwarding,
      max_skipped_modules);

    // Due to ttf_insert_pointer
    __syncthreads();

    // Seeding
    track_seeding(
      velo_cluster_container,
      module_data,
      hit_used,
      tracklets,
      tracks_to_follow,
      h1_rel_indices,
      dev_atomics_velo,
      max_scatter_seeding,
      max_tracks_to_follow,
      hit_phi);

    first_module -= 2;
  }

  // Due to last seeding ttf_insert_pointer
  __syncthreads();

  const auto prev_ttf = last_ttf;
  last_ttf = dev_atomics_velo[atomics::tracks_to_follow];
  const auto diff_ttf = last_ttf - prev_ttf;

  // Process the last bunch of track_to_follows
  for (uint ttf_element = threadIdx.x; ttf_element < diff_ttf; ttf_element += blockDim.x) {
    const auto full_track_number = tracks_to_follow[(prev_ttf + ttf_element) % max_tracks_to_follow];
    const bool track_flag = (full_track_number & bits::seed) == bits::seed;

    // Here we are only interested in three-hit tracks,
    // to mark them as "doubtful"
    if (track_flag) {
      const auto track_number = full_track_number & bits::track_number;
      const Velo::TrackHits* t = (Velo::TrackHits*) &(tracklets[track_number]);
      const auto three_hit_tracks_p = atomicAdd(dev_atomics_velo + atomics::number_of_three_hit_tracks, 1);
      three_hit_tracks[three_hit_tracks_p] = Velo::TrackletHits {t->hits[0], t->hits[1], t->hits[2]};
    }
  }
}

/**
 * @brief Performs the track forwarding of forming tracks
 */
__device__ void track_forwarding(
  Velo::ConstClusters& velo_cluster_container,
  const int16_t* hit_phi,
  bool* hit_used,
  const Velo::Module* module_data,
  const uint diff_ttf,
  uint* tracks_to_follow,
  Velo::TrackletHits* three_hit_tracks,
  const uint prev_ttf,
  Velo::TrackletHits* tracklets,
  Velo::TrackHits* tracks,
  uint* dev_atomics_velo,
  uint* dev_number_of_velo_tracks,
  const float forward_phi_tolerance,
  const uint max_tracks_to_follow,
  const float max_scatter_forwarding,
  const uint max_skipped_modules)
{
  // Assign a track to follow to each thread
  for (uint ttf_element = threadIdx.x; ttf_element < diff_ttf; ttf_element += blockDim.x) {
    const auto full_track_number = tracks_to_follow[(prev_ttf + ttf_element) % max_tracks_to_follow];
    const bool track_flag = (full_track_number & bits::seed) == bits::seed;
    const auto skipped_modules = (full_track_number & bits::skipped_modules) >> bits::skipped_module_position;
    auto track_number = full_track_number & bits::track_number;

    assert(track_flag ? track_number < max_tracks_to_follow : track_number < Velo::Constants::max_tracks);

    uint number_of_hits;
    Velo::TrackHits* t;

    if (track_flag) {
      t = (Velo::TrackHits*) &(tracklets[track_number]);
      number_of_hits = 3;
    }
    else {
      t = tracks + track_number;
      number_of_hits = t->hitsNum;
    }

    // Load last two hits in h0, h1
    const auto h0_num = t->hits[number_of_hits - 2];
    const auto h1_num = t->hits[number_of_hits - 1];

    const Velo::HitBase h0 {
      velo_cluster_container.x(h0_num), velo_cluster_container.y(h0_num), velo_cluster_container.z(h0_num)};

    const Velo::HitBase h1 {
      velo_cluster_container.x(h1_num), velo_cluster_container.y(h1_num), velo_cluster_container.z(h1_num)};

    // Track forwarding over t, for all hits in the next module
    // Line calculations
    const auto td = 1.0f / (h1.z - h0.z);
    const auto txn = (h1.x - h0.x);
    const auto tyn = (h1.y - h0.y);
    const auto tx = txn * td;
    const auto ty = tyn * td;

    // Find the best candidate
    float best_fit = max_scatter_forwarding;
    int best_h2 = -1;
    const int16_t forward_phi_tolerance_int = static_cast<int16_t>(forward_phi_tolerance * Velo::Tools::convert_factor);

    // Get candidates by performing a binary search in expected phi
    const auto odd_module_candidates =
      find_forward_candidates(module_data[shared::next_module_pair], tx, ty, hit_phi, h0, 1, forward_phi_tolerance_int);

    const auto even_module_candidates = find_forward_candidates(
      module_data[shared::next_module_pair + 1], tx, ty, hit_phi, h0, 0, forward_phi_tolerance_int);

    // Search on both modules in the same for loop
    const int total_odd_candidates = std::get<1>(odd_module_candidates);
    const int total_even_candidates = std::get<1>(even_module_candidates);
    const int total_candidates = total_odd_candidates + total_even_candidates;

    for (int j = 0; j < total_candidates; ++j) {
      const int h2_index = j < total_odd_candidates ? std::get<0>(odd_module_candidates) + j :
                                                      std::get<0>(even_module_candidates) + j - total_odd_candidates;

      const Velo::HitBase h2 {
        velo_cluster_container.x(h2_index), velo_cluster_container.y(h2_index), velo_cluster_container.z(h2_index)};

      const auto dz = h2.z - h0.z;
      const auto predx = h0.x + tx * dz;
      const auto predy = h0.y + ty * dz;
      const auto dx = predx - h2.x;
      const auto dy = predy - h2.y;

      // Scatter
      const auto scatter = (dx * dx) + (dy * dy);

      // We keep the best one found
      if (scatter < best_fit) {
        best_fit = scatter;
        best_h2 = h2_index;
      }
    }

    // Condition for finding a h2
    if (best_h2 != -1) {
      // Mark h2 as used
      hit_used[best_h2] = true;

      // Update the track in the bag
      if (number_of_hits == 3) {
        // Also mark the first three as used
        hit_used[t->hits[0]] = true;
        hit_used[t->hits[1]] = true;
        hit_used[t->hits[2]] = true;

        // If it is a track made out of less than or equal to 4 hits,
        // we have to allocate it in the tracks pointer
        track_number = atomicAdd(dev_number_of_velo_tracks + blockIdx.x, 1);
        tracks[track_number].hits[0] = t->hits[0];
        tracks[track_number].hits[1] = t->hits[1];
        tracks[track_number].hits[2] = t->hits[2];
        tracks[track_number].hits[3] = best_h2;
        tracks[track_number].hitsNum = 4;
      }
      else {
        t->hits[t->hitsNum++] = best_h2;
      }

      if (number_of_hits + 1 < Velo::Constants::max_track_size) {
        // Add the tracks to the bag of tracks to_follow
        const auto ttf_p = atomicAdd(dev_atomics_velo + atomics::tracks_to_follow, 1) % max_tracks_to_follow;
        tracks_to_follow[ttf_p] = track_number;
      }
    }
    // A track just skipped a module
    // We keep it for another round
    else if (skipped_modules < max_skipped_modules) {
      // Form the new mask
      track_number = ((skipped_modules + 1) << bits::skipped_module_position) |
                     (full_track_number & (bits::seed | bits::track_number));

      // Add the tracks to the bag of tracks to_follow
      const auto ttf_p = atomicAdd(dev_atomics_velo + atomics::tracks_to_follow, 1) % max_tracks_to_follow;
      tracks_to_follow[ttf_p] = track_number;
    }
    // If there are only three hits in this track,
    // mark it as "doubtful"
    else if (number_of_hits == 3) {
      const auto three_hit_tracks_p = atomicAdd(dev_atomics_velo + atomics::number_of_three_hit_tracks, 1);
      three_hit_tracks[three_hit_tracks_p] = Velo::TrackletHits {t->hits[0], t->hits[1], t->hits[2]};
    }
    // In the "else" case, we couldn't follow up the track,
    // so we won't be track following it anymore.
  }
}

/**
 * @brief Search for compatible triplets in
 *        three neighbouring modules on one side
 */
__device__ void track_seeding(
  Velo::ConstClusters& velo_cluster_container,
  const Velo::Module* module_data,
  bool* hit_used,
  Velo::TrackletHits* tracklets,
  uint* tracks_to_follow,
  unsigned short* h1_indices,
  uint* dev_atomics_velo,
  const float max_scatter_seeding,
  const uint max_tracks_to_follow,
  const int16_t* hit_phi)
{
  // Add to an array all non-used h1 hits
  for (auto module_index : {shared::current_module_pair, shared::current_module_pair + 1}) {
    for (uint h1_rel_index = threadIdx.x; h1_rel_index < module_data[module_index].hitNums;
         h1_rel_index += blockDim.x) {
      const auto h1_index = module_data[module_index].hitStart + h1_rel_index;
      if (!hit_used[h1_index]) {
        const auto current_hit = atomicAdd(dev_atomics_velo + atomics::local_number_of_hits, 1);
        const auto oddity = module_index % 2;
        h1_indices[current_hit] = (oddity << bits::oddity_position) | h1_index;
      }
    }
  }

  // Due to h1_indices
  __syncthreads();

  // Assign a h1 to each threadIdx.x
  const auto number_of_hits_h1 = dev_atomics_velo[atomics::local_number_of_hits];
  for (uint h1_rel_index = threadIdx.x; h1_rel_index < number_of_hits_h1; h1_rel_index += blockDim.x) {
    // The output we are searching for
    uint16_t best_h0 = 0;
    uint16_t best_h2 = 0;
    float best_fit = max_scatter_seeding;

    // Fetch h1
    const auto h1_index_total = h1_indices[h1_rel_index];
    const uint16_t h1_index = h1_index_total & bits::hit_number;
    const bool oddity = h1_index_total >> bits::oddity_position;

    const Velo::HitBase h1 {
      velo_cluster_container.x(h1_index), velo_cluster_container.y(h1_index), velo_cluster_container.z(h1_index)};

    const auto h1_phi = hit_phi[h1_index];

    // Get candidates on previous module
    uint best_h0s[number_of_h0_candidates];

    // Iterate over previous module until the first n candidates are found
    int phi_index = binary_search_leftmost(
      hit_phi + module_data[shared::previous_module_pair + oddity].hitStart,
      module_data[shared::previous_module_pair + oddity].hitNums,
      h1_phi);

    // Do a "pendulum search" to find the candidates, consisting in iterating in the following manner:
    // phi_index, phi_index + 1, phi_index - 1, phi_index + 2, ...
    int found_h0_candidates = 0;
    for (uint i = 0; i < module_data[shared::previous_module_pair + oddity].hitNums &&
                     found_h0_candidates < number_of_h0_candidates;
         ++i) {
      // Note: By setting the sign to the oddity of i, the search behaviour is achieved.
      const auto sign = i & 0x01;
      const int index_diff = sign ? i : -i;
      phi_index += index_diff;

      const auto index_in_bounds = (phi_index < 0 ? phi_index + module_data[shared::previous_module_pair + oddity].hitNums :
        (phi_index >= static_cast<int>(module_data[shared::previous_module_pair + oddity].hitNums) ?
          phi_index - static_cast<int>(module_data[shared::previous_module_pair + oddity].hitNums) :
          phi_index));
      const auto h0_index = module_data[shared::previous_module_pair + oddity].hitStart + index_in_bounds;

      // Discard the candidate if it is used
      if (!hit_used[h0_index]) {
        best_h0s[found_h0_candidates++] = h0_index;
      }
    }

    // Use the candidates found previously (best_h0s) to find the best triplet
    // Since data is sorted, search using a binary search
    for (int i = 0; i < found_h0_candidates; ++i) {
      const auto h0_index = best_h0s[i];
      const Velo::HitBase h0 {
        velo_cluster_container.x(h0_index), velo_cluster_container.y(h0_index), velo_cluster_container.z(h0_index)};

      const auto td = 1.0f / (h1.z - h0.z);
      const auto txn = (h1.x - h0.x);
      const auto tyn = (h1.y - h0.y);
      const auto tx = txn * td;
      const auto ty = tyn * td;

      // Get candidates by performing a binary search in expected phi
      int candidate_h2 =
        find_seeding_candidate(module_data[shared::next_module_pair + oddity], tx, ty, hit_phi, h0, oddity == 0);

      // Allow a window of hits in the next module. Use pendulum search.
      for (int i = 0; i < number_of_h2_candidates; ++i) {
        const auto sign = i & 0x01;
        const int index_diff = sign ? i : -i;
        candidate_h2 += index_diff;

        const auto h2_index = module_data[shared::next_module_pair + oddity].hitStart + candidate_h2;
        if (
          candidate_h2 >= 0 &&
          candidate_h2 < static_cast<int>(module_data[shared::next_module_pair + oddity].hitNums) &&
          !hit_used[h2_index]) {
          const Velo::HitBase h2 {
            velo_cluster_container.x(h2_index), velo_cluster_container.y(h2_index), velo_cluster_container.z(h2_index)};

          const auto dz = h2.z - h0.z;
          const auto predx = h0.x + tx * dz;
          const auto predy = h0.y + ty * dz;
          const auto dx = predx - h2.x;
          const auto dy = predy - h2.y;

          // Scatter
          const auto scatter = (dx * dx) + (dy * dy);

          // Keep the best one found
          if (scatter < best_fit) {
            best_fit = scatter;
            best_h0 = h0_index;
            best_h2 = h2_index;
          }
        }
      }
    }

    if (best_fit < max_scatter_seeding) {
      // Add the track to the container of seeds
      const auto trackP = atomicAdd(dev_atomics_velo + atomics::number_of_seeds, 1) % max_tracks_to_follow;
      tracklets[trackP] = Velo::TrackletHits {best_h0, h1_index, best_h2};

      // Add the tracks to the bag of tracks to_follow
      // Note: The first bit flag marks this is a tracklet (hitsNum == 3),
      // and hence it is stored in tracklets
      const auto ttfP = atomicAdd(dev_atomics_velo + atomics::tracks_to_follow, 1) % max_tracks_to_follow;
      tracks_to_follow[ttfP] = bits::seed | trackP;
    }
  }
}
