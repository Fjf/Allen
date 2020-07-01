/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/

#include "TrackForwarding.cuh"
#include "ClusteringDefinitions.cuh"
#include "SearchByTriplet.cuh"
#include "VeloTools.cuh"
#include "Vector.h"
#include <cstdio>
#include <array>
#include <algorithm>

using namespace Velo::Tracking;
using namespace Allen::device;

void velo_search_by_triplet::velo_search_by_triplet_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_tracks_t>(arguments, first<host_number_of_selected_events_t>(arguments) * Velo::Constants::max_tracks);
  set_size<dev_tracklets_t>(
    arguments, first<host_number_of_selected_events_t>(arguments) * Velo::Constants::max_tracks_to_follow);
  set_size<dev_tracks_to_follow_t>(
    arguments, first<host_number_of_selected_events_t>(arguments) * Velo::Constants::max_tracks_to_follow);
  set_size<dev_three_hit_tracks_t>(
    arguments, first<host_number_of_selected_events_t>(arguments) * Velo::Constants::max_three_hit_tracks);
  set_size<dev_hit_used_t>(arguments, first<host_total_number_of_velo_clusters_t>(arguments));
  set_size<dev_atomics_velo_t>(arguments, first<host_number_of_selected_events_t>(arguments) * Velo::num_atomics);
  set_size<dev_number_of_velo_tracks_t>(arguments, first<host_number_of_selected_events_t>(arguments));
  set_size<dev_rel_indices_t>(
    arguments, first<host_number_of_selected_events_t>(arguments) * Velo::Constants::max_numhits_in_module_pair);
}

void velo_search_by_triplet::velo_search_by_triplet_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  HostBuffers&,
  cudaStream_t& cuda_stream,
  cudaEvent_t&) const
{
  initialize<dev_atomics_velo_t>(arguments, 0, cuda_stream);
  initialize<dev_hit_used_t>(arguments, 0, cuda_stream);
  initialize<dev_number_of_velo_tracks_t>(arguments, 0, cuda_stream);

  global_function(velo_search_by_triplet)(
    dim3(first<host_number_of_selected_events_t>(arguments)), dim3(property<block_dim_x_t>().get()), cuda_stream)(
    arguments, constants.dev_velo_geometry);
}

/**
 * @brief Track forwarding algorithm based on triplet finding.
 *
 * @detail Search by triplet is a parallel local track forwarding algorithm, whose main building blocks are two steps:
 *         track seeding and track forwarding. These two steps are applied iteratively
 *         throughout the VELO detector. The forward end of the detector is used as the start of the search,
 *         and the detector is traversed in the backwards direction, in groups of two modules at a time:
 *
 *         i-3    i-2   [i-1   i   i+1]
 *                      =============== Track seeding of triplet of module pairs {i-1, i, i+1}
 *
 *         i-3   [i-2] [i-1   i   i+1]
 *               =====                  Track forwarding to module pair i-2
 *
 *         i-3   [i-2   i-1   i]  i+1
 *               ===============        Track seeding of triplet of module pairs {i-2, i-1, i}
 *
 *         [i-3] [i-2   i-1   i   i+1]
 *         =====                        Track forwarding to module pair i-3
 *
 *         [i-3   i-2   i-1]  i   i+1
 *         =================            Track seeding of triplet of module pairs {i-3, i-2, i-1}
 *
 *         * Track seeding: Triplets of hits in consecutive module pairs are sought.
 *         Incoming VELO cluster data is expected to be sorted by phi previously. This fact allows for several
 *         optimizations in the triplet seed search. First, the closest hit in the previous module pair is sought with
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
  // Shared memory size is a constant, enough to fit information about three module pairs.
  __shared__ Velo::ModulePair module_pair_data[3];

  // Initialize event number and number of events based on kernel invoking parameters
  const unsigned event_number = blockIdx.x;
  const unsigned number_of_events = gridDim.x;

  // Pointers to data within the event
  const unsigned tracks_offset = event_number * Velo::Constants::max_tracks;
  const unsigned total_estimated_number_of_clusters =
    parameters.dev_offsets_estimated_input_size[Velo::Constants::n_module_pairs * number_of_events];
  const unsigned* module_hit_start =
    parameters.dev_offsets_estimated_input_size + event_number * Velo::Constants::n_module_pairs;
  const unsigned* module_hit_num = parameters.dev_module_cluster_num + event_number * Velo::Constants::n_module_pairs;
  const unsigned hit_offset = module_hit_start[0];

  const auto velo_cluster_container =
    Velo::ConstClusters {parameters.dev_sorted_velo_cluster_container, total_estimated_number_of_clusters, hit_offset};

  const auto hit_phi = parameters.dev_hit_phi + hit_offset;

  Velo::TrackHits* tracks = parameters.dev_tracks + tracks_offset;
  bool* hit_used = parameters.dev_hit_used + hit_offset;

  unsigned* tracks_to_follow = parameters.dev_tracks_to_follow + event_number * Velo::Constants::max_tracks_to_follow;
  Velo::TrackletHits* three_hit_tracks =
    parameters.dev_three_hit_tracks + event_number * Velo::Constants::max_three_hit_tracks;
  Velo::TrackletHits* tracklets = parameters.dev_tracklets + event_number * Velo::Constants::max_tracks_to_follow;
  unsigned short* h1_rel_indices =
    parameters.dev_rel_indices + event_number * Velo::Constants::max_numhits_in_module_pair;

  unsigned* dev_atomics_velo = parameters.dev_atomics_velo + blockIdx.x * Velo::num_atomics;
  const int16_t phi_tolerance = hit_phi_float_to_16(parameters.phi_tolerance);

  unsigned first_module_pair = Velo::Constants::n_module_pairs - 1;

  // Prepare the first seeding iteration
  // Load shared module information
  for (unsigned i = local_id<0>(); i < 3; i += local_size<0>()) {
    const auto module_pair_number = first_module_pair - i;
    module_pair_data[i].hit_start = module_hit_start[module_pair_number] - hit_offset;
    module_pair_data[i].hit_num = module_hit_num[module_pair_number];
    module_pair_data[i].z[0] = dev_velo_geometry->module_zs[2 * module_pair_number];
    module_pair_data[i].z[1] = dev_velo_geometry->module_zs[2 * module_pair_number + 1];
  }

  // Due to shared module data initialization
  barrier();

  // Do first track seeding
  track_seeding(
    velo_cluster_container,
    module_pair_data,
    hit_used,
    tracklets,
    tracks_to_follow,
    h1_rel_indices,
    dev_atomics_velo,
    parameters.max_scatter,
    hit_phi,
    phi_tolerance);

  // Prepare forwarding - seeding loop
  // For an explanation on ttf, see below
  unsigned last_ttf = 0;
  --first_module_pair;

  while (first_module_pair > 1) {
    // Due to WAR between track_seeding and population of shared memory.
    barrier();

    // Iterate in modules
    // Load in shared
    for (int i = local_id<0>(); i < 3; i += local_size<0>()) {
      const auto module_pair_number = first_module_pair - i;
      module_pair_data[i].hit_start = module_hit_start[module_pair_number] - hit_offset;
      module_pair_data[i].hit_num = module_hit_num[module_pair_number];
      module_pair_data[i].z[0] = dev_velo_geometry->module_zs[2 * module_pair_number];
      module_pair_data[i].z[1] = dev_velo_geometry->module_zs[2 * module_pair_number + 1];
    }

    // ttf stands for "tracks to forward"
    // The tracks to forward are stored in a circular buffer.
    const auto prev_ttf = last_ttf;
    last_ttf = dev_atomics_velo[atomics::tracks_to_follow];
    const auto diff_ttf = last_ttf - prev_ttf;

    // Reset local number of hits
    dev_atomics_velo[atomics::local_number_of_hits] = 0;

    // Due to module data loading
    barrier();

    // Track Forwarding
    track_forwarding(
      velo_cluster_container,
      hit_phi,
      hit_used,
      module_pair_data,
      diff_ttf,
      tracks_to_follow,
      three_hit_tracks,
      prev_ttf,
      tracklets,
      tracks,
      dev_atomics_velo,
      parameters.dev_number_of_velo_tracks,
      phi_tolerance,
      parameters.max_scatter,
      parameters.max_skipped_modules);

    // Due to module data reading
    barrier();

    // Seeding
    track_seeding(
      velo_cluster_container,
      module_pair_data,
      hit_used,
      tracklets,
      tracks_to_follow,
      h1_rel_indices,
      dev_atomics_velo,
      parameters.max_scatter,
      hit_phi,
      phi_tolerance);

    --first_module_pair;
  }

  // Due to last seeding
  barrier();

  const auto prev_ttf = last_ttf;
  last_ttf = dev_atomics_velo[atomics::tracks_to_follow];
  const auto diff_ttf = last_ttf - prev_ttf;

  // Process the last bunch of track_to_follows
  for (unsigned ttf_element = local_id<0>(); ttf_element < diff_ttf; ttf_element += local_size<0>()) {
    const auto full_track_number = tracks_to_follow[(prev_ttf + ttf_element) % Velo::Constants::max_tracks_to_follow];
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
 * @brief Search for compatible triplets in
 *        three neighbouring modules on one side
 */
__device__ void track_seeding(
  Velo::ConstClusters& velo_cluster_container,
  const Velo::ModulePair* module_pair_data,
  const bool* hit_used,
  Velo::TrackletHits* tracklets,
  unsigned* tracks_to_follow,
  unsigned short* h1_indices,
  unsigned* dev_atomics_velo,
  const float max_scatter,
  const int16_t* hit_phi,
  const int16_t phi_tolerance)
{
  // Add to an array all non-used h1 hits
  for (unsigned h1_rel_index = local_id<0>(); h1_rel_index < module_pair_data[shared::current_module_pair].hit_num;
       h1_rel_index += local_size<0>()) {
    const auto h1_index = module_pair_data[shared::current_module_pair].hit_start + h1_rel_index;
    if (!hit_used[h1_index]) {
      const auto current_hit = atomicAdd(dev_atomics_velo + atomics::local_number_of_hits, 1);
      h1_indices[current_hit] = h1_index;
    }
  }

  // Due to h1_indices
  barrier();

  // Assign a h1 to each local_id<0>()
  const auto number_of_hits_h1 = dev_atomics_velo[atomics::local_number_of_hits];
  for (unsigned h1_rel_index = local_id<0>(); h1_rel_index < number_of_hits_h1; h1_rel_index += local_size<0>()) {
    // Fetch h1
    const auto h1_index = h1_indices[h1_rel_index];

    track_seeding_impl(
      h1_index,
      velo_cluster_container,
      module_pair_data,
      hit_used,
      tracklets,
      tracks_to_follow,
      dev_atomics_velo,
      max_scatter,
      hit_phi,
      phi_tolerance);
  }
}

#if defined(TARGET_DEVICE_CPU)
/**
 * @brief Search for compatible triplets in
 *        three neighbouring modules on one side
 */
__device__ void track_seeding_vectorized(
  Velo::ConstClusters& velo_cluster_container,
  const Velo::ModulePair* module_pair_data,
  const bool* hit_used,
  Velo::TrackletHits* tracklets,
  unsigned* tracks_to_follow,
  uint16_t*,
  unsigned* dev_atomics_velo,
  const float max_scatter,
  const int16_t* hit_phi,
  const int16_t phi_tolerance)
{
  // Iterate over non-used h1 indices
  std::array<uint16_t, vector_length()> h1_indices_local;
  unsigned h1_rel_index = 0;

  while (h1_rel_index < module_pair_data[shared::current_module_pair].hit_num) {
    unsigned h1_elements_v = 0;
    while (h1_elements_v < vector_length() && h1_rel_index < module_pair_data[shared::current_module_pair].hit_num) {
      const auto h1_index = module_pair_data[shared::current_module_pair].hit_start + h1_rel_index;
      if (!hit_used[h1_index]) {
        h1_indices_local[h1_elements_v++] = h1_index;
      }
      h1_rel_index++;
    }

    if (h1_elements_v < vector_length()) {
      // Do these iterations sequentially
      for (unsigned i = 0; i < h1_elements_v; ++i) {
        track_seeding_impl(
          h1_indices_local[i],
          velo_cluster_container,
          module_pair_data,
          hit_used,
          tracklets,
          tracks_to_follow,
          dev_atomics_velo,
          max_scatter,
          hit_phi,
          phi_tolerance);
      }
    }
    else {
      // Get best h0s to work with
      std::array<unsigned, vector_length() * number_of_h0_candidates> best_h0s_local;
      std::array<unsigned, vector_length()> found_h0_candidates_local;
      unsigned found_h0_candidates_max = 0;

      for (unsigned i_vec = 0; i_vec < vector_length(); ++i_vec) {
        const auto h1_index = h1_indices_local[i_vec];
        auto& found_h0_candidates = found_h0_candidates_local[i_vec];

        // Iterate over previous module until the first n candidates are found
        auto phi_index = binary_search_leftmost(
          hit_phi + module_pair_data[shared::previous_module_pair].hit_start,
          module_pair_data[shared::previous_module_pair].hit_num,
          hit_phi[h1_index]);

        // Do a "pendulum search" to find the candidates, consisting in iterating in the following manner:
        // phi_index, phi_index + 1, phi_index - 1, phi_index + 2, ...
        found_h0_candidates = 0;
        for (unsigned i = 0; i < module_pair_data[shared::previous_module_pair].hit_num &&
                             found_h0_candidates < number_of_h0_candidates;
             ++i) {
          // Note: By setting the sign to the oddity of i, the search behaviour is achieved.
          const auto sign = i & 0x01;
          const int index_diff = sign ? i : -i;
          phi_index += index_diff;

          const auto index_in_bounds =
            (phi_index < 0 ? phi_index + module_pair_data[shared::previous_module_pair].hit_num :
                             (phi_index >= static_cast<int>(module_pair_data[shared::previous_module_pair].hit_num) ?
                                phi_index - static_cast<int>(module_pair_data[shared::previous_module_pair].hit_num) :
                                phi_index));
          const auto h0_index = module_pair_data[shared::previous_module_pair].hit_start + index_in_bounds;

          // Discard the candidate if it is used
          if (!hit_used[h0_index]) {
            best_h0s_local[vector_length() * found_h0_candidates + i_vec] = h0_index;
            found_h0_candidates++;
          }
        }

        if (found_h0_candidates > found_h0_candidates_max) {
          found_h0_candidates_max = found_h0_candidates;
        }
      }

      // Output
      Vector<uint16_t> best_h0 = 0;
      Vector<uint16_t> best_h2 = 0;
      Vector<float> best_fit = max_scatter;

      // There are vector_length() elements to process
      Vector<float> h1_xs;
      Vector<float> h1_ys;
      Vector<float> h1_zs;

      for (unsigned i = 0; i < vector_length(); ++i) {
        const auto h1_index = h1_indices_local[i];
        h1_xs[i] = velo_cluster_container.x(h1_index);
        h1_ys[i] = velo_cluster_container.y(h1_index);
        h1_zs[i] = velo_cluster_container.z(h1_index);
      }

      // print_vector(h1_xs, "h1_xs");
      // print_vector(h1_ys, "h1_ys");
      // print_vector(h1_zs, "h1_zs");

      // Use the candidates found previously (best_h0s) to find the best triplet
      // Since data is sorted, search using a binary search
      for (unsigned found_h0_candidates_iteration = 0; found_h0_candidates_iteration < found_h0_candidates_max;
           ++found_h0_candidates_iteration) {
        // Gather h0 x, y and zs, and make execution mask
        Vector<uint16_t> h0_indices;
        Vector<float> h0_xs;
        Vector<float> h0_ys;
        Vector<float> h0_zs;

        for (unsigned i = 0; i < vector_length(); ++i) {
          if (found_h0_candidates_local[i] > found_h0_candidates_iteration) {
            const auto h0_index = best_h0s_local[vector_length() * found_h0_candidates_iteration + i];
            h0_indices[i] = h0_index;
            h0_xs[i] = velo_cluster_container.x(h0_index);
            h0_ys[i] = velo_cluster_container.y(h0_index);
            h0_zs[i] = velo_cluster_container.z(h0_index);
          }
        }

        // print_vector(h0_indices, "h0_indices");
        // print_vector(h0_xs, "h0_xs");
        // print_vector(h0_ys, "h0_ys");
        // print_vector(h0_zs, "h0_zs");

        const auto h1_h0_diff = h1_zs - h0_zs;
        const auto td = 1.0f / h1_h0_diff;
        const auto txn = (h1_xs - h0_xs);
        const auto tyn = (h1_ys - h0_ys);
        const auto tx = txn * td;
        const auto ty = tyn * td;

        // Calculate phi extrapolation
        const auto dz =
          module_pair_data[shared::next_module_pair].z[0] - module_pair_data[shared::previous_module_pair].z[0];
        const auto predx = tx * dz;
        const auto predy = ty * dz;
        const auto x_prediction = h0_xs + predx;
        const auto y_prediction = h0_ys + predy;
        const auto atan_value_f =
          (Velo::Tools::cudart_pi_f_float + fast_atan2f(y_prediction, x_prediction)) * Velo::Tools::convert_factor -
          phi_tolerance;

        // print_vector(atan_value_f, "atan_value_f");

        // Get h2 candidates
        constexpr unsigned number_of_h2_candidates = 3;
        std::array<unsigned, number_of_h2_candidates * vector_length()> best_h2s_local;
        std::array<unsigned, vector_length()> found_h2_candidates_local;
        unsigned found_h2_candidates_max = 0;

        for (unsigned i_vec = 0; i_vec < vector_length(); ++i_vec) {
          found_h2_candidates_local[i_vec] = 0;
          if (found_h0_candidates_local[i_vec] > found_h0_candidates_iteration) {
            auto& found_h2_candidates = found_h2_candidates_local[i_vec];

            const uint16_t atan_value_u16 = static_cast<uint16_t>(atan_value_f[i_vec]);
            const int16_t* atan_value_i16p = reinterpret_cast<const int16_t*>(&atan_value_u16);
            const int16_t atan_value_i16 = atan_value_i16p[0];

            // printf("atan_value_i16: %i\n", atan_value_i16);

            // Get candidates by performing a binary search in expected phi
            const auto candidate_h2_index_found = binary_search_leftmost(
              hit_phi + module_pair_data[shared::next_module_pair].hit_start,
              module_pair_data[shared::next_module_pair].hit_num,
              atan_value_i16);

            // First candidate in the next module pair.
            // Since the buffer is circular, finding the container size means finding the first element.
            const auto candidate_h2_index =
              candidate_h2_index_found < static_cast<int>(module_pair_data[shared::next_module_pair].hit_num) ?
                candidate_h2_index_found :
                0;

            for (unsigned i = 0; i < module_pair_data[shared::next_module_pair].hit_num &&
                                 found_h2_candidates < number_of_h2_candidates;
                 ++i) {
              const auto index_in_bounds =
                (candidate_h2_index + i) % module_pair_data[shared::next_module_pair].hit_num;
              const auto h2_index = module_pair_data[shared::next_module_pair].hit_start + index_in_bounds;

              // Check the phi difference is within the tolerance with modulo arithmetic.
              const int16_t phi_diff = hit_phi[h2_index] - atan_value_i16;
              const int16_t abs_phi_diff = phi_diff < 0 ? -phi_diff : phi_diff;
              if (abs_phi_diff > phi_tolerance) {
                break;
              }

              if (!hit_used[h2_index]) {
                best_h2s_local[vector_length() * found_h2_candidates + i_vec] = h2_index;
                found_h2_candidates++;
              }
            }

            if (found_h2_candidates > found_h2_candidates_max) {
              found_h2_candidates_max = found_h2_candidates;
            }
          }
        }

        // printf("found_h2_candidates_max: %i\n", found_h2_candidates_max);

        // Iterate over h2s
        for (unsigned found_h2_candidates_iteration = 0; found_h2_candidates_iteration < found_h2_candidates_max;
             ++found_h2_candidates_iteration) {
          // Gather h0 x, y and zs, and make execution mask
          Vector<uint16_t> h2_indices;
          Vector<float> h2_xs;
          Vector<float> h2_ys;
          Vector<float> h2_zs;
          Vector<bool> active;

          for (unsigned i = 0; i < vector_length(); ++i) {
            if (found_h2_candidates_local[i] > found_h2_candidates_iteration) {
              active.insert(i, true);
              const auto h2_index = best_h2s_local[vector_length() * found_h2_candidates_iteration + i];
              h2_indices[i] = h2_index;
              h2_xs[i] = velo_cluster_container.x(h2_index);
              h2_ys[i] = velo_cluster_container.y(h2_index);
              h2_zs[i] = velo_cluster_container.z(h2_index);
            }
          }

          // print_vector(h2_xs, "h2_xs");
          // print_vector(h2_ys, "h2_ys");
          // print_vector(h2_zs, "h2_zs");

          const auto dz = h2_zs - h0_zs;
          const auto predx = h0_xs + tx * dz;
          const auto predy = h0_ys + ty * dz;
          const auto dx = predx - h2_xs;
          const auto dy = predy - h2_ys;

          // Scatter
          const auto scatter = dx * dx + dy * dy;

          // Keep the best one found
          const Vector<bool> mask = scatter < best_fit && active;

          best_fit = best_fit.blend(mask, scatter);
          best_h0 = best_h0.blend(mask, h0_indices);
          best_h2 = best_h2.blend(mask, h2_indices);
        }
      }

      for (unsigned i = 0; i < vector_length(); ++i) {
        const auto best_fit_s = best_fit[i];
        if (best_fit_s < max_scatter) {
          const auto best_h0_s = best_h0[i];
          const auto h1_index_s = h1_indices_local[i];
          const auto best_h2_s = best_h2[i];

          // Add the track to the container of seeds
          const auto trackP =
            atomicAdd(dev_atomics_velo + atomics::number_of_seeds, 1) % Velo::Constants::max_tracks_to_follow;
          tracklets[trackP] = Velo::TrackletHits {best_h0_s, h1_index_s, best_h2_s};

          // Add the tracks to the bag of tracks to_follow
          // Note: The first bit flag marks this is a tracklet (hitsNum == 3),
          // and hence it is stored in tracklets
          const auto ttfP =
            atomicAdd(dev_atomics_velo + atomics::tracks_to_follow, 1) % Velo::Constants::max_tracks_to_follow;
          tracks_to_follow[ttfP] = bits::seed | trackP;
        }
      }
    }
  }
}
#endif

__device__ void track_seeding_impl(
  const uint16_t h1_index,
  Velo::ConstClusters& velo_cluster_container,
  const Velo::ModulePair* module_pair_data,
  const bool* hit_used,
  Velo::TrackletHits* tracklets,
  unsigned* tracks_to_follow,
  unsigned* dev_atomics_velo,
  const float max_scatter,
  const int16_t* hit_phi,
  const int16_t phi_tolerance)
{
  // The output we are searching for
  uint16_t best_h0 = 0;
  uint16_t best_h2 = 0;
  float best_fit = max_scatter;

  const Velo::HitBase h1 {
    velo_cluster_container.x(h1_index), velo_cluster_container.y(h1_index), velo_cluster_container.z(h1_index)};

  const auto h1_phi = hit_phi[h1_index];

  // Get candidates on previous module
  std::array<unsigned, number_of_h0_candidates> best_h0s;

  // Iterate over previous module until the first n candidates are found
  auto phi_index = binary_search_leftmost(
    hit_phi + module_pair_data[shared::previous_module_pair].hit_start,
    module_pair_data[shared::previous_module_pair].hit_num,
    h1_phi);

  // Do a "pendulum search" to find the candidates, consisting in iterating in the following manner:
  // phi_index, phi_index + 1, phi_index - 1, phi_index + 2, ...
  int found_h0_candidates = 0;
  for (unsigned i = 0;
       i < module_pair_data[shared::previous_module_pair].hit_num && found_h0_candidates < number_of_h0_candidates;
       ++i) {
    // Note: By setting the sign to the oddity of i, the search behaviour is achieved.
    const auto sign = i & 0x01;
    const int index_diff = sign ? i : -i;
    phi_index += index_diff;

    const auto index_in_bounds =
      (phi_index < 0 ? phi_index + module_pair_data[shared::previous_module_pair].hit_num :
                       (phi_index >= static_cast<int>(module_pair_data[shared::previous_module_pair].hit_num) ?
                          phi_index - static_cast<int>(module_pair_data[shared::previous_module_pair].hit_num) :
                          phi_index));
    const auto h0_index = module_pair_data[shared::previous_module_pair].hit_start + index_in_bounds;

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
    const auto candidate_h2 = find_forward_candidate(
      module_pair_data[shared::next_module_pair],
      hit_phi,
      h0,
      tx,
      ty,
      module_pair_data[shared::next_module_pair].z[0] - module_pair_data[shared::previous_module_pair].z[0],
      phi_tolerance);

    // First candidate in the next module pair.
    // Since the buffer is circular, finding the container size means finding the first element.
    const auto candidate_h2_index =
      std::get<0>(candidate_h2) < static_cast<int>(module_pair_data[shared::next_module_pair].hit_num) ?
        std::get<0>(candidate_h2) :
        0;
    const auto extrapolated_phi = std::get<1>(candidate_h2);

    for (unsigned i = 0; i < module_pair_data[shared::next_module_pair].hit_num; ++i) {
      const auto index_in_bounds = (candidate_h2_index + i) % module_pair_data[shared::next_module_pair].hit_num;
      const auto h2_index = module_pair_data[shared::next_module_pair].hit_start + index_in_bounds;

      // Check the phi difference is within the tolerance with modulo arithmetic.
      const int16_t phi_diff = hit_phi[h2_index] - extrapolated_phi;
      const int16_t abs_phi_diff = phi_diff < 0 ? -phi_diff : phi_diff;
      if (abs_phi_diff > phi_tolerance) {
        break;
      }

      if (!hit_used[h2_index]) {
        const Velo::HitBase h2 {
          velo_cluster_container.x(h2_index), velo_cluster_container.y(h2_index), velo_cluster_container.z(h2_index)};

        const auto dz = h2.z - h0.z;
        const auto predx = h0.x + tx * dz;
        const auto predy = h0.y + ty * dz;
        const auto dx = predx - h2.x;
        const auto dy = predy - h2.y;

        // Scatter
        const auto scatter = dx * dx + dy * dy;

        // Keep the best one found
        if (scatter < best_fit) {
          best_fit = scatter;
          best_h0 = h0_index;
          best_h2 = h2_index;
        }
      }
    }
  }

  if (best_fit < max_scatter) {
    // Add the track to the container of seeds
    const auto trackP =
      atomicAdd(dev_atomics_velo + atomics::number_of_seeds, 1) % Velo::Constants::max_tracks_to_follow;
    tracklets[trackP] = Velo::TrackletHits {best_h0, h1_index, best_h2};

    // Add the tracks to the bag of tracks to_follow
    // Note: The first bit flag marks this is a tracklet (hitsNum == 3),
    // and hence it is stored in tracklets
    const auto ttfP =
      atomicAdd(dev_atomics_velo + atomics::tracks_to_follow, 1) % Velo::Constants::max_tracks_to_follow;
    tracks_to_follow[ttfP] = bits::seed | trackP;
  }
}

/**
 * @brief Performs the track forwarding of forming tracks
 */
__device__ void track_forwarding(
  Velo::ConstClusters& velo_cluster_container,
  const int16_t* hit_phi,
  bool* hit_used,
  const Velo::ModulePair* module_pair_data,
  const unsigned diff_ttf,
  unsigned* tracks_to_follow,
  Velo::TrackletHits* three_hit_tracks,
  const unsigned prev_ttf,
  Velo::TrackletHits* tracklets,
  Velo::TrackHits* tracks,
  unsigned* dev_atomics_velo,
  unsigned* dev_number_of_velo_tracks,
  const int16_t phi_tolerance,
  const float max_scatter,
  const unsigned max_skipped_modules)
{
  // Assign a track to follow to each thread
  for (unsigned ttf_element = local_id<0>(); ttf_element < diff_ttf; ttf_element += local_size<0>()) {
    const auto full_track_number = tracks_to_follow[(prev_ttf + ttf_element) % Velo::Constants::max_tracks_to_follow];
    const bool track_flag = (full_track_number & bits::seed) == bits::seed;
    const auto skipped_modules = (full_track_number & bits::skipped_modules) >> bits::skipped_module_position;
    auto track_number = full_track_number & bits::track_number;

    assert(
      track_flag ? track_number < Velo::Constants::max_tracks_to_follow : track_number < Velo::Constants::max_tracks);

    unsigned number_of_hits;
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
    const auto h0_module =
      ((velo_cluster_container.id(h0_num) & Allen::VPChannelID::sensorMask) >> Allen::VPChannelID::sensorBits) / 4;

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
    float best_fit = max_scatter;
    int best_h2 = -1;

    // Get candidates by performing a binary search in expected phi
    const auto candidate_h2 = find_forward_candidate(
      module_pair_data[shared::next_module_pair],
      hit_phi,
      h0,
      tx,
      ty,
      module_pair_data[shared::next_module_pair].z[h0_module % 2] - h0.z,
      phi_tolerance);

    // First candidate in the next module pair.
    // Since the buffer is circular, finding the container size means finding the first element.
    const auto candidate_h2_index =
      std::get<0>(candidate_h2) < static_cast<int>(module_pair_data[shared::next_module_pair].hit_num) ?
        std::get<0>(candidate_h2) :
        0;
    const auto extrapolated_phi = std::get<1>(candidate_h2);

    for (unsigned i = 0; i < module_pair_data[shared::next_module_pair].hit_num; ++i) {
      const auto index_in_bounds = (candidate_h2_index + i) % module_pair_data[shared::next_module_pair].hit_num;
      const auto h2_index = module_pair_data[shared::next_module_pair].hit_start + index_in_bounds;

      // Check the phi difference is within the tolerance with modulo arithmetic.
      const int16_t phi_diff = hit_phi[h2_index] - extrapolated_phi;
      const int16_t abs_phi_diff = phi_diff < 0 ? -phi_diff : phi_diff;
      if (abs_phi_diff > phi_tolerance) {
        break;
      }

      const Velo::HitBase h2 {
        velo_cluster_container.x(h2_index), velo_cluster_container.y(h2_index), velo_cluster_container.z(h2_index)};

      const auto dz = h2.z - h0.z;
      const auto predx = h0.x + tx * dz;
      const auto predy = h0.y + ty * dz;
      const auto dx = predx - h2.x;
      const auto dy = predy - h2.y;

      // Scatter
      const auto scatter = dx * dx + dy * dy;

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
        const auto ttf_p =
          atomicAdd(dev_atomics_velo + atomics::tracks_to_follow, 1) % Velo::Constants::max_tracks_to_follow;
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
      const auto ttf_p =
        atomicAdd(dev_atomics_velo + atomics::tracks_to_follow, 1) % Velo::Constants::max_tracks_to_follow;
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
