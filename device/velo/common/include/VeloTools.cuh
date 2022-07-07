/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "BackendCommon.h"
#include "ArgumentOps.cuh"
#include "VeloEventModel.cuh"

namespace Velo {
  namespace Tools {
    constexpr float max_input_value = Allen::constants::pi_f_float;
    constexpr float max_output_value_i16 = 1 << 15;
    constexpr float convert_factor_i16 = max_output_value_i16 / max_input_value;
  } // namespace Tools
} // namespace Velo

template<typename T>
__device__ inline T fast_atan2f(const T& y, const T& x)
{
  // error < 0.07 rad, no 0/0 security
  const T c1 = .25f * Allen::constants::pi_f_float;
  const T c2 = .75f * Allen::constants::pi_f_float;
  const T abs_y = fabsf(y);

  const T x_plus_y = x + abs_y;
  const T x_sub_y = x - abs_y;
  const T y_sub_x = abs_y - x;

  const T nom = signselect(x, x_sub_y, x_plus_y);
  const T den = signselect(x, x_plus_y, y_sub_x);

  const T r = nom / den;
  const T angle = signselect(x, c1, c2) - c1 * r;

  return copysignf(angle, y);
}

/**
 * @brief Calculates the hit phi in a int16 format.
 * @details The range of the atan2 function is mapped onto the range of the int16,
 *          such that for two hit phis in this format, the difference is stable
 *          regardless of the values.
 */
__device__ inline int16_t hit_phi_16(const float x, const float y)
{
  // We have to convert the range {-PI, +PI} into {-2^15, (2^15 - 1)} by multiplying
  // by a factor and then reinterpret cast into int16_t.
  const float float_value = fast_atan2f(y, x) * Velo::Tools::convert_factor_i16;
  const int16_t int16_value = static_cast<int16_t>(float_value);
  return int16_value;
}

/**
 * @brief Converts a hit phi from float to int16.
 * @pre   hit_phi is assumed to be in range [0, PI].
 */
__device__ inline int16_t hit_phi_float_to_16(const float hit_phi)
{
  const float float_value = hit_phi * Velo::Tools::convert_factor_i16;
  const int16_t int16_value = static_cast<int16_t>(float_value);
  return int16_value;
}

/**
 * @brief Converts a hit phi from int32 to float.
 */
__device__ inline float hit_phi_16_to_float(const int16_t phi)
{
  return static_cast<float>(phi) / Velo::Tools::convert_factor_i16;
}

/**
 * @brief Prints a VELO cluster container.
 */
template<
  typename VeloContainer,
  typename Offsets,
  typename ClusterNum,
  typename TotalNumberOfClusters,
  typename NumberOfEvents,
  typename Arguments>
__host__ inline void print_velo_clusters(Arguments arguments, const Allen::Context& context)
{
  // Prints the velo clusters
  const auto a = Allen::ArgumentOperations::make_host_buffer<VeloContainer>(arguments, context);
  const auto offsets_estimated_input_size = Allen::ArgumentOperations::make_host_buffer<Offsets>(arguments, context);
  const auto module_cluster_num = Allen::ArgumentOperations::make_host_buffer<ClusterNum>(arguments, context);

  const auto velo_cluster_container =
    Velo::ConstClusters {a.data(), Allen::ArgumentOperations::first<TotalNumberOfClusters>(arguments)};
  for (unsigned event_number = 0; event_number < Allen::ArgumentOperations::first<NumberOfEvents>(arguments);
       ++event_number) {
    const auto event_number_of_hits =
      offsets_estimated_input_size[(event_number + 1) * Velo::Constants::n_module_pairs] -
      offsets_estimated_input_size[event_number * Velo::Constants::n_module_pairs];
    if (event_number_of_hits > 0) {
      std::cout << "Event number " << event_number << " (" << event_number_of_hits << " hits)\n";
      for (unsigned i = 0; i < Velo::Constants::n_module_pairs; ++i) {
        const auto module_hit_start = offsets_estimated_input_size[event_number * Velo::Constants::n_module_pairs + i];
        const auto module_hit_num = module_cluster_num[event_number * Velo::Constants::n_module_pairs + i];

        std::cout << "Module pair " << i << " (offset " << module_hit_start << ", size " << module_hit_num << "):\n";
        for (unsigned hit_number = 0; hit_number < module_hit_num; ++hit_number) {
          const auto hit_index = module_hit_start + hit_number;
          std::cout << " " << velo_cluster_container.x(hit_index) << ", " << velo_cluster_container.y(hit_index) << ", "
                    << velo_cluster_container.z(hit_index) << ", " << velo_cluster_container.phi(hit_index) << ", "
                    << velo_cluster_container.id(hit_index) << "\n";
        }
        std::cout << "\n";
      }
    }
  }
}

/**
 * @brief Prints the VELO track numbers.
 */
template<
  typename VeloTracks,
  typename NumberOfVeloTracks,
  typename VeloTracklets,
  typename NumberOfVeloTracklets,
  typename OffsetsEstimatedInputSize,
  typename Arguments>
__host__ inline void print_velo_tracks(Arguments arguments, const Allen::Context& context)
{
  // Prints the velo clusters
  const auto trackhits = Allen::ArgumentOperations::make_host_buffer<VeloTracks>(arguments, context);
  const auto number_of_velo_tracks =
    Allen::ArgumentOperations::make_host_buffer<NumberOfVeloTracks>(arguments, context);
  const auto tracklethits = Allen::ArgumentOperations::make_host_buffer<VeloTracklets>(arguments, context);
  const auto number_of_velo_tracklets =
    Allen::ArgumentOperations::make_host_buffer<NumberOfVeloTracklets>(arguments, context);
  const auto offsets_estimated_input_size =
    Allen::ArgumentOperations::make_host_buffer<OffsetsEstimatedInputSize>(arguments, context);

  for (unsigned event_number = 0; event_number < number_of_velo_tracks.size(); ++event_number) {
    const auto event_number_of_velo_tracks = number_of_velo_tracks[event_number];
    const auto event_number_of_velo_tracklets =
      number_of_velo_tracklets[event_number * Velo::num_atomics + Velo::Tracking::atomics::number_of_three_hit_tracks];

    std::cout << "Event #" << event_number << ": " << event_number_of_velo_tracks << " tracks and "
              << event_number_of_velo_tracklets << " tracklets:\n";

    const auto track_offset = Velo::track_offset(offsets_estimated_input_size.data(), event_number);
    for (unsigned i = 0; i < event_number_of_velo_tracks; ++i) {
      std::cout << " Track #" << i << ": ";
      const auto track = trackhits[track_offset + i];
      for (unsigned j = 0; j < track.hitsNum; ++j) {
        std::cout << track.hits[j] << ", ";
      }
      std::cout << "\n";
    }
    for (unsigned i = 0; i < event_number_of_velo_tracklets; ++i) {
      std::cout << " Tracklet #" << i << ": ";
      const auto track = tracklethits[track_offset + i];
      for (unsigned j = 0; j < 3; ++j) {
        std::cout << track.hits[j] << ", ";
      }
      std::cout << "\n";
    }

    std::cout << "\n";
  }
}

template<typename VeloTracks, typename NumberOfVeloTracks, typename OffsetsEstimatedInputSize, typename Arguments>
__host__ inline void print_velo_three_hit_tracks(Arguments arguments, const Allen::Context& context)
{
  // Prints the velo clusters
  const auto trackhits = Allen::ArgumentOperations::make_host_buffer<VeloTracks>(arguments, context);
  const auto number_of_velo_tracks =
    Allen::ArgumentOperations::make_host_buffer<NumberOfVeloTracks>(arguments, context);
  const auto offsets_estimated_input_size =
    Allen::ArgumentOperations::make_host_buffer<OffsetsEstimatedInputSize>(arguments, context);

  for (unsigned event_number = 0; event_number < number_of_velo_tracks.size(); ++event_number) {
    const auto event_number_of_velo_tracks = number_of_velo_tracks[event_number];
    std::cout << "Event #" << event_number << ": " << event_number_of_velo_tracks << " tracklets:\n";

    const auto track_offset = Velo::track_offset(offsets_estimated_input_size.data(), event_number);
    for (unsigned i = 0; i < event_number_of_velo_tracks; ++i) {
      std::cout << " Track #" << i << ": ";
      const auto track = trackhits[track_offset + i];
      for (unsigned j = 0; j < 3; ++j) {
        std::cout << track.hits[j] << ", ";
      }
      std::cout << "\n";
    }

    std::cout << "\n";
  }
}
