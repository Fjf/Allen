#pragma once

#include "VeloConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "SciFiEventModel.cuh"
#include "SciFiDefinitions.cuh"
#include "DeviceAlgorithm.cuh"
#include "LookingForwardConstants.cuh"
#include "LookingForwardTools.cuh"

<<<<<<< HEAD
namespace lf_triplet_seeding {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    HOST_INPUT(host_number_of_reconstructed_ut_tracks_t, uint);
    DEVICE_INPUT(dev_scifi_hits_t, char) dev_scifi_hits;
    DEVICE_INPUT(dev_scifi_hit_offsets_t, uint) dev_scifi_hit_count;
    DEVICE_INPUT(dev_offsets_all_velo_tracks_t, uint) dev_atomics_velo;
    DEVICE_INPUT(dev_velo_states_t, char) dev_velo_states;
    DEVICE_INPUT(dev_offsets_ut_tracks_t, uint) dev_atomics_ut;
    DEVICE_INPUT(dev_offsets_ut_track_hit_number_t, uint) dev_ut_track_hit_number;
    DEVICE_INPUT(dev_ut_track_velo_indices_t, uint) dev_ut_track_velo_indices;
    DEVICE_INPUT(dev_ut_qop_t, float) dev_ut_qop;
    DEVICE_INPUT(dev_scifi_lf_initial_windows_t, int) dev_scifi_lf_initial_windows;
    DEVICE_INPUT(dev_ut_states_t, MiniState) dev_ut_states;
    DEVICE_INPUT(dev_scifi_lf_process_track_t, bool) dev_scifi_lf_process_track;
    DEVICE_OUTPUT(dev_scifi_lf_found_triplets_t, int) dev_scifi_lf_found_triplets;
    DEVICE_OUTPUT(dev_scifi_lf_number_of_found_triplets_t, int8_t) dev_scifi_lf_number_of_found_triplets;
  };

  __global__ void lf_triplet_seeding(
    Parameters,
    const LookingForward::Constants* dev_looking_forward_constants);

  template<typename T, char... S>
  struct lf_triplet_seeding_t : public DeviceAlgorithm, Parameters {
    constexpr static auto name = Name<S...>::s;
    decltype(global_function(lf_triplet_seeding)) function {lf_triplet_seeding};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const HostBuffers& host_buffers) const
    {
      set_size<dev_scifi_lf_found_triplets_t>(
        arguments,
        value<host_number_of_reconstructed_ut_tracks_t>(arguments) * LookingForward::n_triplet_seeds *
          LookingForward::triplet_seeding_block_dim_x * LookingForward::maximum_number_of_triplets_per_thread);
      set_size<dev_scifi_lf_number_of_found_triplets_t>(
        arguments,
        value<host_number_of_reconstructed_ut_tracks_t>(arguments) * LookingForward::n_triplet_seeds *
          LookingForward::triplet_seeding_block_dim_x);
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      cudaStream_t& cuda_stream,
      cudaEvent_t& cuda_generic_event) const
    {
      cudaCheck(cudaMemsetAsync(
        offset<dev_scifi_lf_number_of_found_triplets_t>(arguments),
        0,
        size<dev_scifi_lf_number_of_found_triplets_t>(arguments),
        cuda_stream));

      function(
        dim3(value<host_number_of_selected_events_t>(arguments)),
        dim3(LookingForward::triplet_seeding_block_dim_x, 2),
        cuda_stream)(
        Parameters {offset<dev_scifi_hits_t>(arguments),
                    offset<dev_scifi_hit_offsets_t>(arguments),
                    offset<dev_offsets_all_velo_tracks_t>(arguments),
                    offset<dev_velo_states_t>(arguments),
                    offset<dev_offsets_ut_tracks_t>(arguments),
                    offset<dev_offsets_ut_track_hit_number_t>(arguments),
                    offset<dev_ut_track_velo_indices_t>(arguments),
                    offset<dev_ut_qop_t>(arguments),
                    offset<dev_scifi_lf_initial_windows_t>(arguments),
                    offset<dev_ut_states_t>(arguments),
                    offset<dev_scifi_lf_process_track_t>(arguments),
                    offset<dev_scifi_lf_found_triplets_t>(arguments),
                    offset<dev_scifi_lf_number_of_found_triplets_t>(arguments)},
        constants.dev_looking_forward_constants);
    }
  };
} // namespace lf_triplet_seeding
=======
__global__ void lf_triplet_seeding(
  uint32_t* dev_scifi_hits,
  const uint32_t* dev_scifi_hit_count,
  const uint* dev_atomics_velo,
  const char* dev_velo_states,
  const uint* dev_atomics_ut,
  const uint* dev_ut_track_hit_number,
  const uint* dev_ut_track_velo_indices,
  const float* dev_ut_qop,
  const char* dev_scifi_geometry,
  const float* dev_inv_clus_res,
  const int* dev_initial_windows,
  const LookingForward::Constants* dev_looking_forward_constants,
  const MiniState* dev_ut_states,
  const bool* dev_scifi_lf_process_track,
  int* dev_scifi_lf_found_triplets,
  int8_t* dev_scifi_lf_number_of_found_triplets);

ALGORITHM(
  lf_triplet_seeding,
  lf_triplet_seeding_t,
  ARGUMENTS(
    dev_scifi_hits,
    dev_scifi_hit_count,
    dev_atomics_ut,
    dev_ut_qop,
    dev_scifi_lf_initial_windows,
    dev_ut_states,
    dev_ut_track_hit_number,
    dev_ut_track_velo_indices,
    dev_atomics_velo,
    dev_velo_states,
    dev_scifi_lf_process_track,
    dev_scifi_lf_found_triplets,
    dev_scifi_lf_number_of_found_triplets))
>>>>>>> origin/master
