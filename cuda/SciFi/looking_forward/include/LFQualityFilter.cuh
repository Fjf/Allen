#pragma once

#include "LookingForwardConstants.cuh"
#include "LookingForwardTools.cuh"
#include "SciFiEventModel.cuh"
#include "DeviceAlgorithm.cuh"
#include "VeloConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "LookingForwardTools.cuh"

namespace lf_quality_filter {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    HOST_INPUT(host_number_of_reconstructed_ut_tracks_t, uint);
    DEVICE_INPUT(dev_scifi_hits_t, char) dev_scifi_hits;
    DEVICE_INPUT(dev_scifi_hit_offsets_t, uint) dev_scifi_hit_count;
    DEVICE_INPUT(dev_offsets_ut_tracks_t, uint) dev_atomics_ut;
    DEVICE_INPUT(dev_offsets_ut_track_hit_number_t, uint) dev_ut_track_hit_number;
    DEVICE_INPUT(dev_scifi_lf_length_filtered_tracks_t, SciFi::TrackHits) dev_scifi_lf_length_filtered_tracks;
    DEVICE_INPUT(dev_scifi_lf_length_filtered_atomics_t, uint) dev_scifi_lf_length_filtered_atomics;
    DEVICE_OUTPUT(dev_lf_quality_of_tracks_t, float) dev_scifi_quality_of_tracks;
    DEVICE_OUTPUT(dev_atomics_scifi_t, uint) dev_atomics_scifi;
    DEVICE_OUTPUT(dev_scifi_tracks_t, SciFi::TrackHits) dev_scifi_tracks;
    DEVICE_INPUT(dev_scifi_lf_parametrization_length_filter_t, float) dev_scifi_lf_parametrization_length_filter;
    DEVICE_OUTPUT(dev_scifi_lf_y_parametrization_length_filter_t, float) dev_scifi_lf_y_parametrization_length_filter;
    DEVICE_OUTPUT(dev_scifi_lf_parametrization_consolidate_t, float) dev_scifi_lf_parametrization_consolidate;
    DEVICE_INPUT(dev_ut_states_t, MiniState) dev_ut_states;
    DEVICE_INPUT(dev_velo_states_t, char) dev_velo_states;
    DEVICE_INPUT(dev_offsets_all_velo_tracks_t, uint) dev_atomics_velo;
    DEVICE_INPUT(dev_offsets_velo_track_hit_number_t, uint) dev_velo_track_hit_number;
    DEVICE_INPUT(dev_ut_track_velo_indices_t, uint) dev_ut_track_velo_indices;
    PROPERTY(block_dim_t, DeviceDimensions, "block_dim", "block dimensions");
  };

  __global__ void lf_quality_filter(
    Parameters,
    const LookingForward::Constants* dev_looking_forward_constants,
    const float* dev_magnet_polarity);

  template<typename T, char... S>
  struct lf_quality_filter_t : public DeviceAlgorithm, Parameters {
    constexpr static auto name = Name<S...>::s;
    decltype(global_function(lf_quality_filter)) function {lf_quality_filter};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const
    {
      set_size<dev_atomics_scifi_t>(
        arguments, value<host_number_of_selected_events_t>(arguments) * LookingForward::num_atomics);
      set_size<dev_scifi_tracks_t>(
        arguments,
        value<host_number_of_reconstructed_ut_tracks_t>(arguments) * SciFi::Constants::max_SciFi_tracks_per_UT_track);
      set_size<dev_scifi_lf_y_parametrization_length_filter_t>(
        arguments,
        2 * value<host_number_of_reconstructed_ut_tracks_t>(arguments) *
          LookingForward::maximum_number_of_candidates_per_ut_track);
      set_size<dev_scifi_lf_parametrization_consolidate_t>(
        arguments,
        6 * value<host_number_of_reconstructed_ut_tracks_t>(arguments) *
          SciFi::Constants::max_SciFi_tracks_per_UT_track);
      set_size<dev_lf_quality_of_tracks_t>(
        arguments,
        LookingForward::maximum_number_of_candidates_per_ut_track * 
        value<host_number_of_reconstructed_ut_tracks_t>(arguments));
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      cudaStream_t& cuda_stream,
      cudaEvent_t&) const
    {
      initialize<dev_atomics_scifi_t>(arguments, 0, cuda_stream);

      function(dim3(value<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(
        Parameters {begin<dev_scifi_hits_t>(arguments),
                    begin<dev_scifi_hit_offsets_t>(arguments),
                    begin<dev_offsets_ut_tracks_t>(arguments),
                    begin<dev_offsets_ut_track_hit_number_t>(arguments),
                    begin<dev_scifi_lf_length_filtered_tracks_t>(arguments),
                    begin<dev_scifi_lf_length_filtered_atomics_t>(arguments),
                    begin<dev_lf_quality_of_tracks_t>(arguments),
                    begin<dev_atomics_scifi_t>(arguments),
                    begin<dev_scifi_tracks_t>(arguments),
                    begin<dev_scifi_lf_parametrization_length_filter_t>(arguments),
                    begin<dev_scifi_lf_y_parametrization_length_filter_t>(arguments),
                    begin<dev_scifi_lf_parametrization_consolidate_t>(arguments),
                    begin<dev_ut_states_t>(arguments),
                    begin<dev_velo_states_t>(arguments),
                    begin<dev_offsets_all_velo_tracks_t>(arguments),
                    begin<dev_offsets_velo_track_hit_number_t>(arguments),
                    begin<dev_ut_track_velo_indices_t>(arguments)},
        constants.dev_looking_forward_constants,
        constants.dev_magnet_polarity.data());

      if (runtime_options.do_check) {
        cudaCheck(cudaMemcpyAsync(
          host_buffers.host_atomics_scifi,
          begin<dev_atomics_scifi_t>(arguments),
          size<dev_atomics_scifi_t>(arguments),
          cudaMemcpyDeviceToHost,
          cuda_stream));

        cudaCheck(cudaMemcpyAsync(
          host_buffers.host_scifi_tracks,
          begin<dev_scifi_tracks_t>(arguments),
          size<dev_scifi_tracks_t>(arguments),
          cudaMemcpyDeviceToHost,
          cuda_stream));
      }
    }

  private:
    Property<block_dim_t> m_block_dim {this, {{128, 1, 1}}};
  };
} // namespace lf_quality_filter