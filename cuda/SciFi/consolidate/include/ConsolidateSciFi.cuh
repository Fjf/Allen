#pragma once

#include "SciFiEventModel.cuh"
#include "SciFiConsolidated.cuh"
#include "SciFiDefinitions.cuh"
#include "States.cuh"
#include "DeviceAlgorithm.cuh"
#include "LookingForwardConstants.cuh"

namespace scifi_consolidate_tracks {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    HOST_INPUT(host_accumulated_number_of_hits_in_scifi_tracks_t, uint);
    HOST_INPUT(host_number_of_reconstructed_scifi_tracks_t, uint);
    DEVICE_INPUT(dev_scifi_hits_t, uint) dev_scifi_hits;
    DEVICE_INPUT(dev_scifi_hit_count_t, uint) dev_scifi_hit_count;
    DEVICE_OUTPUT(dev_scifi_track_hits_t, char) dev_scifi_track_hits;
    DEVICE_INPUT(dev_atomics_scifi_t, uint) dev_atomics_scifi;
    DEVICE_INPUT(dev_scifi_track_hit_number_t, uint) dev_scifi_track_hit_number;
    DEVICE_OUTPUT(dev_scifi_qop_t, float) dev_scifi_qop;
    DEVICE_OUTPUT(dev_scifi_states_t, MiniState) dev_scifi_states;
    DEVICE_OUTPUT(dev_scifi_track_ut_indices_t, uint) dev_scifi_track_ut_indices;
    DEVICE_INPUT(dev_atomics_ut_t, uint) dev_atomics_ut;
    DEVICE_INPUT(dev_scifi_tracks_t, SciFi::TrackHits) dev_scifi_tracks;
    DEVICE_INPUT(dev_scifi_lf_parametrization_consolidate_t, float) dev_scifi_lf_parametrization_consolidate;
  };

  __global__ void scifi_consolidate_tracks(Parameters, const char* dev_scifi_geometry, const float* dev_inv_clus_res);

  template<typename T>
  struct scifi_consolidate_tracks_t : public DeviceAlgorithm, Parameters {
    constexpr static auto name {"scifi_consolidate_tracks_t"};
    decltype(global_function(scifi_consolidate_tracks)) function {scifi_consolidate_tracks};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const HostBuffers& host_buffers) const
    {
      set_size<dev_scifi_track_hits_t>(
        arguments, value<host_accumulated_number_of_hits_in_scifi_tracks_t>(arguments) * sizeof(SciFi::Hit));
      set_size<dev_scifi_qop_t>(arguments, value<host_number_of_reconstructed_scifi_tracks_t>(arguments));
      set_size<dev_scifi_track_ut_indices_t>(arguments, value<host_number_of_reconstructed_scifi_tracks_t>(arguments));
      set_size<dev_scifi_states_t>(arguments, value<host_number_of_reconstructed_scifi_tracks_t>(arguments));
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      cudaStream_t& cuda_stream,
      cudaEvent_t& cuda_generic_event) const
    {
      function(dim3(value<host_number_of_selected_events_t>(arguments)), block_dimension(), cuda_stream)(
        Parameters {offset<dev_scifi_hits_t>(arguments),
                    offset<dev_scifi_hit_count_t>(arguments),
                    offset<dev_scifi_track_hits_t>(arguments),
                    offset<dev_atomics_scifi_t>(arguments),
                    offset<dev_scifi_track_hit_number_t>(arguments),
                    offset<dev_scifi_qop_t>(arguments),
                    offset<dev_scifi_states_t>(arguments),
                    offset<dev_scifi_track_ut_indices_t>(arguments),
                    offset<dev_atomics_ut_t>(arguments),
                    offset<dev_scifi_tracks_t>(arguments),
                    offset<dev_scifi_lf_parametrization_consolidate_t>(arguments)},
        constants.dev_scifi_geometry,
        constants.dev_inv_clus_res);

      // Transmission device to host of Scifi consolidated tracks
      if (runtime_options.do_check) {
        cudaCheck(cudaMemcpyAsync(
          host_buffers.host_atomics_scifi,
          offset<dev_atomics_scifi_t>(arguments),
          size<dev_atomics_scifi_t>(arguments),
          cudaMemcpyDeviceToHost,
          cuda_stream));

        cudaCheck(cudaMemcpyAsync(
          host_buffers.host_scifi_track_hit_number,
          offset<dev_scifi_track_hit_number_t>(arguments),
          size<dev_scifi_track_hit_number_t>(arguments),
          cudaMemcpyDeviceToHost,
          cuda_stream));

        cudaCheck(cudaMemcpyAsync(
          host_buffers.host_scifi_track_hits,
          offset<dev_scifi_track_hits_t>(arguments),
          size<dev_scifi_track_hits_t>(arguments),
          cudaMemcpyDeviceToHost,
          cuda_stream));

        cudaCheck(cudaMemcpyAsync(
          host_buffers.host_scifi_qop,
          offset<dev_scifi_qop_t>(arguments),
          size<dev_scifi_qop_t>(arguments),
          cudaMemcpyDeviceToHost,
          cuda_stream));

        cudaCheck(cudaMemcpyAsync(
          host_buffers.host_scifi_track_ut_indices,
          offset<dev_scifi_track_ut_indices_t>(arguments),
          size<dev_scifi_track_ut_indices_t>(arguments),
          cudaMemcpyDeviceToHost,
          cuda_stream));
      }
    }
  };
} // namespace scifi_consolidate_tracks