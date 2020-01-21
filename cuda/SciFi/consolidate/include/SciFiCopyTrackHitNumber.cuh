#pragma once

#include "SciFiEventModel.cuh"
#include "SciFiConsolidated.cuh"
#include "SciFiDefinitions.cuh"
#include "States.cuh"
#include "DeviceAlgorithm.cuh"
#include "LookingForwardConstants.cuh"

namespace scifi_copy_track_hit_number {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    HOST_INPUT(host_number_of_reconstructed_scifi_tracks_t, uint);
    DEVICE_INPUT(dev_offsets_ut_tracks_t, uint) dev_atomics_ut;
    DEVICE_INPUT(dev_scifi_tracks_t, SciFi::TrackHits) dev_scifi_tracks;
    DEVICE_INPUT(dev_offsets_forward_tracks_t, uint) dev_atomics_scifi;
    DEVICE_OUTPUT(dev_scifi_track_hit_number_t, uint) dev_scifi_track_hit_number;
  };

  __global__ void scifi_copy_track_hit_number(Parameters);

  template<typename T, char... S>
  struct scifi_copy_track_hit_number_t : public DeviceAlgorithm, Parameters {
    constexpr static auto name = Name<S...>::s;
    decltype(global_function(scifi_copy_track_hit_number)) function {scifi_copy_track_hit_number};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const HostBuffers& host_buffers) const
    {
      set_size<dev_scifi_track_hit_number_t>(arguments, value<host_number_of_reconstructed_scifi_tracks_t>(arguments));
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
        Parameters {offset<dev_offsets_ut_tracks_t>(arguments),
                    offset<dev_scifi_tracks_t>(arguments),
                    offset<dev_offsets_forward_tracks_t>(arguments),
                    offset<dev_scifi_track_hit_number_t>(arguments)});
    }
  };
} // namespace scifi_copy_track_hit_number