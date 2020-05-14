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
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions);
  };

  __global__ void scifi_copy_track_hit_number(Parameters);

  template<typename T>
  struct scifi_copy_track_hit_number_t : public DeviceAlgorithm, Parameters {

    decltype(global_function(scifi_copy_track_hit_number)) function {scifi_copy_track_hit_number};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const
    {
      set_size<dev_scifi_track_hit_number_t>(arguments, first<host_number_of_reconstructed_scifi_tracks_t>(arguments));
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions&,
      const Constants&,
      HostBuffers&,
      cudaStream_t& cuda_stream,
      cudaEvent_t&) const
    {
      function(dim3(first<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(
        Parameters {data<dev_offsets_ut_tracks_t>(arguments),
                    data<dev_scifi_tracks_t>(arguments),
                    data<dev_offsets_forward_tracks_t>(arguments),
                    data<dev_scifi_track_hit_number_t>(arguments)});
    }
    
  private:
    Property<block_dim_t> m_block_dim {this, {{512, 1, 1}}};
  };
} // namespace scifi_copy_track_hit_number