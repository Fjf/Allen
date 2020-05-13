#pragma once

#include "UTDefinitions.cuh"
#include "UTMagnetToolDefinitions.h"
#include "CompassUTDefinitions.cuh"
#include "DeviceAlgorithm.cuh"

namespace ut_select_velo_tracks_with_windows {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, uint);
    DEVICE_INPUT(dev_offsets_all_velo_tracks_t, uint) dev_atomics_velo;
    DEVICE_INPUT(dev_offsets_velo_track_hit_number_t, uint) dev_velo_track_hit_number;
    DEVICE_INPUT(dev_velo_states_t, char) dev_velo_states;
    DEVICE_INPUT(dev_accepted_velo_tracks_t, bool) dev_accepted_velo_tracks;
    DEVICE_INPUT(dev_ut_number_of_selected_velo_tracks_t, uint) dev_ut_number_of_selected_velo_tracks;
    DEVICE_INPUT(dev_ut_selected_velo_tracks_t, uint) dev_ut_selected_velo_tracks;
    DEVICE_INPUT(dev_ut_windows_layers_t, short) dev_ut_windows_layers;
    DEVICE_OUTPUT(dev_ut_number_of_selected_velo_tracks_with_windows_t, uint) dev_ut_number_of_selected_velo_tracks_with_windows;
    DEVICE_OUTPUT(dev_ut_selected_velo_tracks_with_windows_t, uint) dev_ut_selected_velo_tracks_with_windows;
    PROPERTY(block_dim_t, DeviceDimensions, "block_dim", "block dimensions");
  };

  __global__ void ut_select_velo_tracks_with_windows(Parameters);

  template<typename T>
  struct ut_select_velo_tracks_with_windows_t : public DeviceAlgorithm, Parameters {

    decltype(global_function(ut_select_velo_tracks_with_windows)) function {ut_select_velo_tracks_with_windows};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const
    {
      set_size<dev_ut_number_of_selected_velo_tracks_with_windows_t>(arguments, first<host_number_of_selected_events_t>(arguments));
      set_size<dev_ut_selected_velo_tracks_with_windows_t>(arguments, first<host_number_of_reconstructed_velo_tracks_t>(arguments));
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions&,
      const Constants&,
      HostBuffers&,
      cudaStream_t& cuda_stream,
      cudaEvent_t&) const
    {
      initialize<dev_ut_number_of_selected_velo_tracks_with_windows_t>(arguments, 0, cuda_stream);

      function(dim3(first<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(
        Parameters {data<dev_offsets_all_velo_tracks_t>(arguments),
                    data<dev_offsets_velo_track_hit_number_t>(arguments),
                    data<dev_velo_states_t>(arguments),
                    data<dev_accepted_velo_tracks_t>(arguments),
                    data<dev_ut_number_of_selected_velo_tracks_t>(arguments),
                    data<dev_ut_selected_velo_tracks_t>(arguments),
                    data<dev_ut_windows_layers_t>(arguments),
                    data<dev_ut_number_of_selected_velo_tracks_with_windows_t>(arguments),
                    data<dev_ut_selected_velo_tracks_with_windows_t>(arguments)});
    }

  private:
    Property<block_dim_t> m_block_dim {this, {{256, 1, 1}}};
  };
} // namespace ut_select_velo_tracks_with_windows