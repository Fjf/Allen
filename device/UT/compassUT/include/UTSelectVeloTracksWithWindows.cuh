/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "UTDefinitions.cuh"
#include "UTMagnetToolDefinitions.h"
#include "CompassUTDefinitions.cuh"
#include "DeviceAlgorithm.cuh"

namespace ut_select_velo_tracks_with_windows {
  DEFINE_PARAMETERS(
    Parameters,
    (HOST_INPUT(host_number_of_selected_events_t, unsigned), host_number_of_selected_events),
    (HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned), host_number_of_reconstructed_velo_tracks),
    (DEVICE_INPUT(dev_offsets_all_velo_tracks_t, unsigned), dev_atomics_velo),
    (DEVICE_INPUT(dev_offsets_velo_track_hit_number_t, unsigned), dev_velo_track_hit_number),
    (DEVICE_INPUT(dev_accepted_velo_tracks_t, bool), dev_accepted_velo_tracks),
    (DEVICE_INPUT(dev_ut_number_of_selected_velo_tracks_t, unsigned), dev_ut_number_of_selected_velo_tracks),
    (DEVICE_INPUT(dev_ut_selected_velo_tracks_t, unsigned), dev_ut_selected_velo_tracks),
    (DEVICE_INPUT(dev_ut_windows_layers_t, short), dev_ut_windows_layers),
    (DEVICE_OUTPUT(dev_ut_number_of_selected_velo_tracks_with_windows_t, unsigned), dev_ut_number_of_selected_velo_tracks_with_windows),
    (DEVICE_OUTPUT(dev_ut_selected_velo_tracks_with_windows_t, unsigned), dev_ut_selected_velo_tracks_with_windows),
    (PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions), block_dim))

  __global__ void ut_select_velo_tracks_with_windows(Parameters);

  struct ut_select_velo_tracks_with_windows_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      HostBuffers&,
      cudaStream_t& cuda_stream,
      cudaEvent_t&) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{256, 1, 1}}};
  };
} // namespace ut_select_velo_tracks_with_windows