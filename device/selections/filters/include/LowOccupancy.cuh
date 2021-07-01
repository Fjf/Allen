/************************************************************************ \
 * (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration      *
\*************************************************************************/
#pragma once

#include "DeviceAlgorithm.cuh"
#include "VeloConsolidated.cuh"

namespace low_occupancy {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    HOST_OUTPUT(host_number_of_selected_events_t, unsigned) host_number_of_selected_events;
    HOST_OUTPUT(host_event_list_output_t, unsigned) host_event_list_output;

    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    DEVICE_INPUT(dev_offsets_velo_tracks_t, unsigned) dev_offsets_velo_tracks;
    DEVICE_INPUT(dev_offsets_velo_track_hit_number_t, unsigned) dev_offsets_velo_track_hit_number;
    DEVICE_OUTPUT(dev_number_of_selected_events_t, unsigned) dev_number_of_selected_events;
    DEVICE_OUTPUT(dev_event_decisions_t, unsigned) dev_event_decisions;

    MASK_INPUT(dev_event_list_t) dev_event_list;
    MASK_OUTPUT(dev_event_list_output_t) dev_event_list_output;

    PROPERTY(minTracks_t, "minTracks", "minimum number of Velo tracks in the event", unsigned int) minTracks;
    PROPERTY(maxTracks_t, "maxTracks", "minimum number of Velo tracks in the event", unsigned int) maxTracks;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
  };

  __global__ void low_occupancy(Parameters);
  struct low_occupancy_t : public DeviceAlgorithm, Parameters {

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
      const Allen::Context&) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{256, 1, 1}}};
    Property<minTracks_t> m_minTracks {this, 0};
    Property<maxTracks_t> m_maxTracks {this, 99999};
  }; // low_occupancy_t

} // namespace low_occupancy
