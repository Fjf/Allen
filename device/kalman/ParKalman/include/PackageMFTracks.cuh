/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "ParKalmanDefinitions.cuh"
#include "ParKalmanMath.cuh"
#include "ParKalmanFittedTrack.cuh"
#include "UTConsolidated.cuh"
#include "VeloConsolidated.cuh"
#include "States.cuh"
#include "AlgorithmTypes.cuh"

namespace package_mf_tracks {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_mf_tracks_t, unsigned) host_number_of_mf_tracks;
    HOST_OUTPUT(host_selected_events_mf_t, unsigned) host_selected_events_mf;
    DEVICE_INPUT(dev_offsets_all_velo_tracks_t, unsigned) dev_atomics_velo;
    DEVICE_INPUT(dev_offsets_velo_track_hit_number_t, unsigned) dev_velo_track_hit_number;
    DEVICE_INPUT(dev_offsets_ut_tracks_t, unsigned) dev_atomics_ut;
    DEVICE_INPUT(dev_offsets_ut_track_hit_number_t, unsigned) dev_ut_track_hit_number;
    DEVICE_INPUT(dev_ut_qop_t, float) dev_ut_qop;
    DEVICE_INPUT(dev_ut_track_velo_indices_t, unsigned) dev_ut_track_velo_indices;
    DEVICE_INPUT(dev_velo_kalman_beamline_states_t, char) dev_velo_kalman_beamline_states;
    DEVICE_INPUT(dev_match_upstream_muon_t, bool) dev_match_upstream_muon;
    DEVICE_INPUT(dev_event_list_mf_t, unsigned) dev_event_list_mf;
    DEVICE_INPUT(dev_mf_track_offsets_t, unsigned) dev_mf_track_offsets;
    DEVICE_OUTPUT(dev_mf_tracks_t, ParKalmanFilter::FittedTrack) dev_mf_tracks;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
  };

  __global__ void package_mf_tracks(Parameters, const unsigned number_of_events);

  struct package_mf_tracks_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{256, 1, 1}}};
  };

} // namespace package_mf_tracks
