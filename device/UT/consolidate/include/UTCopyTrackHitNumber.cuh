/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "VeloEventModel.cuh"
#include "UTDefinitions.cuh"
#include "UTEventModel.cuh"
#include "UTConsolidated.cuh"
#include "AlgorithmTypes.cuh"

namespace ut_copy_track_hit_number {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_ut_tracks_t, unsigned) host_number_of_reconstructed_ut_tracks;
    DEVICE_INPUT(dev_ut_tracks_t, UT::TrackHits) dev_ut_tracks;
    DEVICE_INPUT(dev_offsets_ut_tracks_t, unsigned) dev_atomics_ut;
    DEVICE_OUTPUT(dev_ut_track_hit_number_t, unsigned) dev_ut_track_hit_number;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
  };

  __global__ void ut_copy_track_hit_number(Parameters);

  struct ut_copy_track_hit_number_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{512, 1, 1}}};
  };
} // namespace ut_copy_track_hit_number