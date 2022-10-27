/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "VertexDefinitions.cuh"
#include "VertexFitDeviceFunctions.cuh"
#include "ParKalmanDefinitions.cuh"
#include "ParKalmanMath.cuh"
#include "AlgorithmTypes.cuh"

namespace MFVertexFit {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_mf_svs_t, unsigned) host_number_of_mf_svs;
    HOST_INPUT(host_selected_events_mf_t, unsigned) host_selected_events_mf;
    DEVICE_INPUT(dev_kf_particles_t, Allen::Views::Physics::BasicParticles) dev_kf_particles;
    DEVICE_INPUT(dev_mf_particles_t, Allen::Views::Physics::BasicParticles) dev_mf_particles;
    DEVICE_INPUT(dev_mf_sv_offsets_t, unsigned) dev_mf_sv_offsets;
    DEVICE_INPUT(dev_svs_kf_idx_t, unsigned) dev_svs_kf_idx;
    DEVICE_INPUT(dev_svs_mf_idx_t, unsigned) dev_svs_mf_idx;
    DEVICE_INPUT(dev_event_list_mf_t, unsigned) dev_event_list_mf;
    DEVICE_OUTPUT(dev_mf_svs_t, VertexFit::TrackMVAVertex) dev_mf_svs;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
  };

  __global__ void fit_mf_vertices(Parameters);

  struct fit_mf_vertices_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions&,
      const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{16, 16, 1}}};
  };

} // namespace MFVertexFit