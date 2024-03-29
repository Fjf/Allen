/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "BackendCommon.h"
#include "AlgorithmTypes.cuh"
#include "GenericContainerContracts.h"

#include <LumiDefinitions.cuh>

#include <VeloConsolidated.cuh>
#include "KinUtils.cuh"

namespace velo_lumi_counters {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_lumi_summaries_count_t, unsigned) host_lumi_summaries_count;
    DEVICE_INPUT(dev_lumi_event_indices_t, unsigned) dev_lumi_event_indices;
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_is_backward_t, bool) dev_is_backward;
    DEVICE_INPUT(dev_velo_states_view_t, Allen::Views::Physics::KalmanStates) dev_velo_states_view;
    DEVICE_INPUT(dev_offsets_all_velo_tracks_t, unsigned) dev_offsets_all_velo_tracks;
    DEVICE_OUTPUT(dev_lumi_infos_t, Lumi::LumiInfo) dev_lumi_infos;
    PROPERTY(
      tracks_eta_bins_t,
      "tracks_eta_bins",
      "tracks eta bins",
      std::array<float, Lumi::Constants::n_velo_eta_bin_edges>)
    tracks_eta_bins;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
    PROPERTY(
      lumi_counter_schema_t,
      "lumi_counter_schema",
      "schema for lumi counters",
      std::map<std::string, std::pair<unsigned, unsigned>>)
    lumi_counter_schema;
    PROPERTY(
      lumi_counter_shifts_and_scales_t,
      "lumi_counter_shifts_and_scales",
      "shifts and scales extracted from the schema for lumi counters",
      std::map<std::string, std::pair<float, float>>)
    lumi_counter_shifts_and_scales;
  }; // struct Parameters

  using offsets_and_sizes_t = std::array<unsigned, 2 * Lumi::Constants::n_velo_counters>;
  using shifts_and_scales_t = std::array<float, 2 * Lumi::Constants::n_velo_counters>;

  __global__ void velo_lumi_counters(
    Parameters,
    const unsigned number_of_events,
    const offsets_and_sizes_t offsets_and_sizes,
    const shifts_and_scales_t shifts_and_scales);

  // doca and eta copied from device/event_model/common/include/CopyTrackParameters.cuh
  // to avoid extra header files requirements
  // use (0, 0, 0) to replace pv position for simplicity
  __device__ float velo_DOCAz(const Allen::Views::Physics::KalmanState& velo_kalman_state)
  {
    float dx = velo_kalman_state.x();
    float dy = velo_kalman_state.y();
    float tx = velo_kalman_state.tx();
    float ty = velo_kalman_state.ty();
    return std::abs(ty * dx - tx * dy) / std::sqrt(tx * tx + ty * ty);
  }

  __device__ float velo_eta(const Allen::Views::Physics::KalmanState& velo_kalman_state, bool backward)
  {
    float tx = velo_kalman_state.tx();
    float ty = velo_kalman_state.ty();
    return eta_from_rho_z(std::sqrt(tx * tx + ty * ty), backward ? -1.f : 1.f);
  }

  struct velo_lumi_counters_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void init();

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const Allen::Context& context) const;

  private:
    Property<tracks_eta_bins_t> m_tracks_eta_bins {this, {-4.f, -3.f, -2.f, 2.f, 3.f, 4.f, 5.f}};
    Property<block_dim_t> m_block_dim {this, {{64, 1, 1}}};
    Property<lumi_counter_schema_t> m_lumi_counter_schema {this, {}};
    Property<lumi_counter_shifts_and_scales_t> m_lumi_counter_shifts_and_scales {this, {}};

    offsets_and_sizes_t m_offsets_and_sizes = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                                               0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    shifts_and_scales_t m_shifts_and_scales = {0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
                                               0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f};
  }; // struct velo_lumi_counters_t
} // namespace velo_lumi_counters
