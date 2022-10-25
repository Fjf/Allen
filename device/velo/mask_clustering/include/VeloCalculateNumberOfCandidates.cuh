/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"
#include "ClusteringDefinitions.cuh"

namespace velo_calculate_number_of_candidates {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_raw_bank_version_t, int) host_raw_bank_version;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_velo_raw_input_t, char) dev_velo_raw_input;
    DEVICE_INPUT(dev_velo_raw_input_offsets_t, unsigned) dev_velo_raw_input_offsets;
    DEVICE_INPUT(dev_velo_raw_input_sizes_t, unsigned) dev_velo_raw_input_sizes;
    DEVICE_INPUT(dev_velo_raw_input_types_t, unsigned) dev_velo_raw_input_types;
    DEVICE_OUTPUT(dev_number_of_candidates_t, unsigned) dev_number_of_candidates;
    DEVICE_OUTPUT(dev_velo_bank_index_t, unsigned) dev_velo_bank_index;
    PROPERTY(block_dim_x_t, "block_dim_x", "block dimension X", unsigned) block_dim_x_prop;
  };

  // Algorithm
  struct velo_calculate_number_of_candidates_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions&,
      const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Property<block_dim_x_t> m_block_dim_x {this, 256};
  };
} // namespace velo_calculate_number_of_candidates
