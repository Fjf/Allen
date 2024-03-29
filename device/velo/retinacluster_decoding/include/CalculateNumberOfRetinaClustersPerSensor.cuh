/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"
#include "ClusteringDefinitions.cuh"

namespace calculate_number_of_retinaclusters_each_sensor_pair {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, uint) host_number_of_events;
    HOST_INPUT(host_raw_bank_version_t, int) host_raw_bank_version;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_velo_retina_raw_input_t, char) dev_velo_retina_raw_input;
    DEVICE_INPUT(dev_velo_retina_raw_input_offsets_t, uint) dev_velo_retina_raw_input_offsets;
    DEVICE_INPUT(dev_velo_retina_raw_input_sizes_t, uint) dev_velo_retina_raw_input_sizes;
    DEVICE_INPUT(dev_velo_retina_raw_input_types_t, uint) dev_velo_retina_raw_input_types;
    DEVICE_OUTPUT(dev_retina_bank_index_t, uint) dev_retina_bank_index;
    DEVICE_OUTPUT(dev_each_sensor_pair_size_t, uint) dev_each_sensor_pair_size;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim_prop;
  };

  struct calculate_number_of_retinaclusters_each_sensor_pair_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{256, 1, 1}}};
  };
} // namespace calculate_number_of_retinaclusters_each_sensor_pair
