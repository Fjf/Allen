#pragma once

#include "UTDefinitions.cuh"
#include "DeviceAlgorithm.cuh"

namespace ut_calculate_number_of_hits {
  DEFINE_PARAMETERS(
    Parameters,
    (HOST_INPUT(host_number_of_selected_events_t, uint), host_number_of_selected_events),
    (DEVICE_INPUT(dev_event_list_t, uint), dev_event_list),
    (DEVICE_INPUT(dev_ut_raw_input_t, char), dev_ut_raw_input),
    (DEVICE_INPUT(dev_ut_raw_input_offsets_t, uint), dev_ut_raw_input_offsets),
    (DEVICE_OUTPUT(dev_ut_hit_sizes_t, uint), dev_ut_hit_sizes),
    (PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions), block_dim))

  __global__ void ut_calculate_number_of_hits(
    Parameters,
    const char* ut_boards,
    const uint* dev_ut_region_offsets,
    const uint* dev_unique_x_sector_layer_offsets,
    const uint* dev_unique_x_sector_offsets);

  __global__ void ut_calculate_number_of_hits_mep(
    Parameters,
    const char* ut_boards,
    const uint* dev_ut_region_offsets,
    const uint* dev_unique_x_sector_layer_offsets,
    const uint* dev_unique_x_sector_offsets);

  struct ut_calculate_number_of_hits_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions&,
      const Constants& constants,
      const HostBuffers&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers&,
      cudaStream_t& cuda_stream,
      cudaEvent_t&) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{64, 4, 1}}};
  };
} // namespace ut_calculate_number_of_hits
