/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "DeviceAlgorithm.cuh"

namespace scifi_raw_bank_decoder_v4 {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_accumulated_number_of_scifi_hits_t, unsigned) host_accumulated_number_of_scifi_hits;
    DEVICE_INPUT(dev_scifi_raw_input_t, char) dev_scifi_raw_input;
    DEVICE_INPUT(dev_scifi_raw_input_offsets_t, unsigned) dev_scifi_raw_input_offsets;
    DEVICE_INPUT(dev_scifi_hit_offsets_t, unsigned) dev_scifi_hit_offsets;
    DEVICE_INPUT(dev_cluster_references_t, unsigned) dev_cluster_references;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    DEVICE_OUTPUT(dev_scifi_hits_t, char) dev_scifi_hits;
    PROPERTY(
      raw_bank_decoder_block_dim_t,
      "raw_bank_decoder_block_dim",
      "block dimensions of raw bank decoder kernel",
      DeviceDimensions)
    raw_bank_decoder_block_dim;
    PROPERTY(
      direct_decoder_block_dim_t,
      "direct_decoder_block_dim",
      "block dimensions of direct decoder",
      DeviceDimensions)
    direct_decoder_block_dim;
  };

  __global__ void scifi_raw_bank_decoder_v4(Parameters, const char* scifi_geometry);

  __global__ void scifi_direct_decoder_v4(Parameters, const char* scifi_geometry);

  __global__ void scifi_raw_bank_decoder_v4_mep(Parameters, const char* scifi_geometry);

  __global__ void scifi_direct_decoder_v4_mep(Parameters, const char* scifi_geometry);

  struct scifi_raw_bank_decoder_v4_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers&,
      const Allen::Context& context) const;

  private:
    Property<raw_bank_decoder_block_dim_t> m_raw_bank_decoder_block_dim {this, {{256, 1, 1}}};
    Property<direct_decoder_block_dim_t> m_direct_decoder_block_dim {this, {{2, 16, 1}}};
  };
} // namespace scifi_raw_bank_decoder_v4
