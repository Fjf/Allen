/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "AlgorithmTypes.cuh"

namespace scifi_pre_decode {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_raw_bank_version_t, int) host_raw_bank_version;
    HOST_INPUT(host_accumulated_number_of_scifi_hits_t, unsigned) host_accumulated_number_of_scifi_hits;
    DEVICE_INPUT(dev_scifi_raw_input_t, char) dev_scifi_raw_input;
    DEVICE_INPUT(dev_scifi_raw_input_offsets_t, unsigned) dev_scifi_raw_input_offsets;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_scifi_hit_offsets_t, unsigned) dev_scifi_hit_offsets;
    DEVICE_OUTPUT(dev_cluster_references_t, unsigned) dev_cluster_references;
  };

  struct scifi_pre_decode_t : public DeviceAlgorithm, Parameters {
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
  };
} // namespace scifi_pre_decode
