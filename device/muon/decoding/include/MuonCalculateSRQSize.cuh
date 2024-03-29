/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"
#include "MuonDefinitions.cuh"
#include "MuonRawToHits.cuh"
#include "MuonRaw.cuh"
#include <gsl/gsl>

namespace muon_calculate_srq_size {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_raw_bank_version_t, int) host_raw_bank_version;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_muon_raw_t, char) dev_muon_raw;
    DEVICE_INPUT(dev_muon_raw_offsets_t, unsigned) dev_muon_raw_offsets;
    DEVICE_INPUT(dev_muon_raw_sizes_t, unsigned) dev_muon_raw_sizes;
    DEVICE_INPUT(dev_muon_raw_types_t, unsigned) dev_muon_raw_types;
    DEVICE_OUTPUT(dev_muon_raw_to_hits_t, Muon::MuonRawToHits) dev_muon_raw_to_hits;
    DEVICE_OUTPUT(dev_storage_station_region_quarter_sizes_t, unsigned) dev_storage_station_region_quarter_sizes;
  };

  struct muon_calculate_srq_size_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const Allen::Context& context) const;
  };
} // namespace muon_calculate_srq_size
