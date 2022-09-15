/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"
#include "MuonDefinitions.cuh"
#include "MuonRawToHits.cuh"
#include "MuonRaw.cuh"

namespace muon_populate_tile_and_tdc {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_muon_total_number_of_tiles_t, unsigned) host_muon_total_number_of_tiles;
    HOST_INPUT(host_raw_bank_version_t, int) host_raw_bank_version;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_muon_raw_t, char) dev_muon_raw;
    DEVICE_INPUT(dev_muon_raw_offsets_t, unsigned) dev_muon_raw_offsets;
    DEVICE_INPUT(dev_muon_raw_sizes_t, unsigned) dev_muon_raw_sizes;
    DEVICE_INPUT(dev_muon_raw_types_t, unsigned) dev_muon_raw_types;
    DEVICE_INPUT(dev_muon_raw_to_hits_t, Muon::MuonRawToHits) dev_muon_raw_to_hits;
    DEVICE_INPUT(dev_storage_station_region_quarter_offsets_t, unsigned) dev_storage_station_region_quarter_offsets;
    DEVICE_OUTPUT(dev_storage_tile_id_t, unsigned) dev_storage_tile_id;
    DEVICE_OUTPUT(dev_storage_tdc_value_t, unsigned) dev_storage_tdc_value;
    DEVICE_OUTPUT(dev_atomics_muon_t, unsigned) dev_atomics_muon;
    DEVICE_OUTPUT(dev_muon_tile_used_t, bool) dev_muon_tile_used;
    DEVICE_OUTPUT(dev_station_ocurrences_sizes_t, unsigned) dev_station_ocurrences_sizes;
  };

  struct muon_populate_tile_and_tdc_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants&,
      HostBuffers&,
      const Allen::Context& context) const;
  };
} // namespace muon_populate_tile_and_tdc
