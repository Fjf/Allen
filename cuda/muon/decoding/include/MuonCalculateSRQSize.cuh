#pragma once

#include "DeviceAlgorithm.cuh"
#include "MuonDefinitions.cuh"
#include "MuonRawToHits.cuh"
#include "MuonRaw.cuh"

namespace muon_calculate_srq_size {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    DEVICE_INPUT(dev_event_list_t, uint) dev_event_list;
    DEVICE_INPUT(dev_muon_raw_t, char) dev_muon_raw;
    DEVICE_INPUT(dev_muon_raw_offsets_t, uint) dev_muon_raw_offsets;
    DEVICE_OUTPUT(dev_muon_raw_to_hits_t, Muon::MuonRawToHits) dev_muon_raw_to_hits;
    DEVICE_OUTPUT(dev_storage_station_region_quarter_sizes_t, uint) dev_storage_station_region_quarter_sizes;
  };

  __global__ void muon_calculate_srq_size(Parameters);

  __global__ void muon_calculate_srq_size_mep(Parameters);

  template<typename T, char... S>
  struct muon_calculate_srq_size_t : public DeviceAlgorithm, Parameters {
    constexpr static auto name = Name<S...>::s;

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const
    {
      set_size<dev_muon_raw_to_hits_t>(arguments, 1);
      set_size<dev_storage_station_region_quarter_sizes_t>(
        arguments,
        value<host_number_of_selected_events_t>(arguments) * Muon::Constants::n_layouts * Muon::Constants::n_stations * Muon::Constants::n_regions *
            Muon::Constants::n_quarters);
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers&,
      cudaStream_t& cuda_stream,
      cudaEvent_t&) const
    {
      // FIXME: this should be done as part of the consumers, but
      // currently it cannot. This is because it is not possible to
      // indicate dependencies between Consumer and/or Producers.
      Muon::MuonRawToHits muonRawToHits {constants.dev_muon_tables, constants.dev_muon_geometry};

      cudaCheck(cudaMemcpyAsync(
        begin<dev_muon_raw_to_hits_t>(arguments),
        &muonRawToHits,
        sizeof(muonRawToHits),
        cudaMemcpyHostToDevice,
        cuda_stream));

      initialize<dev_storage_station_region_quarter_sizes_t>(arguments, 0, cuda_stream);

      const auto parameters = Parameters {
        begin<dev_event_list_t>(arguments),
        begin<dev_muon_raw_t>(arguments),
        begin<dev_muon_raw_offsets_t>(arguments),
        begin<dev_muon_raw_to_hits_t>(arguments),
        begin<dev_storage_station_region_quarter_sizes_t>(arguments)};

      using function_t = decltype(global_function(muon_calculate_srq_size));
      function_t function =
        runtime_options.mep_layout ? function_t {muon_calculate_srq_size_mep} : function_t {muon_calculate_srq_size};
      function(
        value<host_number_of_selected_events_t>(arguments),
        Muon::MuonRawEvent::number_of_raw_banks * Muon::MuonRawEvent::batches_per_bank,
        cuda_stream)(parameters);
    }
  };
} // namespace muon_calculate_srq_size
