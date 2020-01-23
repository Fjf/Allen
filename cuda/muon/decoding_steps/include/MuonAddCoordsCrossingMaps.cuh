#pragma once

#include "DeviceAlgorithm.cuh"
#include "MuonDefinitions.cuh"
#include "MuonRawToHits.cuh"
#include "MuonRaw.cuh"

namespace muon_add_coords_crossing_maps {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    DEVICE_INPUT(dev_storage_station_region_quarter_offsets_t, uint) dev_storage_station_region_quarter_offsets;
    DEVICE_OUTPUT(dev_storage_tile_id_t, uint) dev_storage_tile_id;
    DEVICE_OUTPUT(dev_storage_tdc_value_t, uint) dev_storage_tdc_value;
    DEVICE_OUTPUT(dev_atomics_muon_t, uint) dev_atomics_muon;
    DEVICE_OUTPUT(dev_muon_hits_t, Muon::HitsSoA) dev_muon_hits;
    DEVICE_INPUT(dev_muon_raw_to_hits_t, Muon::MuonRawToHits) dev_muon_raw_to_hits;
    DEVICE_OUTPUT(dev_muon_compact_hit_t, uint64_t) dev_muon_compact_hit;
    DEVICE_OUTPUT(dev_station_ocurrences_sizes_t, uint) dev_station_ocurrences_sizes;
    PROPERTY(block_dim_t, DeviceDimensions, "block_dim", "block dimensions", {256, 1, 1});
  };

  __global__ void muon_add_coords_crossing_maps(Parameters);

  template<typename T, char... S>
  struct muon_add_coords_crossing_maps_t : public DeviceAlgorithm, Parameters {
    constexpr static auto name = Name<S...>::s;
    decltype(global_function(muon_add_coords_crossing_maps)) function {muon_add_coords_crossing_maps};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const HostBuffers& host_buffers) const
    {
      set_size<dev_muon_hits_t>(arguments, value<host_number_of_selected_events_t>(arguments));
      set_size<dev_station_ocurrences_sizes_t>(
        arguments, value<host_number_of_selected_events_t>(arguments) * Muon::Constants::n_stations);
      set_size<dev_muon_compact_hit_t>(
        arguments, value<host_number_of_selected_events_t>(arguments) * Muon::Constants::max_numhits_per_event);
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      cudaStream_t& cuda_stream,
      cudaEvent_t& cuda_generic_event) const
    {
      cudaCheck(cudaMemsetAsync(
        begin<dev_station_ocurrences_sizes_t>(arguments),
        0,
        size<dev_station_ocurrences_sizes_t>(arguments),
        cuda_stream));

      cudaCheck(cudaMemsetAsync(
        begin<dev_muon_compact_hit_t>(arguments), 0, size<dev_muon_compact_hit_t>(arguments), cuda_stream));

      function(dim3(value<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(
        Parameters {begin<dev_storage_station_region_quarter_offsets_t>(arguments),
                    begin<dev_storage_tile_id_t>(arguments),
                    begin<dev_storage_tdc_value_t>(arguments),
                    begin<dev_atomics_muon_t>(arguments),
                    begin<dev_muon_hits_t>(arguments),
                    begin<dev_muon_raw_to_hits_t>(arguments),
                    begin<dev_muon_compact_hit_t>(arguments),
                    begin<dev_station_ocurrences_sizes_t>(arguments)});
    }

  private:
    Property<block_dim_t> m_block_dim {this};
  };
} // namespace muon_add_coords_crossing_maps