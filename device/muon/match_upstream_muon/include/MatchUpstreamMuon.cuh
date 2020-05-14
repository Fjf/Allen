#pragma once

#include "DeviceAlgorithm.cuh"
#include "MuonDefinitions.cuh"
#include "FunctionsMatchUpstreamMuon.cuh"
#include "UTConsolidated.cuh"
#include "VeloConsolidated.cuh"

namespace MatchUpstreamMuon {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    HOST_INPUT(host_number_of_reconstructed_ut_tracks_t, uint);
    HOST_INPUT(host_selected_events_mf_t, uint);
    DEVICE_INPUT(dev_offsets_all_velo_tracks_t, uint) dev_atomics_velo;
    DEVICE_INPUT(dev_offsets_velo_track_hit_number_t, uint) dev_velo_track_hit_number;
    DEVICE_INPUT(dev_velo_kalman_beamline_states_t, char) dev_kalmanvelo_states;
    DEVICE_INPUT(dev_offsets_ut_tracks_t, uint) dev_atomics_ut;
    DEVICE_INPUT(dev_offsets_ut_track_hit_number_t, uint) dev_ut_track_hit_number;
    DEVICE_INPUT(dev_ut_qop_t, float) dev_ut_qop;
    DEVICE_INPUT(dev_ut_track_velo_indices_t, uint) dev_ut_track_velo_indices;
    DEVICE_INPUT(dev_station_ocurrences_offset_t, uint) dev_station_ocurrences_offset;
    DEVICE_INPUT(dev_muon_hits_t, char) dev_muon_hits;
    DEVICE_INPUT(dev_event_list_mf_t, uint) dev_event_list_mf;
    DEVICE_OUTPUT(dev_match_upstream_muon_t, bool) dev_muon_match;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions);
  };

  __global__ void match_upstream_muon(
    Parameters,
    const float* magnet_polarity,
    const MuonChambers* dev_muonmatch_search_muon_chambers,
    const SearchWindows* dev_muonmatch_search_windows,
    const uint number_of_events);

  template<typename T>
  struct match_upstream_muon_t : public DeviceAlgorithm, Parameters {

    decltype(global_function(match_upstream_muon)) function {match_upstream_muon};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const
    {
      set_size<dev_match_upstream_muon_t>(arguments, first<host_number_of_reconstructed_ut_tracks_t>(arguments));
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      cudaStream_t& cuda_stream,
      cudaEvent_t&) const
    {
      initialize<dev_match_upstream_muon_t>(arguments, 0, cuda_stream);

      function(dim3(first<host_selected_events_mf_t>(arguments)), property<block_dim_t>(), cuda_stream)(
        Parameters {data<dev_offsets_all_velo_tracks_t>(arguments),
            data<dev_offsets_velo_track_hit_number_t>(arguments),
            data<dev_velo_kalman_beamline_states_t>(arguments),
            data<dev_offsets_ut_tracks_t>(arguments),
            data<dev_offsets_ut_track_hit_number_t>(arguments),
            data<dev_ut_qop_t>(arguments),
            data<dev_ut_track_velo_indices_t>(arguments),
            data<dev_station_ocurrences_offset_t>(arguments),
            data<dev_muon_hits_t>(arguments),
            data<dev_event_list_mf_t>(arguments),
            data<dev_match_upstream_muon_t>(arguments)},
        constants.dev_magnet_polarity.data(),
        constants.dev_muonmatch_search_muon_chambers,
        constants.dev_muonmatch_search_windows,
        first<host_number_of_selected_events_t>(arguments));
      
      if (runtime_options.do_check) {
        cudaCheck(cudaMemcpyAsync(
          host_buffers.host_match_upstream_muon,
          data<dev_match_upstream_muon_t>(arguments),
          size<dev_match_upstream_muon_t>(arguments),
          cudaMemcpyDeviceToHost,
          cuda_stream));
      }
    }

  private:
    Property<block_dim_t> m_block_dim {this, {{128, 1, 1}}};
  };
}