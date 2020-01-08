#pragma once

#include "VeloConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "LookingForwardConstants.cuh"
#include "LookingForwardTools.cuh"
#include "SciFiEventModel.cuh"
#include "DeviceAlgorithm.cuh"

namespace lf_extend_tracks_x {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    DEVICE_INPUT(dev_scifi_hits_t, uint) dev_scifi_hits;
    DEVICE_INPUT(dev_scifi_hit_count_t, uint) dev_scifi_hit_count;
    DEVICE_INPUT(dev_atomics_ut_t, uint) dev_atomics_ut;
    DEVICE_OUTPUT(dev_scifi_lf_tracks_t, SciFi::TrackHits) dev_scifi_lf_tracks;
    DEVICE_INPUT(dev_scifi_lf_atomics_t, uint) dev_scifi_lf_atomics;
    DEVICE_INPUT(dev_scifi_lf_initial_windows_t, int) dev_scifi_lf_initial_windows;
    DEVICE_INPUT(dev_scifi_lf_parametrization_t, float) dev_scifi_lf_parametrization;
  };

  __global__ void lf_extend_tracks_x(
    Parameters,
    const char* dev_scifi_geometry,
    const LookingForward::Constants* dev_looking_forward_constants,
    const float* dev_inv_clus_res);

  template<typename T>
  struct lf_extend_tracks_x_t : public DeviceAlgorithm, Parameters {
    constexpr static auto name {"lf_extend_tracks_x_t"};
    decltype(global_function(lf_extend_tracks_x)) function {lf_extend_tracks_x};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const HostBuffers& host_buffers) const
    {}

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      cudaStream_t& cuda_stream,
      cudaEvent_t& cuda_generic_event) const
    {
      function(dim3(value<host_number_of_selected_events_t>(arguments)), block_dimension(), cuda_stream)(
        Parameters {offset<dev_scifi_hits_t>(arguments),
                    offset<dev_scifi_hit_count_t>(arguments),
                    offset<dev_atomics_ut_t>(arguments),
                    offset<dev_scifi_lf_tracks_t>(arguments),
                    offset<dev_scifi_lf_atomics_t>(arguments),
                    offset<dev_scifi_lf_initial_windows_t>(arguments),
                    offset<dev_scifi_lf_parametrization_t>(arguments)},
        constants.dev_scifi_geometry,
        constants.dev_looking_forward_constants,
        constants.dev_inv_clus_res);
    }
  };
} // namespace lf_extend_tracks_x