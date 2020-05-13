#pragma once

#include "VertexFitter.cuh"
#include "VertexDefinitions.cuh"
#include "ParKalmanDefinitions.cuh"
#include "ParKalmanMath.cuh"
#include "DeviceAlgorithm.cuh"

namespace MFVertexFit {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    HOST_INPUT(host_number_of_mf_svs_t, uint);
    HOST_INPUT(host_selected_events_mf_t, uint);
    DEVICE_INPUT(dev_kf_tracks_t, ParKalmanFilter::FittedTrack) dev_kf_tracks;
    DEVICE_INPUT(dev_mf_tracks_t, ParKalmanFilter::FittedTrack) dev_mf_tracks;
    DEVICE_INPUT(dev_offsets_forward_tracks_t, uint) dev_offsets_forward_tracks;
    DEVICE_INPUT(dev_mf_track_offsets_t, uint) dev_mf_track_offsets;
    DEVICE_INPUT(dev_mf_sv_offsets_t, uint) dev_mf_sv_offsets;
    DEVICE_INPUT(dev_svs_kf_idx_t, uint) dev_svs_kf_idx;
    DEVICE_INPUT(dev_svs_mf_idx_t, uint) dev_svs_mf_idx;
    DEVICE_INPUT(dev_event_list_mf_t, uint) dev_event_list_mf;
    DEVICE_OUTPUT(dev_mf_svs_t, VertexFit::TrackMVAVertex) dev_mf_svs;
    PROPERTY(block_dim_t, DeviceDimensions, "block_dim", "block dimensions");
  };

  __global__ void fit_mf_vertices(Parameters);

  template<typename T>
  struct fit_mf_vertices_t : public DeviceAlgorithm, Parameters {

    decltype(global_function(fit_mf_vertices)) function {fit_mf_vertices};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const
    {
      set_size<dev_mf_svs_t>(arguments, first<host_number_of_mf_svs_t>(arguments));
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions&,
      const Constants&,
      HostBuffers& host_buffers,
      cudaStream_t& cuda_stream,
      cudaEvent_t&) const
    {
      initialize<dev_mf_svs_t>(arguments, 0, cuda_stream);

      function(dim3(first<host_selected_events_mf_t>(arguments)), property<block_dim_t>(), cuda_stream)(
        Parameters {data<dev_kf_tracks_t>(arguments),
                    data<dev_mf_tracks_t>(arguments),
                    data<dev_offsets_forward_tracks_t>(arguments),
                    data<dev_mf_track_offsets_t>(arguments),
                    data<dev_mf_sv_offsets_t>(arguments),
                    data<dev_svs_kf_idx_t>(arguments),
                    data<dev_svs_mf_idx_t>(arguments),
                    data<dev_event_list_mf_t>(arguments),
                    data<dev_mf_svs_t>(arguments)});

      safe_assign_to_host_buffer<dev_mf_svs_t>(
        host_buffers.host_mf_secondary_vertices, host_buffers.host_mf_secondary_vertices_size, arguments, cuda_stream);

      cudaCheck(cudaMemcpyAsync(
        host_buffers.host_mf_sv_offsets,
        data<dev_mf_sv_offsets_t>(arguments),
        size<dev_mf_sv_offsets_t>(arguments),
        cudaMemcpyDeviceToHost,
        cuda_stream));
    }

  private:
    Property<block_dim_t> m_block_dim {this, {{16, 16, 1}}};
  };

} // namespace MFVertexFit