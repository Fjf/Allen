#include "VertexFitter.cuh"
#include "SequenceVisitor.cuh"

template<>
void SequenceVisitor::set_arguments_size<fit_secondary_vertices_t>(
  const fit_secondary_vertices_t& state,
  fit_secondary_vertices_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_secondary_vertices>(VertexFit::max_svs * host_buffers.host_number_of_selected_events[0]);
  arguments.set_size<dev_sv_atomics>(2 * host_buffers.host_number_of_selected_events[0] + 1);
}

template<>
void SequenceVisitor::visit<fit_secondary_vertices_t>(
  fit_secondary_vertices_t& state,
  const fit_secondary_vertices_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{

  cudaCheck(cudaMemsetAsync(
    arguments.offset<dev_sv_atomics>(),
    0,
    arguments.size<dev_sv_atomics>(),
    cuda_stream));
  
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_kf_tracks>(),
    arguments.offset<dev_atomics_scifi>(),
    arguments.offset<dev_scifi_track_hit_number>(),
    arguments.offset<dev_scifi_qop>(),
    arguments.offset<dev_scifi_states>(),
    arguments.offset<dev_scifi_track_ut_indices>(),
    arguments.offset<dev_multi_fit_vertices>(),
    arguments.offset<dev_number_of_multi_fit_vertices>(),
    arguments.offset<dev_kalman_pv_ipchi2>(),
    arguments.offset<dev_sv_atomics>(),
    arguments.offset<dev_secondary_vertices>());
  state.invoke();

  if (runtime_options.do_check) {
    cudaCheck(cudaMemcpyAsync(
      host_buffers.host_sv_atomics,
      arguments.offset<dev_sv_atomics>(),
      arguments.size<dev_sv_atomics>(),
      cudaMemcpyDeviceToHost,
      cuda_stream));
    cudaCheck(cudaMemcpyAsync(
      host_buffers.host_sv_atomics,
      arguments.offset<dev_sv_atomics>(),
      arguments.size<dev_sv_atomics>(),
      cudaMemcpyDeviceToHost,
      cuda_stream));
  }
}
