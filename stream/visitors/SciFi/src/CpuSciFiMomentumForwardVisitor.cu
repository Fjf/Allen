#include "SequenceVisitor.cuh"
#include "RunForwardCPU.h"
#include "RunMomentumForwardCPU.h"
#include "Tools.h"

template<> 
void SequenceVisitor::set_arguments_size<cpu_scifi_momentum_forward_t>(
  cpu_scifi_momentum_forward_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_scifi_tracks>(host_buffers.host_number_of_selected_events[0] * SciFi::Constants::max_tracks);
  arguments.set_size<dev_atomics_scifi>(host_buffers.host_number_of_selected_events[0] * SciFi::num_atomics);
}

template<>
void SequenceVisitor::visit<cpu_scifi_momentum_forward_t>(
  cpu_scifi_momentum_forward_t& state,
  const cpu_scifi_momentum_forward_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Synchronize previous CUDA transmissions
  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  // Run Forward on x86 architecture
  // ATTENTION: when using SciFi raw bank version 5, 
  // need: 2*host_buffers.host_number_of_selected_events[0]*...
  host_buffers.host_velo_states.resize(arguments.size<dev_velo_states>());
  host_buffers.host_scifi_hits.resize(arguments.size<dev_scifi_hits>());
  host_buffers.host_scifi_hit_count.resize(arguments.size<dev_scifi_hit_count>());

  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_scifi_hits.data(),
    arguments.offset<dev_scifi_hits>(),
    arguments.size<dev_scifi_hits>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaCheck(cudaMemcpyAsync(
    host_buffers.host_scifi_hit_count.data(),
    arguments.offset<dev_scifi_hit_count>(),
    arguments.size<dev_scifi_hit_count>(),
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaCheck(cudaMemcpy(
    host_buffers.host_velo_states.data(),
    arguments.offset<dev_velo_states>(),
    arguments.size<dev_velo_states>(),
    cudaMemcpyDeviceToHost));

  /* INFO: use this call statement when running the momentum forward x86 algorithm.
     For the looking forward x86 algorithm, leave this commented because the function is invoked in 
     stream/checkers/include/UTSequenceCheckers_impl.cuh to have the true SciFi IDs linked to a matched UT track available */
     
  // int rv = state.invoke(
  //   host_buffers.scifi_tracks_events.data(),
  //   host_buffers.host_atomics_scifi,
  //   host_buffers.host_scifi_hits.data(),
  //   host_buffers.host_scifi_hit_count.data(),
  //   constants.host_scifi_geometry,
  //   constants.host_inv_clus_res, 
  //   host_buffers.host_atomics_velo,
  //   host_buffers.host_velo_track_hit_number,
  //   host_buffers.host_velo_states.data(),
  //   host_buffers.host_atomics_ut,
  //   host_buffers.host_ut_track_hit_number,
  //   host_buffers.host_ut_qop,
  //   host_buffers.host_ut_x,
  //   host_buffers.host_ut_tx,
  //   host_buffers.host_ut_z,
  //   host_buffers.host_ut_track_velo_indices,
  //   host_buffers.host_number_of_selected_events[0]);

  // // copy SciFi tracks to device for consolidation
  // cudaCheck(cudaMemcpyAsync(
  //   arguments.offset<dev_atomics_scifi>(), 
  //   host_buffers.host_atomics_scifi,
  //   arguments.size<dev_atomics_scifi>(),
  //   cudaMemcpyHostToDevice, 
  //   cuda_stream));

  // cudaCheck(cudaMemcpyAsync(
  //   arguments.offset<dev_scifi_tracks>(),
  //   host_buffers.scifi_tracks_events.data(),
  //   arguments.size<dev_scifi_tracks>(),
  //   cudaMemcpyHostToDevice,
  //   cuda_stream));
  
  // cudaEventRecord(cuda_generic_event, cuda_stream);
  // cudaEventSynchronize(cuda_generic_event);
}