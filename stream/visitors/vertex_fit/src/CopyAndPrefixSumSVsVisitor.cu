#include "SequenceVisitor.cuh"
#include "PrefixSum.cuh"
#include "CpuPrefixSum.cuh"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(copy_and_prefix_sum_single_block_sv_t)

template<>
void SequenceVisitor::visit<copy_and_prefix_sum_single_block_sv_t>(
  copy_and_prefix_sum_single_block_sv_t& state,
  const copy_and_prefix_sum_single_block_sv_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  if (runtime_options.cpu_offload) {
    cudaCheck(cudaMemcpyAsync(
      (uint*) arguments.offset<dev_sv_atomics>() + host_buffers.host_number_of_selected_events[0],
      (uint*) arguments.offset<dev_sv_atomics>(),
      host_buffers.host_number_of_selected_events[0] * sizeof(uint),
      cudaMemcpyDeviceToDevice,
      cuda_stream));

    cpu_prefix_sum(
      host_buffers.host_prefix_sum_buffer,
      host_buffers.host_allocated_prefix_sum_space,
      (uint*) arguments.offset<dev_sv_atomics>() + host_buffers.host_number_of_selected_events[0],
      (host_buffers.host_number_of_selected_events[0] + 1) * sizeof(uint),
      cuda_stream,
      cuda_generic_event,
      host_buffers.host_number_of_svs);
  }
  else {
    state.set_opts(cuda_stream);
    state.set_arguments(
      (uint*) arguments.offset<dev_sv_atomics>() + host_buffers.host_number_of_selected_events[0] * 2,
      (uint*) arguments.offset<dev_sv_atomics>(),
      (uint*) arguments.offset<dev_sv_atomics>() + host_buffers.host_number_of_selected_events[0],
      host_buffers.host_number_of_selected_events[0]);
    state.invoke();

    // Fetch number of secondary vertices
    cudaCheck(cudaMemcpyAsync(
      host_buffers.host_number_of_svs,
      arguments.offset<dev_sv_atomics>() + host_buffers.host_number_of_selected_events[0],
      sizeof(uint),
      cudaMemcpyDeviceToHost,
      cuda_stream));

    cudaEventRecord(cuda_generic_event, cuda_stream);
    cudaEventSynchronize(cuda_generic_event);
  }

  if (runtime_options.do_check) {
    cudaCheck(cudaMemcpyAsync(
      host_buffers.host_number_of_svs,
      arguments.offset<dev_sv_atomics>() + host_buffers.host_number_of_selected_events[0] * 2,
      sizeof(uint),
      cudaMemcpyDeviceToHost,
      cuda_stream));

    cudaCheck(cudaMemcpyAsync(
      host_buffers.host_sv_atomics,
      arguments.offset<dev_sv_atomics>(),
      (2 * host_buffers.host_number_of_selected_events[0] + 1) * sizeof(uint),
      cudaMemcpyDeviceToHost,
      cuda_stream));
  }
}
