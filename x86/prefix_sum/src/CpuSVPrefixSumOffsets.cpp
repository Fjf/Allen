#include "CpuSVPrefixSumOffsets.h"

void cpu_sv_prefix_sum_offsets_t::operator()(
  const ArgumentRefManager<Arguments>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event) const
{
  cudaCheck(cudaMemcpyAsync(
    (uint*) arguments.offset<dev_sv_offsets>(),
    (uint*) arguments.offset<dev_atomics_scifi>(),
    host_buffers.host_number_of_selected_events[0] * sizeof(uint),
    cudaMemcpyDeviceToDevice,
    cuda_stream));

  function.invoke(
    host_buffers.host_prefix_sum_buffer,
    host_buffers.host_allocated_prefix_sum_space,
    (uint*) arguments.offset<dev_sv_offsets>(),
    (host_buffers.host_number_of_selected_events[0] + 1) * sizeof(uint),
    cuda_stream,
    cuda_generic_event,
    host_buffers.host_number_of_svs);
}
