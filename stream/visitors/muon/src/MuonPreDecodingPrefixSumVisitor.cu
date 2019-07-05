#include "PrefixSumHandler.cuh"
#include "SequenceVisitor.cuh"
#include "CpuPrefixSum.cuh"

template<>
void SequenceVisitor::set_arguments_size<muon_pre_decoding_prefix_sum_t>(
  muon_pre_decoding_prefix_sum_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_prefix_sum_auxiliary_array_8>(muon_pre_decoding_prefix_sum_t::aux_array_size(
    arguments.size<dev_storage_station_region_quarter_offsets>() /
    sizeof(dev_storage_station_region_quarter_offsets::type)));
}

template<>
void SequenceVisitor::visit<muon_pre_decoding_prefix_sum_t>(
  muon_pre_decoding_prefix_sum_t& state,
  const muon_pre_decoding_prefix_sum_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  if (runtime_options.cpu_offload) {
    cpu_prefix_sum(
      host_buffers.host_prefix_sum_buffer,
      host_buffers.host_allocated_prefix_sum_space,
      arguments.offset<dev_storage_station_region_quarter_offsets>(),
      arguments.size<dev_storage_station_region_quarter_offsets>(),
      cuda_stream,
      cuda_generic_event,
      host_buffers.host_muon_total_number_of_tiles);

    // arguments.print<dev_storage_station_region_quarter_offsets>();
  }
  else {
    // Set size of the main array to be prefix summed
    state.set_size((arguments.size<dev_storage_station_region_quarter_offsets>() >> 2) - 1);

    // Set the cuda_stream
    state.set_opts(cuda_stream);

    // Set arguments: Array to prefix sum and auxiliary array
    state.set_arguments(
      arguments.offset<dev_storage_station_region_quarter_offsets>(),
      arguments.offset<dev_prefix_sum_auxiliary_array_8>());

    // Invoke all steps of prefix sum
    state.invoke();

    // Fetch total number of hits accumulated with all windows
    cudaCheck(cudaMemcpyAsync(
      host_buffers.host_muon_total_number_of_tiles,
      arguments.offset<dev_storage_station_region_quarter_offsets>() +
        (arguments.size<dev_storage_station_region_quarter_offsets>() >> 2) - 1,
      sizeof(int),
      cudaMemcpyDeviceToHost,
      cuda_stream));

    cudaEventRecord(cuda_generic_event, cuda_stream);
    cudaEventSynchronize(cuda_generic_event);
  }
}
