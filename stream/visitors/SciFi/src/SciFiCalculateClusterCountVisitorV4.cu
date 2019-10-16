#include "SequenceVisitor.cuh"
#include "SciFiCalculateClusterCountV4.cuh"

template<>
void SequenceVisitor::set_arguments_size<scifi_calculate_cluster_count_v4_t>(
  scifi_calculate_cluster_count_v4_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_scifi_hit_count>(
    host_buffers.host_number_of_selected_events[0] * SciFi::Constants::n_mat_groups_and_mats + 1);
}

template<>
void SequenceVisitor::visit<scifi_calculate_cluster_count_v4_t>(
  scifi_calculate_cluster_count_v4_t& state,
  const scifi_calculate_cluster_count_v4_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  cudaCheck(
    cudaMemsetAsync(arguments.offset<dev_scifi_hit_count>(), 0, arguments.size<dev_scifi_hit_count>(), cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  state.set_opts(runtime_options.mep_layout, dim3(host_buffers.host_number_of_selected_events[0]), dim3(240), cuda_stream);
  state.set_arguments(
    runtime_options.mep_layout,
    arguments.offset<dev_scifi_raw_input>(),
    arguments.offset<dev_scifi_raw_input_offsets>(),
    arguments.offset<dev_scifi_hit_count>(),
    arguments.offset<dev_event_list>(),
    constants.dev_scifi_geometry);

  state.invoke(runtime_options.mep_layout);
}
