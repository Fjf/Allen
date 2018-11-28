#include "SequenceVisitor.cuh"
#include "CalculatePhiAndSort.cuh"

template<>
void SequenceVisitor::set_arguments_size<velo_calculate_phi_and_sort_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_hit_permutation>(host_buffers.host_total_number_of_velo_clusters[0]);
}

template<>
void SequenceVisitor::visit<velo_calculate_phi_and_sort_t>(
  velo_calculate_phi_and_sort_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(runtime_options.number_of_events), dim3(64), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_estimated_input_size>(),
    arguments.offset<dev_module_cluster_num>(),
    arguments.offset<dev_velo_cluster_container>(),
    arguments.offset<dev_hit_permutation>()
  );

  state.invoke();
}
