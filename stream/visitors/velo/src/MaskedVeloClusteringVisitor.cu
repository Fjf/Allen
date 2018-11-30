#include "SequenceVisitor.cuh"
#include "MaskedVeloClustering.cuh"

template<>
void SequenceVisitor::set_arguments_size<velo_masked_clustering_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_velo_cluster_container>(6 * host_buffers.host_total_number_of_velo_clusters[0]);
}

template<>
void SequenceVisitor::visit<velo_masked_clustering_t>(
  velo_masked_clustering_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(runtime_options.number_of_events), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_raw_input>(),
    arguments.offset<dev_raw_input_offsets>(),
    arguments.offset<dev_estimated_input_size>(),
    arguments.offset<dev_module_cluster_num>(),
    arguments.offset<dev_module_candidate_num>(),
    arguments.offset<dev_cluster_candidates>(),
    arguments.offset<dev_velo_cluster_container>(),
    constants.dev_velo_geometry,
    constants.dev_velo_sp_patterns,
    constants.dev_velo_sp_fx,
    constants.dev_velo_sp_fy
  );
  
  state.invoke();
}
