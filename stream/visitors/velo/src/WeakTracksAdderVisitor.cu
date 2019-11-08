#include "SequenceVisitor.cuh"
#include "WeakTracksAdder.cuh"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(velo_weak_tracks_adder_t)

template<>
void SequenceVisitor::visit<velo_weak_tracks_adder_t>(
  velo_weak_tracks_adder_t& state,
  const velo_weak_tracks_adder_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  // Setup opts and arguments
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_velo_cluster_container>(),
    arguments.offset<dev_estimated_input_size>(),
    arguments.offset<dev_tracks>(),
    arguments.offset<dev_weak_tracks>(),
    arguments.offset<dev_hit_used>(),
    arguments.offset<dev_atomics_velo>());

  state.invoke();
}
