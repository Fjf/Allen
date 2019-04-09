#include "LFPromoteCandidates.cuh"
#include "SequenceVisitor.cuh"

template<>
void SequenceVisitor::set_arguments_size<lf_promote_candidates_t>(
  lf_promote_candidates_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_scifi_track_promoted_candidates>(
    host_buffers.host_number_of_selected_events[0] * SciFi::Constants::max_track_candidates);
  arguments.set_size<dev_scifi_tracks>(
    host_buffers.host_number_of_selected_events[0] * SciFi::Constants::max_tracks);
}

template<>
void SequenceVisitor::visit<lf_promote_candidates_t>(
  lf_promote_candidates_t& state,
  const lf_promote_candidates_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  cudaCheck(cudaMemsetAsync(
    arguments.offset<dev_scifi_track_promoted_candidates>(),
    0,
    arguments.size<dev_scifi_track_promoted_candidates>(),
    cuda_stream
  ));

  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_scifi_hits>(),
    arguments.offset<dev_scifi_hit_count>(),
    arguments.offset<dev_atomics_ut>(),
    arguments.offset<dev_scifi_track_candidates>(),
    arguments.offset<dev_extrapolation_layer_candidates>(),
    arguments.offset<dev_scifi_tracks>(),
    arguments.offset<dev_scifi_track_promoted_candidates>(),
    arguments.offset<dev_atomics_scifi>(),
    constants.dev_scifi_geometry,
    constants.dev_looking_forward_constants,
    constants.dev_inv_clus_res,
    arguments.offset<dev_ut_states>(),
    7);

  state.invoke();
}