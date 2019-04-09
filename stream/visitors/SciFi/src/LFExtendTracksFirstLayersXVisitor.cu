#include "LFExtendTracksFirstLayersX.cuh"
#include "SequenceVisitor.cuh"

DEFINE_EMPTY_SET_ARGUMENTS_SIZE(lf_extend_tracks_first_layers_x_t)

template<>
void SequenceVisitor::visit<lf_extend_tracks_first_layers_x_t>(
  lf_extend_tracks_first_layers_x_t& state,
  const lf_extend_tracks_first_layers_x_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(128), cuda_stream);

  const auto forwarding_set_arguments = [&state, &constants, &arguments] (const uint8_t relative_extrapolation_layer) {
    state.set_arguments(
      arguments.offset<dev_scifi_hits>(),
      arguments.offset<dev_scifi_hit_count>(),
      arguments.offset<dev_atomics_ut>(),
      arguments.offset<dev_scifi_lf_tracks>(),
      arguments.offset<dev_scifi_lf_atomics>(),
      constants.dev_scifi_geometry,
      constants.dev_looking_forward_constants,
      constants.dev_inv_clus_res,
      arguments.offset<dev_scifi_lf_number_of_candidates>(),
      arguments.offset<dev_scifi_lf_candidates>(),
      relative_extrapolation_layer);
  };

  // * Forward to layer 1
  // * Forward to layer 0
  for (int i=0; i<2; ++i) {
    forwarding_set_arguments(1 - i);
    state.invoke();
  }
}