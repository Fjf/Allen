#include "StreamVisitor.cuh"

template<>
void StreamVisitor::visit<decltype(weak_tracks_adder_t(weak_tracks_adder))>(
  decltype(weak_tracks_adder_t(weak_tracks_adder))& state,
  const int sequence_step,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  ArgumentManager<argument_tuple_t>& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  scheduler.setup_next(arguments, sequence_step);

  state.set_opts(dim3(runtime_options.number_of_events, VeloUTTracking::n_layers), dim3(64), cuda_stream);
  state.set_arguments(
    arguments.offset<arg::dev_ut_raw_input>(),
    arguments.offset<arg::dev_ut_raw_input_offsets>(),
    constants.dev_ut_boards,
    constants.dev_ut_geometry,
    constants.dev_ut_region_offsets,
    constants.dev_unique_x_sector_layer_offsets,
    constants.dev_unique_x_sector_offsets,
    arguments.offset<arg::dev_ut_hit_offsets>(),
    arguments.offset<arg::dev_ut_hits>(),
    arguments.offset<arg::dev_ut_hit_count>(),
    arguments.offset<arg::dev_ut_hit_permutations>()
  );

  state.invoke();
}



