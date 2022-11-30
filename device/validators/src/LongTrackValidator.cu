/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "LongTrackValidator.cuh"
#include "CopyTrackParameters.cuh"

INSTANTIATE_ALGORITHM(long_track_validator::long_track_validator_t)

__global__ void long_track_validator::long_track_validator(long_track_validator::Parameters parameters)
{
  const unsigned event_number = blockIdx.x;
  const auto event_long_tracks = parameters.dev_multi_event_long_tracks_view->container(event_number);
  const auto endvelo_states = parameters.dev_velo_states_view[event_number];
  const unsigned offset_long_tracks = event_long_tracks.offset();
  Checker::Track* long_checker_tracks_event = parameters.dev_long_checker_tracks + offset_long_tracks;

  prepare_long_tracks(event_long_tracks, endvelo_states, long_checker_tracks_event);
}

void long_track_validator::long_track_validator_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_long_checker_tracks_t>(arguments, first<host_number_of_reconstructed_long_tracks_t>(arguments));
}

void long_track_validator::long_track_validator_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{
  global_function(long_track_validator)(first<host_number_of_events_t>(arguments), 256, context)(arguments);

  const auto event_list = make_host_buffer<dev_event_list_t>(arguments, context);
  const auto long_tracks_for_checker = make_host_buffer<dev_long_checker_tracks_t>(arguments, context);
  const auto event_tracks_offsets = make_host_buffer<dev_offsets_long_tracks_t>(arguments, context);
  std::vector<Checker::Tracks> tracks;
  tracks.resize(event_list.size());
  for (size_t i = 0; i < event_list.size(); ++i) {
    const auto evnum = event_list[i];
    const auto event_offset = event_tracks_offsets[evnum];
    const auto n_tracks = event_tracks_offsets[evnum + 1] - event_offset;
    std::vector<Checker::Track> event_trakcs = {long_tracks_for_checker.begin() + event_offset,
                                                long_tracks_for_checker.begin() + event_offset + n_tracks};
    tracks[i] = event_trakcs;
  }

  auto& checker =
    runtime_options.checker_invoker->checker<TrackCheckerLong>(name(), property<root_output_filename_t>());
  checker.accumulate(*first<host_mc_events_t>(arguments), tracks, event_list);
}