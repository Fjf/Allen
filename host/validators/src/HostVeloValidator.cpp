/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "HostVeloValidator.h"
#include "PrepareTracks.h"

INSTANTIATE_ALGORITHM(host_velo_validator::host_velo_validator_t)

void host_velo_validator::host_velo_validator_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{
  const auto offsets_all_velo_tracks = make_host_buffer<dev_offsets_all_velo_tracks_t>(arguments, context);
  const auto offsets_velo_track_hit_number = make_host_buffer<dev_offsets_velo_track_hit_number_t>(arguments, context);
  const auto velo_track_hits = make_host_buffer<dev_velo_track_hits_t>(arguments, context);
  const auto event_list = make_host_buffer<dev_event_list_t>(arguments, context);

  auto tracks = prepareVeloTracks(
    first<host_number_of_events_t>(arguments),
    offsets_all_velo_tracks,
    offsets_velo_track_hit_number,
    velo_track_hits,
    event_list);

  auto& checker =
    runtime_options.checker_invoker->checker<TrackCheckerVelo>(name(), property<root_output_filename_t>());
  checker.accumulate(*first<host_mc_events_t>(arguments), tracks, event_list);
}
