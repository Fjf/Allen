/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "SciFiCopyTrackHitNumber.cuh"

INSTANTIATE_ALGORITHM(scifi_copy_track_hit_number::scifi_copy_track_hit_number_t)

void scifi_copy_track_hit_number::scifi_copy_track_hit_number_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_scifi_track_hit_number_t>(arguments, first<host_number_of_reconstructed_scifi_tracks_t>(arguments));
}

void scifi_copy_track_hit_number::scifi_copy_track_hit_number_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  global_function(scifi_copy_track_hit_number)(
    dim3(first<host_number_of_events_t>(arguments)), property<block_dim_t>(), context)(arguments);
}

/**
 * @brief Copies UT track hit numbers on a consecutive container
 */
__global__ void scifi_copy_track_hit_number::scifi_copy_track_hit_number(
  scifi_copy_track_hit_number::Parameters parameters)
{
  const auto event_number = blockIdx.x;
  const auto ut_event_tracks_offset = parameters.dev_atomics_input[event_number];

  const auto event_tracks =
    parameters.dev_scifi_tracks + ut_event_tracks_offset * SciFi::Constants::max_SciFi_tracks_per_UT_track;
  const auto accumulated_tracks = parameters.dev_atomics_scifi[event_number];
  const auto number_of_tracks =
    parameters.dev_atomics_scifi[event_number + 1] - parameters.dev_atomics_scifi[event_number];

  // Pointer to scifi_track_hit_number of current event.
  unsigned* scifi_track_hit_number = parameters.dev_scifi_track_hit_number + accumulated_tracks;

  // Loop over tracks.
  for (unsigned element = threadIdx.x; element < number_of_tracks; ++element) {
    scifi_track_hit_number[element] = event_tracks[element].hitsNum;
  }
}
