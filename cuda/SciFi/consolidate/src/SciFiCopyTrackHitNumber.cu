#include "SciFiCopyTrackHitNumber.cuh"

/**
 * @brief Copies UT track hit numbers on a consecutive container
 */
__global__ void scifi_copy_track_hit_number::scifi_copy_track_hit_number(
  scifi_copy_track_hit_number::Parameters parameters)
{
  const auto event_number = blockIdx.x;
  const auto ut_event_tracks_offset = parameters.dev_atomics_ut[event_number];

  const auto event_tracks =
    parameters.dev_scifi_tracks + ut_event_tracks_offset * SciFi::Constants::max_SciFi_tracks_per_UT_track;
  const auto accumulated_tracks = parameters.dev_atomics_scifi[event_number];
  const auto number_of_tracks =
    parameters.dev_atomics_scifi[event_number + 1] - parameters.dev_atomics_scifi[event_number];

  // Pointer to scifi_track_hit_number of current event.
  uint* scifi_track_hit_number = parameters.dev_scifi_track_hit_number + accumulated_tracks;

  // Loop over tracks.
  for (uint element = threadIdx.x; element < number_of_tracks; ++element) {
    scifi_track_hit_number[element] = event_tracks[element].hitsNum;
  }
}
