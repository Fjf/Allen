/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "UTCopyTrackHitNumber.cuh"

INSTANTIATE_ALGORITHM(ut_copy_track_hit_number::ut_copy_track_hit_number_t)

void ut_copy_track_hit_number::ut_copy_track_hit_number_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_ut_track_hit_number_t>(arguments, first<host_number_of_reconstructed_ut_tracks_t>(arguments));
}

void ut_copy_track_hit_number::ut_copy_track_hit_number_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  HostBuffers&,
  const Allen::Context& context) const
{
  global_function(ut_copy_track_hit_number)(
    dim3(first<host_number_of_events_t>(arguments)), property<block_dim_t>(), context)(arguments);
}

/**
 * @brief Copies UT track hit numbers on a consecutive container
 */
__global__ void ut_copy_track_hit_number::ut_copy_track_hit_number(ut_copy_track_hit_number::Parameters parameters)
{
  const auto event_number = blockIdx.x;
  const auto event_tracks = parameters.dev_ut_tracks + event_number * UT::Constants::max_num_tracks;
  const auto accumulated_tracks = parameters.dev_atomics_ut[event_number];
  const auto number_of_tracks = parameters.dev_atomics_ut[event_number + 1] - parameters.dev_atomics_ut[event_number];

  // Pointer to ut_track_hit_number of current event.
  unsigned* ut_track_hit_number = parameters.dev_ut_track_hit_number + accumulated_tracks;

  // Loop over tracks.
  for (unsigned element = threadIdx.x; element < number_of_tracks; ++element) {
    ut_track_hit_number[element] = event_tracks[element].hits_num;
  }
}
