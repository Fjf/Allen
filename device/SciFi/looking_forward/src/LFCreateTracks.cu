/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "LFCreateTracks.cuh"

INSTANTIATE_ALGORITHM(lf_create_tracks::lf_create_tracks_t)

void lf_create_tracks::lf_create_tracks_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_scifi_lf_tracks_t>(
    arguments,
    first<host_number_of_reconstructed_ut_tracks_t>(arguments) *
      LookingForward::maximum_number_of_candidates_per_ut_track);
  set_size<dev_scifi_lf_atomics_t>(arguments, first<host_number_of_events_t>(arguments));
  set_size<dev_scifi_lf_total_number_of_found_triplets_t>(
    arguments, first<host_number_of_reconstructed_ut_tracks_t>(arguments));
  set_size<dev_scifi_lf_parametrization_t>(
    arguments,
    4 * first<host_number_of_reconstructed_ut_tracks_t>(arguments) *
      LookingForward::maximum_number_of_candidates_per_ut_track);
}

void lf_create_tracks::lf_create_tracks_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  HostBuffers&,
  const Allen::Context& context) const
{
  initialize<dev_scifi_lf_total_number_of_found_triplets_t>(arguments, 0, context);
  initialize<dev_scifi_lf_atomics_t>(arguments, 0, context);

  global_function(lf_triplet_keep_best)(
    dim3(size<dev_event_list_t>(arguments)), property<triplet_keep_best_block_dim_t>(), context)(
    arguments, constants.dev_looking_forward_constants);

  global_function(lf_calculate_parametrization)(
    dim3(size<dev_event_list_t>(arguments)), property<calculate_parametrization_block_dim_t>(), context)(
    arguments, constants.dev_looking_forward_constants);

  global_function(lf_extend_tracks)(
    dim3(size<dev_event_list_t>(arguments)), property<extend_tracks_block_dim_t>(), context)(
    arguments, constants.dev_looking_forward_constants);
}
