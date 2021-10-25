/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "LFQualityFilterLength.cuh"

void lf_quality_filter_length::lf_quality_filter_length_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_scifi_lf_length_filtered_tracks_t>(
    arguments,
    first<host_number_of_reconstructed_ut_tracks_t>(arguments) *
      LookingForward::maximum_number_of_candidates_per_ut_track);
  set_size<dev_scifi_lf_length_filtered_atomics_t>(
    arguments, first<host_number_of_events_t>(arguments) * LookingForward::num_atomics);
  set_size<dev_scifi_lf_parametrization_length_filter_t>(
    arguments,
    4 * first<host_number_of_reconstructed_ut_tracks_t>(arguments) *
      LookingForward::maximum_number_of_candidates_per_ut_track);
}

void lf_quality_filter_length::lf_quality_filter_length_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  HostBuffers&,
  const Allen::Context& context) const
{
  initialize<dev_scifi_lf_length_filtered_atomics_t>(arguments, 0, context);

  global_function(lf_quality_filter_length)(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(
    arguments);
}

__global__ void lf_quality_filter_length::lf_quality_filter_length(lf_quality_filter_length::Parameters parameters)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const unsigned number_of_events = parameters.dev_number_of_events[0];

  // UT consolidated tracks
  const auto ut_tracks_view = parameters.dev_ut_tracks_view[event_number];
  const auto ut_event_tracks_offset = ut_tracks_view.offset();
  // TODO: Don't do this. Will be replaced when SciFi EM is updated.
  const auto ut_total_number_of_tracks = parameters.dev_ut_tracks_view[number_of_events - 1].offset() +
                                         parameters.dev_ut_tracks_view[number_of_events - 1].size();

  const auto number_of_tracks = parameters.dev_scifi_lf_atomics[event_number];

  for (unsigned i = threadIdx.x; i < number_of_tracks; i += blockDim.x) {
    const auto scifi_track_index =
      ut_event_tracks_offset * LookingForward::maximum_number_of_candidates_per_ut_track + i;
    const SciFi::TrackHits& track = parameters.dev_scifi_lf_tracks[scifi_track_index];

    if (track.hitsNum >= LookingForward::track_min_hits) {
      const auto insert_index = atomicAdd(parameters.dev_scifi_lf_length_filtered_atomics + event_number, 1);
      const auto new_scifi_track_index =
        ut_event_tracks_offset * LookingForward::maximum_number_of_candidates_per_ut_track + insert_index;

      parameters.dev_scifi_lf_length_filtered_tracks[new_scifi_track_index] = track;

      // Save track parameters to new container as well
      const auto a1 = parameters.dev_scifi_lf_parametrization[scifi_track_index];
      const auto b1 =
        parameters.dev_scifi_lf_parametrization
          [ut_total_number_of_tracks * LookingForward::maximum_number_of_candidates_per_ut_track + scifi_track_index];
      const auto c1 = parameters.dev_scifi_lf_parametrization
                        [2 * ut_total_number_of_tracks * LookingForward::maximum_number_of_candidates_per_ut_track +
                         scifi_track_index];
      const auto d_ratio =
        parameters.dev_scifi_lf_parametrization
          [3 * ut_total_number_of_tracks * LookingForward::maximum_number_of_candidates_per_ut_track +
           scifi_track_index];

      parameters.dev_scifi_lf_parametrization_length_filter[new_scifi_track_index] = a1;
      parameters.dev_scifi_lf_parametrization_length_filter
        [ut_total_number_of_tracks * LookingForward::maximum_number_of_candidates_per_ut_track +
         new_scifi_track_index] = b1;
      parameters.dev_scifi_lf_parametrization_length_filter
        [2 * ut_total_number_of_tracks * LookingForward::maximum_number_of_candidates_per_ut_track +
         new_scifi_track_index] = c1;
      parameters.dev_scifi_lf_parametrization_length_filter
        [3 * ut_total_number_of_tracks * LookingForward::maximum_number_of_candidates_per_ut_track +
         new_scifi_track_index] = d_ratio;
    }
  }
}
