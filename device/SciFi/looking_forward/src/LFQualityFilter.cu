/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "LFQualityFilter.cuh"
#include <climits>

INSTANTIATE_ALGORITHM(lf_quality_filter::lf_quality_filter_t)

void lf_quality_filter::lf_quality_filter_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_atomics_scifi_t>(arguments, first<host_number_of_events_t>(arguments) * LookingForward::num_atomics);
  set_size<dev_scifi_tracks_t>(
    arguments,
    first<host_number_of_reconstructed_input_tracks_t>(arguments) * SciFi::Constants::max_SciFi_tracks_per_UT_track);
  set_size<dev_scifi_lf_y_parametrization_length_filter_t>(
    arguments,
    2 * first<host_number_of_reconstructed_input_tracks_t>(arguments) *
      property<maximum_number_of_candidates_per_ut_track_t>());
  set_size<dev_scifi_lf_parametrization_consolidate_t>(
    arguments,
    6 * first<host_number_of_reconstructed_input_tracks_t>(arguments) *
      SciFi::Constants::max_SciFi_tracks_per_UT_track);
  set_size<dev_lf_quality_of_tracks_t>(
    arguments,
    property<maximum_number_of_candidates_per_ut_track_t>() *
      first<host_number_of_reconstructed_input_tracks_t>(arguments));
}

void lf_quality_filter::lf_quality_filter_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_atomics_scifi_t>(arguments, 0, context);

  global_function(lf_quality_filter)(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(
    arguments, constants.dev_looking_forward_constants);

  if (property<verbosity_t>() >= logger::debug) {
    print<dev_atomics_scifi_t>(arguments);
  }
}

template<bool with_ut, typename T>
__device__ void quality_filter(
  lf_quality_filter::Parameters parameters,
  const LookingForward::Constants* dev_looking_forward_constants,
  const T* tracks)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const unsigned number_of_events = parameters.dev_number_of_events[0];

  const unsigned maximum_number_of_candidates_per_ut_track = parameters.maximum_number_of_candidates_per_ut_track;
  const float max_diff_ty_window = parameters.max_diff_ty_window;

  const auto input_tracks_view = tracks->container(event_number);

  const int event_number_of_tracks = input_tracks_view.size();
  const int event_tracks_offset = input_tracks_view.offset();
  // TODO: Don't do this. Will be replaced when SciFi EM is updated.
  const unsigned total_number_of_tracks =
    tracks->container(number_of_events - 1).offset() + tracks->container(number_of_events - 1).size();

  // SciFi hits
  const unsigned total_number_of_hits =
    parameters.dev_scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats];

  SciFi::ConstHitCount scifi_hit_count {parameters.dev_scifi_hit_count, event_number};
  SciFi::ConstHits scifi_hits {parameters.dev_scifi_hits, total_number_of_hits};

  const auto number_of_tracks = parameters.dev_scifi_lf_length_filtered_atomics[event_number];
  const auto event_offset = scifi_hit_count.event_offset();

  for (unsigned i = threadIdx.x; i < number_of_tracks; i += blockDim.x) {
    const auto scifi_track_index = event_tracks_offset * maximum_number_of_candidates_per_ut_track + i;
    const SciFi::TrackHits& track = parameters.dev_scifi_lf_length_filtered_tracks[scifi_track_index];
    const auto& input_state = parameters.dev_input_states[event_tracks_offset + track.input_track_index];

    bool hit_in_T1_UV = false;
    bool hit_in_T2_UV = false;
    bool hit_in_T3_UV = false;
    unsigned number_of_uv_hits = 0;
    for (unsigned j = 3; j < track.hitsNum; ++j) {
      const auto hit_index = event_offset + track.hits[j];
      const auto layer_number = scifi_hits.planeCode(hit_index) / 2;

      const bool current_hit_in_T1_UV = (layer_number == 1) || (layer_number == 2);
      const bool current_hit_in_T2_UV = (layer_number == 5) || (layer_number == 6);
      const bool current_hit_in_T3_UV = (layer_number == 9) || (layer_number == 10);

      hit_in_T1_UV |= current_hit_in_T1_UV;
      hit_in_T2_UV |= current_hit_in_T2_UV;
      hit_in_T3_UV |= current_hit_in_T3_UV;
      number_of_uv_hits += current_hit_in_T1_UV + current_hit_in_T2_UV + current_hit_in_T3_UV;
    }

    // Load parametrization
    const auto a1 = parameters.dev_scifi_lf_parametrization_length_filter[scifi_track_index];
    const auto b1 = parameters.dev_scifi_lf_parametrization_length_filter
                      [total_number_of_tracks * maximum_number_of_candidates_per_ut_track + scifi_track_index];
    const auto c1 = parameters.dev_scifi_lf_parametrization_length_filter
                      [2 * total_number_of_tracks * maximum_number_of_candidates_per_ut_track + scifi_track_index];
    const auto d_ratio = parameters.dev_scifi_lf_parametrization_length_filter
                           [3 * total_number_of_tracks * maximum_number_of_candidates_per_ut_track + scifi_track_index];

    // Do Y line fit
    const auto y_lms_fit = LookingForward::least_mean_square_y_fit(
      track, number_of_uv_hits, scifi_hits, a1, b1, c1, d_ratio, event_offset, dev_looking_forward_constants);

    // Save Y line fit
    parameters.dev_scifi_lf_y_parametrization_length_filter[scifi_track_index] = std::get<1>(y_lms_fit);
    parameters.dev_scifi_lf_y_parametrization_length_filter
      [total_number_of_tracks * maximum_number_of_candidates_per_ut_track + scifi_track_index] = std::get<2>(y_lms_fit);

    const float y_fit_contribution = std::get<0>(y_lms_fit) / LookingForward::range_y_fit_end;

    const auto in_ty_window = fabsf(std::get<2>(y_lms_fit) - input_state.ty) < max_diff_ty_window;
    const bool acceptable = hit_in_T1_UV && hit_in_T2_UV && hit_in_T3_UV &&
                            (track.hitsNum >= LookingForward::min_hits_or_ty_window || in_ty_window);

    // Combined value
    const auto combined_value = track.quality / (track.hitsNum - 3);

    float updated_track_quality = acceptable ? combined_value + y_fit_contribution : 10000.f;

    // Apply multipliers to quality of tracks depending on number of hits
    if (track.hitsNum == 9) {
      updated_track_quality *= LookingForward::track_9_hits_quality_multiplier;
    }
    else if (track.hitsNum == 10) {
      updated_track_quality *= LookingForward::track_10_hits_quality_multiplier;
    }
    else if (track.hitsNum == 11) {
      updated_track_quality *= LookingForward::track_11_hits_quality_multiplier;
    }
    else if (track.hitsNum == 12) {
      updated_track_quality *= LookingForward::track_12_hits_quality_multiplier;
    }

    parameters.dev_scifi_quality_of_tracks[scifi_track_index] = updated_track_quality;

    // // This code is to keep all the tracks
    // const auto insert_index = atomicAdd(parameters.dev_atomics_scifi + event_number, 1);
    // parameters.dev_scifi_tracks[ut_event_tracks_offset * SciFi::Constants::max_SciFi_tracks_per_UT_track +
    // insert_index] = track;
  }

  // Due to parameters.dev_scifi_quality_of_tracks RAW dependency
  __syncthreads();

  for (int i = threadIdx.x; i < event_number_of_tracks; i += blockDim.x) {
    float best_quality = LookingForward::quality_filter_max_quality;
    int best_track_index = -1;
    assert(number_of_tracks < INT_MAX); // assertion to make sure best_track_index is in range
    for (unsigned j = 0; j < number_of_tracks; j++) {
      const auto index = event_tracks_offset * maximum_number_of_candidates_per_ut_track + j;
      const auto track_quality = parameters.dev_scifi_quality_of_tracks[index];
      const SciFi::TrackHits& track = parameters.dev_scifi_lf_length_filtered_tracks[index];

      if (track.input_track_index == i && track_quality < best_quality) {
        best_quality = track_quality;
        best_track_index = j;
      }
    }

    if (best_track_index != -1) {

      const int insert_index = atomicAdd(parameters.dev_atomics_scifi + event_number, 1);
      assert(insert_index < event_number_of_tracks * SciFi::Constants::max_SciFi_tracks_per_UT_track);

      const auto scifi_track_index = event_tracks_offset * maximum_number_of_candidates_per_ut_track + best_track_index;
      const auto& track = parameters.dev_scifi_lf_length_filtered_tracks[scifi_track_index];

      const auto new_scifi_track_index =
        event_tracks_offset * SciFi::Constants::max_SciFi_tracks_per_UT_track + insert_index;
      parameters.dev_scifi_tracks[new_scifi_track_index] = track;

      // Save track parameters to last container as well
      const auto a1 = parameters.dev_scifi_lf_parametrization_length_filter[scifi_track_index];
      const auto b1 = parameters.dev_scifi_lf_parametrization_length_filter
                        [total_number_of_tracks * maximum_number_of_candidates_per_ut_track + scifi_track_index];
      const auto c1 = parameters.dev_scifi_lf_parametrization_length_filter
                        [2 * total_number_of_tracks * maximum_number_of_candidates_per_ut_track + scifi_track_index];
      const auto d_ratio =
        parameters.dev_scifi_lf_parametrization_length_filter
          [3 * total_number_of_tracks * maximum_number_of_candidates_per_ut_track + scifi_track_index];
      const auto y_b = parameters.dev_scifi_lf_y_parametrization_length_filter[scifi_track_index];
      const auto y_m = parameters.dev_scifi_lf_y_parametrization_length_filter
                         [total_number_of_tracks * maximum_number_of_candidates_per_ut_track + scifi_track_index];

      parameters.dev_scifi_lf_parametrization_consolidate[new_scifi_track_index] = a1;
      parameters.dev_scifi_lf_parametrization_consolidate
        [total_number_of_tracks * SciFi::Constants::max_SciFi_tracks_per_UT_track + new_scifi_track_index] = b1;
      parameters.dev_scifi_lf_parametrization_consolidate
        [2 * total_number_of_tracks * SciFi::Constants::max_SciFi_tracks_per_UT_track + new_scifi_track_index] = c1;
      parameters.dev_scifi_lf_parametrization_consolidate
        [3 * total_number_of_tracks * SciFi::Constants::max_SciFi_tracks_per_UT_track + new_scifi_track_index] =
        d_ratio;
      parameters.dev_scifi_lf_parametrization_consolidate
        [4 * total_number_of_tracks * SciFi::Constants::max_SciFi_tracks_per_UT_track + new_scifi_track_index] = y_b;
      parameters.dev_scifi_lf_parametrization_consolidate
        [5 * total_number_of_tracks * SciFi::Constants::max_SciFi_tracks_per_UT_track + new_scifi_track_index] = y_m;
    }
  }
}

__global__ void lf_quality_filter::lf_quality_filter(
  lf_quality_filter::Parameters parameters,
  const LookingForward::Constants* dev_looking_forward_constants)
{
  const auto* ut_tracks =
    Allen::dyn_cast<const Allen::Views::UT::Consolidated::MultiEventTracks*>(*parameters.dev_tracks_view);
  if (ut_tracks) {
    quality_filter<true>(parameters, dev_looking_forward_constants, ut_tracks);
  }
  else {
    const auto* velo_tracks =
      static_cast<const Allen::Views::Velo::Consolidated::MultiEventTracks*>(*parameters.dev_tracks_view);
    quality_filter<false>(parameters, dev_looking_forward_constants, velo_tracks);
  }
}
