#include "LFExtendTracksX.cuh"

__global__ void lf_extend_tracks_x(
  const uint32_t* dev_scifi_hits,
  const uint32_t* dev_scifi_hit_count,
  const int* dev_atomics_ut,
  SciFi::TrackHits* dev_scifi_tracks,
  const int* dev_atomics_scifi,
  const char* dev_scifi_geometry,
  const LookingForward::Constants* dev_looking_forward_constants,
  const float* dev_inv_clus_res,
  const uint* dev_scifi_lf_number_of_candidates,
  const short* dev_scifi_lf_candidates,
  bool* dev_scifi_lf_candidates_flag,
  const uint8_t relative_extrapolation_layer)
{
  const auto number_of_events = gridDim.x;
  const auto event_number = blockIdx.x;

  // UT consolidated tracks
  const int ut_event_tracks_offset = dev_atomics_ut[number_of_events + event_number];

  // SciFi hits
  const uint total_number_of_hits = dev_scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats];
  const SciFi::HitCount scifi_hit_count {(uint32_t*) dev_scifi_hit_count, event_number};
  const SciFi::SciFiGeometry scifi_geometry {dev_scifi_geometry};
  const SciFi::Hits scifi_hits {
    const_cast<uint32_t*>(dev_scifi_hits), total_number_of_hits, &scifi_geometry, dev_inv_clus_res};
  const auto event_offset = scifi_hit_count.event_offset();

  // SciFi un-consolidated track types
  const int number_of_tracks = dev_atomics_scifi[event_number];

  for (int i = threadIdx.x; i < number_of_tracks; i += blockDim.x) {
    SciFi::TrackHits& track = dev_scifi_tracks[event_number * SciFi::Constants::max_tracks + i];
    const auto current_ut_track_index = ut_event_tracks_offset + track.ut_track_index;

    // Candidates pointer for current UT track
    const auto scifi_lf_candidates = dev_scifi_lf_candidates + current_ut_track_index *
                                                                 LookingForward::number_of_x_layers *
                                                                 LookingForward::maximum_number_of_candidates;

    const int8_t number_of_candidates =
      dev_scifi_lf_number_of_candidates
        [current_ut_track_index * LookingForward::number_of_x_layers + relative_extrapolation_layer + 1] -
      dev_scifi_lf_number_of_candidates
        [current_ut_track_index * LookingForward::number_of_x_layers + relative_extrapolation_layer];

    const auto h0 = event_offset + track.hits[track.hitsNum - 2];
    const auto h1 = event_offset + track.hits[track.hitsNum - 1];

    const auto layer0 = scifi_hits.planeCode(h0) >> 1;
    const auto layer1 = scifi_hits.planeCode(h1) >> 1;

    const auto x0 = scifi_hits.x0[h0];
    const auto x1 = scifi_hits.x0[h1];

    const auto z0 = dev_looking_forward_constants->Zone_zPos[layer0];
    const auto z1 = dev_looking_forward_constants->Zone_zPos[layer1];
    const auto z2 = dev_looking_forward_constants->Zone_zPos_xlayers[relative_extrapolation_layer];

    // For the flags
    const auto h_prev = event_offset + track.hits[track.hitsNum - 3];
    const auto layer_prev = scifi_hits.planeCode(h_prev) >> 1;
    const auto l_prev_offset = dev_scifi_lf_number_of_candidates
      [current_ut_track_index * LookingForward::number_of_x_layers +
       dev_looking_forward_constants->convert_layer[layer_prev]];

    const auto l0_offset = dev_scifi_lf_number_of_candidates
      [current_ut_track_index * LookingForward::number_of_x_layers +
       dev_looking_forward_constants->convert_layer[layer0]];

    const auto l1_offset = dev_scifi_lf_number_of_candidates
      [current_ut_track_index * LookingForward::number_of_x_layers +
       dev_looking_forward_constants->convert_layer[layer1]];

    const auto extrapolation_layer_offset = dev_scifi_lf_number_of_candidates
      [current_ut_track_index * LookingForward::number_of_x_layers + relative_extrapolation_layer];

    lf_extend_tracks_x_impl(
      scifi_hits,
      scifi_lf_candidates + relative_extrapolation_layer * LookingForward::maximum_number_of_candidates,
      number_of_candidates,
      track,
      x0,
      x1,
      z0,
      z1,
      z2,
      dev_looking_forward_constants->chi2_mean_extrapolation_to_x_layers[relative_extrapolation_layer - 3] +
        2.5f * dev_looking_forward_constants->chi2_stddev_extrapolation_to_x_layers[relative_extrapolation_layer - 3],
      event_offset,
      dev_scifi_lf_candidates_flag,
      relative_extrapolation_layer,
      l_prev_offset,
      l0_offset,
      l1_offset,
      extrapolation_layer_offset,
      current_ut_track_index);
  }
}
