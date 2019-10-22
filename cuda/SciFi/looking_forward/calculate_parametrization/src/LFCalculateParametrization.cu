#include "LFCalculateParametrization.cuh"

__global__ void lf_calculate_parametrization(
  const uint32_t* dev_scifi_hits,
  const uint32_t* dev_scifi_hit_count,
  const uint* dev_atomics_velo,
  const uint* dev_velo_track_hit_number,
  const char* dev_velo_states,
  const uint* dev_atomics_ut,
  const uint* dev_ut_track_hit_number,
  const uint* dev_ut_track_velo_indices,
  const float* dev_ut_qop,
  const SciFi::TrackHits* dev_scifi_tracks,
  const uint* dev_atomics_scifi,
  const char* dev_scifi_geometry,
  const LookingForward::Constants* dev_looking_forward_constants,
  const float* dev_inv_clus_res,
  float* dev_scifi_lf_parametrization)
{
  // if (Configuration::verbosity_level >= logger::debug) {
  //   if (blockIdx.y == 0) {
  //     printf("---- Extend Missing X ----\n");
  //   }
  // }

  const auto number_of_events = gridDim.x;
  const auto event_number = blockIdx.x;

  // Velo consolidated types
  const Velo::Consolidated::Tracks velo_tracks {
    (uint*) dev_atomics_velo, (uint*) dev_velo_track_hit_number, event_number, number_of_events};
  const Velo::Consolidated::States velo_states {(char*) dev_velo_states, velo_tracks.total_number_of_tracks};
  const uint velo_tracks_offset_event = velo_tracks.tracks_offset(event_number);

  // UT consolidated tracks
  const auto ut_event_tracks_offset = dev_atomics_ut[number_of_events + event_number];
  const auto ut_event_number_of_tracks = dev_atomics_ut[number_of_events + event_number + 1] - ut_event_tracks_offset;
  const auto ut_total_number_of_tracks = dev_atomics_ut[2 * number_of_events];

  // UT consolidated tracks
  UT::Consolidated::Tracks ut_tracks {(uint*) dev_atomics_ut,
                                      (uint*) dev_ut_track_hit_number,
                                      (float*) dev_ut_qop,
                                      (uint*) dev_ut_track_velo_indices,
                                      event_number,
                                      number_of_events};
  // SciFi hits
  const uint total_number_of_hits = dev_scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats];
  const SciFi::HitCount scifi_hit_count {(uint32_t*) dev_scifi_hit_count, event_number};
  const SciFi::SciFiGeometry scifi_geometry {dev_scifi_geometry};
  const SciFi::Hits scifi_hits {
    const_cast<uint32_t*>(dev_scifi_hits), total_number_of_hits, &scifi_geometry, dev_inv_clus_res};
  const auto event_offset = scifi_hit_count.event_offset();

  for (uint i_ut_track = threadIdx.x; i_ut_track < ut_event_number_of_tracks; i_ut_track += blockDim.x) {
    const auto current_ut_track_index = ut_event_tracks_offset + i_ut_track;
    const auto velo_track_index = ut_tracks.velo_track[i_ut_track];
    const auto number_of_tracks = dev_atomics_scifi[current_ut_track_index];

    const uint velo_states_index = velo_tracks_offset_event + velo_track_index;
    const MiniState velo_state = velo_states.getMiniState(velo_states_index);

    for (uint i = threadIdx.y; i < number_of_tracks; i += blockDim.y) {
      const auto scifi_track_index = current_ut_track_index * LookingForward::maximum_number_of_candidates_per_ut_track + i;
      const SciFi::TrackHits& track = dev_scifi_tracks[scifi_track_index];

      // Note: The notation 1, 2, 3 is used here (instead of h0, h1, h2)
      //       to avoid mistakes, as the code is similar to that of Hybrid Seeding
      const auto h1 = event_offset + track.hits[0];
      const auto h2 = event_offset + track.hits[1];
      const auto h3 = event_offset + track.hits[2];
      const auto x1 = scifi_hits.x0[h1];
      const auto x2 = scifi_hits.x0[h2];
      const auto x3 = scifi_hits.x0[h3];
      const auto z1_noref = dev_looking_forward_constants->Zone_zPos_xlayers[track.get_layer(0)];
      const auto z2_noref = dev_looking_forward_constants->Zone_zPos_xlayers[track.get_layer(1)];
      const auto z3_noref = dev_looking_forward_constants->Zone_zPos_xlayers[track.get_layer(2)];

      // From hybrid seeding
      constexpr float z_mid_t = 8520.f * Gaudi::Units::mm;

      // the dRatio for solving the parabola has to be
      constexpr float d_ratio_par_0 = 0.000267957f;
      constexpr float d_ratio_par_1 = -8.651e-06f;
      constexpr float d_ratio_par_2 = 4.60324e-05f;

      // constexpr float d_ratio = -0.0000262f;
      // Updated d_ratio
      const auto track_y_ref = velo_state.y + velo_state.ty * (z2_noref - velo_state.z);
      const auto radius_position = sqrtf((5.f * 5.f * 1.e-8f * x2 * x2 + 1e-6f * track_y_ref * track_y_ref));
      const auto d_ratio =
        -1.f * (d_ratio_par_0 + d_ratio_par_1 * radius_position + d_ratio_par_2 * radius_position * radius_position);

      const auto z1 = z1_noref - z_mid_t;
      const auto z2 = z2_noref - z_mid_t;
      const auto z3 = z3_noref - z_mid_t;
      const auto corrZ1 = 1.f + d_ratio * z1;
      const auto corrZ2 = 1.f + d_ratio * z2;
      const auto corrZ3 = 1.f + d_ratio * z3;

      const auto det = z1 * z1 * corrZ1 * z2 + z1 * z3 * z3 * corrZ3 + z2 * z2 * corrZ2 * z3 - z2 * z3 * z3 * corrZ3 -
                       z1 * z2 * z2 * corrZ2 - z3 * z1 * z1 * corrZ1;
      const auto det1 = x1 * z2 + z1 * x3 + x2 * z3 - z2 * x3 - z1 * x2 - z3 * x1;
      const auto det2 = z1 * z1 * corrZ1 * x2 + x1 * z3 * z3 * corrZ3 + z2 * z2 * corrZ2 * x3 - x2 * z3 * z3 * corrZ3 -
                        x1 * z2 * z2 * corrZ2 - x3 * z1 * z1 * corrZ1;
      const auto det3 = z1 * z1 * corrZ1 * z2 * x3 + z1 * z3 * z3 * corrZ3 * x2 + z2 * z2 * corrZ2 * z3 * x1 -
                        z2 * z3 * z3 * corrZ3 * x1 - z1 * z2 * z2 * corrZ2 * x3 - z3 * z1 * z1 * corrZ1 * x2;

      const auto recdet = 1.f / det;
      const auto a1 = recdet * det1;
      const auto b1 = recdet * det2;
      const auto c1 = recdet * det3;

      dev_scifi_lf_parametrization[scifi_track_index] = a1;
      dev_scifi_lf_parametrization[ut_total_number_of_tracks * LookingForward::maximum_number_of_candidates_per_ut_track + scifi_track_index] = b1;
      dev_scifi_lf_parametrization[2 * ut_total_number_of_tracks * LookingForward::maximum_number_of_candidates_per_ut_track + scifi_track_index] = c1;
      dev_scifi_lf_parametrization[3 * ut_total_number_of_tracks * LookingForward::maximum_number_of_candidates_per_ut_track + scifi_track_index] = d_ratio;
    }
  }
}
