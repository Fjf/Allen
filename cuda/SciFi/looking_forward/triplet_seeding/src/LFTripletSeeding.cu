#include "LFTripletSeeding.cuh"
#include "LFTripletSeedingImpl.cuh"
#include "LookingForwardTools.cuh"

__global__ void lf_triplet_seeding(
  uint32_t* dev_scifi_hits,
  const uint32_t* dev_scifi_hit_count,
  const uint* dev_atomics_velo,
  const char* dev_velo_states,
  const uint* dev_atomics_ut,
  const uint* dev_ut_track_hit_number,
  const uint* dev_ut_track_velo_indices,
  const float* dev_ut_qop,
  const char* dev_scifi_geometry,
  const float* dev_inv_clus_res,
  const int* dev_initial_windows,
  const LookingForward::Constants* dev_looking_forward_constants,
  const MiniState* dev_ut_states,
  SciFi::CombinedValue* dev_scifi_lf_triplet_best,
  SciFi::TrackHits* dev_scifi_tracks,
  uint* dev_atomics_scifi)
{
  __shared__ float shared_precalc_expected_x1[32];

  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  // Velo consolidated types
  const Velo::Consolidated::States velo_states {(char*) dev_velo_states, dev_atomics_velo[2 * number_of_events]};
  const uint velo_tracks_offset_event = dev_atomics_velo[number_of_events + event_number];

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
  const SciFi::Hits scifi_hits {dev_scifi_hits, total_number_of_hits, &scifi_geometry, dev_inv_clus_res};
  const auto event_offset = scifi_hit_count.event_offset();

  constexpr int number_of_seeds = 2;
  uint triplet_seeding_layers[number_of_seeds][3] {
    {0, 2, 4},
    {1, 3, 5}
  };

  for (uint i = blockIdx.y; i < ut_event_number_of_tracks; i += gridDim.y) {
    const auto current_ut_track_index = ut_event_tracks_offset + i;
    const auto velo_track_index = ut_tracks.velo_track[i];
    // const auto velo_track_index = dev_ut_track_velo_indices[current_ut_track_index];
    const auto qop = dev_ut_qop[current_ut_track_index];
    const int* initial_windows = dev_initial_windows + current_ut_track_index;
    
    const uint velo_states_index = velo_tracks_offset_event + velo_track_index;
    const MiniState velo_state = velo_states.getMiniState(velo_states_index);
    const auto x_at_z_magnet = velo_state.x + (LookingForward::z_magnet - velo_state.z) * velo_state.tx;

    for (uint triplet_seed = 0; triplet_seed < number_of_seeds; ++triplet_seed) {
      const auto layer_0 = triplet_seeding_layers[triplet_seed][0];
      const auto layer_2 = triplet_seeding_layers[triplet_seed][2];

      // int l0_start = initial_windows[(layer_0 * 8) * ut_total_number_of_tracks];
      // int l0_extrapolated = initial_windows[(layer_0 * 8 + 4) * ut_total_number_of_tracks];
      // int l0_size = initial_windows[(layer_0 * 8 + 1) * ut_total_number_of_tracks];

      // int l1_start = initial_windows[(layer_1 * 8) * ut_total_number_of_tracks];
      // int l1_extrapolated = initial_windows[(layer_1 * 8 + 4) * ut_total_number_of_tracks];
      // int l1_size = initial_windows[(layer_1 * 8 + 1) * ut_total_number_of_tracks];

      // int l2_start = initial_windows[(layer_2 * 8) * ut_total_number_of_tracks];
      // int l2_extrapolated = initial_windows[(layer_2 * 8 + 4) * ut_total_number_of_tracks];
      // int l2_size = initial_windows[(layer_2 * 8 + 1) * ut_total_number_of_tracks];

      lf_triplet_seeding_impl(
        scifi_hits.x0 + event_offset,
        layer_0,
        layer_2,
        initial_windows,
        ut_total_number_of_tracks,
        qop,
        (dev_ut_states + current_ut_track_index)->tx,
        velo_state.tx,
        x_at_z_magnet,
        shared_precalc_expected_x1,
        dev_scifi_lf_triplet_best + (current_ut_track_index * LookingForward::n_triplet_seeds + triplet_seed) *
                              LookingForward::maximum_number_of_triplets_per_seed,
        dev_looking_forward_constants);
    }
  }
}
