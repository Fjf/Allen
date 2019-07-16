#include "LFQualityFilterX.cuh"

__global__ void lf_quality_filter_x(
  const int* dev_atomics_ut,
  const uint* dev_ut_track_hit_number,
  const float* dev_ut_qop,
  const uint* dev_ut_track_velo_indices,
  const char* dev_velo_states,
  const int* dev_atomics_velo,
  const uint* dev_velo_track_hit_number,
  const uint32_t* dev_scifi_hits,
  const uint32_t* dev_scifi_hit_count,
  const SciFi::TrackHits* dev_scifi_lf_tracks,
  const int* dev_scifi_lf_atomics,
  SciFi::TrackHits* dev_scifi_lf_x_filtered_tracks,
  int* dev_scifi_lf_x_filtered_atomics,
  float* dev_scifi_lf_xAtRef,
  const char* dev_scifi_geometry,
  const float* dev_inv_clus_res,
  const LookingForward::Constants* dev_looking_forward_constants,
  const SciFi::Tracking::Arrays* constArrays)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  // UT consolidated tracks
  const UT::Consolidated::Tracks ut_tracks {(uint*) dev_atomics_ut,
                                            (uint*) dev_ut_track_hit_number,
                                            (float*) dev_ut_qop,
                                            (uint*) dev_ut_track_velo_indices,
                                            event_number,
                                            number_of_events};
  const int ut_event_tracks_offset = ut_tracks.tracks_offset(event_number);
  const int ut_event_number_of_tracks = ut_tracks.number_of_tracks(event_number);

  // Velo states
  const Velo::Consolidated::Tracks velo_tracks {
    (uint*) dev_atomics_velo, (uint*) dev_velo_track_hit_number, event_number, number_of_events};
  const uint velo_tracks_offset_event = velo_tracks.tracks_offset(event_number);
  const Velo::Consolidated::States velo_states {(char*) dev_velo_states, velo_tracks.total_number_of_tracks};

  // SciFi hits
  const uint total_number_of_hits = dev_scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats];
  const SciFi::HitCount scifi_hit_count {(uint32_t*) dev_scifi_hit_count, event_number};
  const SciFi::SciFiGeometry scifi_geometry {dev_scifi_geometry};
  const SciFi::Hits scifi_hits {
    const_cast<uint32_t*>(dev_scifi_hits), total_number_of_hits, &scifi_geometry, dev_inv_clus_res};
  const auto event_offset = scifi_hit_count.event_offset();

  __shared__ float xAtRef_average_spread[LookingForward::maximum_number_of_candidates_per_ut_track];
  __shared__ float xAtRef_average_array[LookingForward::maximum_number_of_candidates_per_ut_track];

  for (uint16_t i = blockIdx.y; i < ut_event_number_of_tracks; i += gridDim.y) {
    const auto current_ut_track_index = ut_event_tracks_offset + i;
    const auto number_of_tracks = dev_scifi_lf_atomics[current_ut_track_index];

    __syncthreads();

    // first save indices and qualities of tracks
    for (int j = threadIdx.x; j < number_of_tracks; j += blockDim.x) {
      const SciFi::TrackHits& track =
        dev_scifi_lf_tracks[current_ut_track_index * LookingForward::maximum_number_of_candidates_per_ut_track + j];

      // calculate xAtRef average and the spread
      const auto velo_states_index = velo_tracks_offset_event + ut_tracks.velo_track[track.ut_track_index];
      const MiniState velo_state = velo_states.getMiniState(velo_states_index);
      const float xAtRef_initial = xFromVelo(SciFi::Tracking::zReference, velo_state);
      const float zMag_initial = zMagnet(velo_state, constArrays);
      float hits_x[6];
      float hits_z[6];
      float hits_x_atRef[6];
      for (int k = 0; k < track.hitsNum; ++k) {
        const int hit = event_offset + track.hits[k];
        const int plane_code = scifi_hits.planeCode(hit) >> 1;
        hits_x[k] = scifi_hits.x0[hit];
        hits_z[k] = dev_looking_forward_constants->Zone_zPos[plane_code];
      }
      const float xAtRef_average = LookingForward::get_average_and_individual_x_at_reference_plane(
        hits_x, hits_z, track.hitsNum, xAtRef_initial, constArrays, velo_state, zMag_initial, hits_x_atRef);

      float xAtRef_spread =
        LookingForward::get_average_x_at_reference_plane_spread(xAtRef_average, hits_x_atRef, track.hitsNum);

      // if (track.hitsNum == 3) // assign larg value to filter out later
      //   xAtRef_spread = LookingForward::filter_x_max_xAtRef_spread;

      xAtRef_average_spread[j] = xAtRef_spread;
      xAtRef_average_array[j] = xAtRef_average;
    }

    __syncthreads();

    // Sort track candidates by quality
    for (int16_t j = threadIdx.x; j < number_of_tracks; j += blockDim.x) {
      float xAtRef_spread = xAtRef_average_spread[j];
      int16_t insert_position = 0;
      for (int16_t k = 0; k < number_of_tracks; ++k) {
        const float other_xAtRef_spread = xAtRef_average_spread[k];
        if (xAtRef_spread > other_xAtRef_spread || (xAtRef_spread == other_xAtRef_spread && j < k)) {
          ++insert_position;
        }
      }

      // Note: xAtRef_spread < 10.f instead of
      //       xAtRef_spread < 1e9
      //       kills about 2.3% fakes with something like 0.5% impact on RE
      if (
        insert_position < LookingForward::maximum_number_of_candidates_per_ut_track_after_x_filter &&
        xAtRef_spread < LookingForward::filter_x_max_xAtRef_spread) {
        // Save best track candidates
        const auto insert_index = atomicAdd(dev_scifi_lf_x_filtered_atomics + event_number, 1);
        const SciFi::TrackHits& track =
          dev_scifi_lf_tracks[current_ut_track_index * LookingForward::maximum_number_of_candidates_per_ut_track + j];
        dev_scifi_lf_x_filtered_tracks
          [ut_event_tracks_offset * LookingForward::maximum_number_of_candidates_per_ut_track_after_x_filter +
           insert_index] = track;
        dev_scifi_lf_xAtRef
          [ut_event_tracks_offset * LookingForward::maximum_number_of_candidates_per_ut_track_after_x_filter +
           insert_index] = xAtRef_average_array[j];
      }
    }
  }
}
