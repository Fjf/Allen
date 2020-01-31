#include "FilterTracks.cuh"
#include "VertexDefinitions.cuh"
#include "ParKalmanMath.cuh"
#include "ParKalmanDefinitions.cuh"

__global__ void FilterTracks::filter_tracks(FilterTracks::Parameters parameters)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint idx_offset = event_number * 10 * VertexFit::max_svs;
  uint* event_sv_number = parameters.dev_sv_atomics + event_number;
  uint* event_svs_trk1_idx = parameters.dev_svs_trk1_idx + idx_offset;
  uint* event_svs_trk2_idx = parameters.dev_svs_trk2_idx + idx_offset;
  
  // Consolidated SciFi tracks.
  SciFi::Consolidated::ConstTracks scifi_tracks {parameters.dev_atomics_scifi,
                                                 parameters.dev_scifi_track_hit_number,
                                                 parameters.dev_scifi_qop,
                                                 parameters.dev_scifi_states,
                                                 parameters.dev_scifi_track_ut_indices,
                                                 event_number,
                                                 number_of_events};
  const uint event_tracks_offset = scifi_tracks.tracks_offset(event_number);
  const uint n_scifi_tracks = scifi_tracks.number_of_tracks(event_number);

  // Track-PV association table.
  Associate::Consolidated::ConstTable kalman_pv_ipchi2 {parameters.dev_kalman_pv_ipchi2,
                                                        scifi_tracks.total_number_of_tracks()};
  const auto pv_table = kalman_pv_ipchi2.event_table(scifi_tracks, event_number);

  // Kalman fitted tracks.
  const ParKalmanFilter::FittedTrack* event_tracks = parameters.dev_kf_tracks + event_tracks_offset;

  // Primary vertices.
  const uint n_pvs_event = *(parameters.dev_number_of_multi_fit_vertices + event_number);
  cuda::span<PV::Vertex const> vertices {parameters.dev_multi_fit_vertices + event_number * PV::max_number_vertices, n_pvs_event};

  // Loop over tracks.
  for (auto i_track = threadIdx.x; i_track < n_scifi_tracks; i_track += blockDim.x) {

    // Filter first track.
    const ParKalmanFilter::FittedTrack trackA = event_tracks[i_track];
    if (
      trackA.pt() < parameters.track_min_pt ||
      (trackA.ipChi2 < parameters.track_min_ipchi2 && !trackA.is_muon)) {
      continue;
    }

    for (auto j_track = threadIdx.y + i_track + 1; j_track < n_scifi_tracks; j_track += blockDim.y) {

      // Filter second track.
      const ParKalmanFilter::FittedTrack trackB = event_tracks[j_track];
      if (
        trackB.pt() < parameters.track_min_pt ||
        (trackB.ipChi2 < parameters.track_min_ipchi2 && !trackB.is_muon)) {
        continue;
      }

      // Same PV cut for non-muons.
      if (
        pv_table.pv(i_track) != pv_table.pv(j_track) &&
        pv_table.value(i_track) < parameters.max_assoc_ipchi2 &&
        pv_table.value(j_track) < parameters.max_assoc_ipchi2 &&
        (!trackA.is_muon || !trackB.is_muon)
          ) {
        continue;
      }

      uint vertex_idx = atomicAdd(event_sv_number, 1);
      event_svs_trk1_idx[vertex_idx] = i_track;
      event_svs_trk2_idx[vertex_idx] = j_track;
      
    } 
  }
}