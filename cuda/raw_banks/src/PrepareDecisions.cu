#include "PrepareDecisions.cuh"
#include "RawBanksDefinitions.cuh"

__global__ void prepare_decisions::prepare_decisions(prepare_decisions::Parameters parameters)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  // Create velo tracks.
  Velo::Consolidated::ConstTracks velo_tracks {
    parameters.dev_atomics_velo, parameters.dev_velo_track_hit_number, event_number, number_of_events};

  // Create UT tracks.
  UT::Consolidated::ConstExtendedTracks ut_tracks {parameters.dev_atomics_ut,
                                                   parameters.dev_ut_track_hit_number,
                                                   parameters.dev_ut_qop,
                                                   parameters.dev_ut_track_velo_indices,
                                                   event_number,
                                                   number_of_events};

  // Create SciFi tracks.
  SciFi::Consolidated::ConstTracks scifi_tracks {parameters.dev_offsets_forward_tracks,
                                                 parameters.dev_scifi_track_hit_number,
                                                 parameters.dev_scifi_qop,
                                                 parameters.dev_scifi_states,
                                                 parameters.dev_scifi_track_ut_indices,
                                                 event_number,
                                                 number_of_events};

  // Selection results.
  const uint* dev_sel_results_offsets = parameters.dev_sel_results_atomics + Hlt1::Hlt1Lines::End;
  
  // Tracks.
  int* event_save_track = parameters.dev_save_track + scifi_tracks.tracks_offset(event_number);
  const int n_tracks_event = scifi_tracks.number_of_tracks(event_number);
  uint* event_saved_tracks_list = parameters.dev_saved_tracks_list + scifi_tracks.tracks_offset(event_number);
  const ParKalmanFilter::FittedTrack* event_kf_tracks = parameters.dev_kf_tracks + scifi_tracks.tracks_offset(event_number);

  // Vertices.
  int* event_save_sv = parameters.dev_save_sv + parameters.dev_sv_offsets[event_number];
  const int n_vertices_event = parameters.dev_sv_offsets[event_number + 1] - parameters.dev_sv_offsets[event_number];
  uint* event_saved_svs_list = parameters.dev_saved_svs_list + parameters.dev_sv_offsets[event_number];
  const VertexFit::TrackMVAVertex* event_svs = parameters.dev_consolidated_svs + parameters.dev_sv_offsets[event_number];

  const int n_hlt1_lines = Hlt1::Hlt1Lines::End;
  uint32_t* event_dec_reports = parameters.dev_dec_reports + (2 + n_hlt1_lines) * event_number;

  // Set vertex decisions first.
  uint32_t dec_mask = HltDecReport::decReportMasks::decisionMask;
  uint insert_index = 0;
  for (uint i_sv = threadIdx.x; i_sv < n_vertices_event; i_sv += blockDim.x) {
    uint32_t save_sv = 0;
    for (uint i_line = Hlt1::startTwoTrackLines; i_line < Hlt1::startThreeTrackLines; i_line++) {
      const bool* decisions = parameters.dev_sel_results +
        dev_sel_results_offsets[i_line] + parameters.dev_sv_offsets[event_number];
      uint* candidate_counts = parameters.dev_candidate_counts + i_line * number_of_events + event_number;
      uint* candidate_list = parameters.dev_candidate_lists + number_of_events * Hlt1::maxCandidates * i_line +
      event_number * Hlt1::maxCandidates;
      uint32_t dec = ((decisions[i_sv] ? 1 : 0) & dec_mask);
      atomicOr(event_dec_reports + 2 + i_line, dec);
      insert_index = atomicAdd(candidate_counts, dec);
      save_sv |= dec;
      if (dec) {
        candidate_list[insert_index] = i_sv;
      }
    }
    if (save_sv & dec_mask) {
      const uint sv_insert_index = atomicAdd(parameters.dev_n_svs_saved + event_number, 1);
      event_save_sv[i_sv] = sv_insert_index;
      event_saved_svs_list[sv_insert_index] = i_sv;
      // Set to 1 for as a placeholder.
      event_save_track[event_svs[i_sv].trk1] = 1;
      event_save_track[event_svs[i_sv].trk2] = 1;
    }
  }
  __syncthreads();

  // Set track decisions.
  for (int i_track = threadIdx.x; i_track < n_tracks_event; i_track += blockDim.x) {
    uint32_t save_track = 0;
    for (uint i_line = Hlt1::startOneTrackLines; i_line < Hlt1::startTwoTrackLines; i_line++) {
      const bool* decisions = parameters.dev_sel_results +
        dev_sel_results_offsets[i_line] + scifi_tracks.tracks_offset(event_number);
      uint* candidate_counts = parameters.dev_candidate_counts + i_line * number_of_events + event_number;
      uint* candidate_list = parameters.dev_candidate_lists + number_of_events * Hlt1::maxCandidates * i_line +
        event_number * Hlt1::maxCandidates;
      uint32_t dec = ((decisions[i_track] ? 1 : 0) & dec_mask);
      atomicOr(event_dec_reports + 2 + i_line, dec);
      insert_index = atomicAdd(candidate_counts, dec);
      save_track |= dec;
      if (dec) {
        candidate_list[insert_index] = i_track;
      }
    }
    if (save_track) {
      event_save_track[i_track] = 1;
    }
    // Count the number of tracks and hits to save in the SelReport.
    if (event_save_track[i_track] >= 0) {
      const int i_ut_track = scifi_tracks.ut_track(i_track);
      const int i_velo_track = ut_tracks.velo_track(i_ut_track);
      const int n_hits = scifi_tracks.number_of_hits(i_track) +
        ut_tracks.number_of_hits(i_ut_track) + velo_tracks.number_of_hits(i_velo_track);
      const uint track_insert_index = atomicAdd(parameters.dev_n_tracks_saved + event_number, 1);
      atomicAdd(parameters.dev_n_hits_saved + event_number, n_hits);
      event_saved_tracks_list[track_insert_index] = (uint) i_track;
      event_save_track[i_track] = (int) track_insert_index;
    }
  }
}