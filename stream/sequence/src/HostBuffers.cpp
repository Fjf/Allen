/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "HostBuffers.cuh"
#include "BackendCommon.h"
#include "VeloEventModel.cuh"
#include "UTDefinitions.cuh"
#include "SciFiDefinitions.cuh"
#include "MuonDefinitions.cuh"
#include "TrackChecker.h"
#include "SciFiEventModel.cuh"
#include "UTEventModel.cuh"
#include "patPV_Definitions.cuh"
#include "PV_Definitions.cuh"
#include "ParKalmanDefinitions.cuh"
#include "VertexDefinitions.cuh"
#include "BeamlinePVConstants.cuh"
#include "LookingForwardConstants.cuh"

void HostBuffers::reserve(const unsigned max_number_of_events, const size_t n_lines)
{
  constexpr unsigned host_buffers_max_velo_tracks = 4000;

  // Datatypes needed to run, regardless of checking
  // Note: These datatypes must be pinned to allow for asynchronicity
  Allen::malloc_host((void**) &host_number_of_events, sizeof(unsigned));
  Allen::malloc_host((void**) &host_total_number_of_velo_clusters, sizeof(unsigned));
  Allen::malloc_host((void**) &host_number_of_reconstructed_velo_tracks, sizeof(unsigned));
  Allen::malloc_host((void**) &host_accumulated_number_of_hits_in_velo_tracks, sizeof(unsigned));
  Allen::malloc_host((void**) &host_accumulated_number_of_ut_hits, sizeof(unsigned));
  Allen::malloc_host((void**) &host_number_of_reconstructed_ut_tracks, sizeof(unsigned));
  Allen::malloc_host((void**) &host_accumulated_number_of_hits_in_ut_tracks, sizeof(unsigned));
  Allen::malloc_host((void**) &host_accumulated_number_of_scifi_hits, sizeof(unsigned));
  Allen::malloc_host((void**) &host_number_of_reconstructed_scifi_tracks, sizeof(unsigned));
  Allen::malloc_host((void**) &host_accumulated_number_of_hits_in_scifi_tracks, sizeof(unsigned));
  Allen::malloc_host((void**) &host_lf_total_number_of_candidates, sizeof(unsigned));
  Allen::malloc_host((void**) &host_lf_total_size_first_window_layer, sizeof(unsigned));
  Allen::malloc_host((void**) &host_muon_total_number_of_tiles, sizeof(unsigned));
  Allen::malloc_host((void**) &host_number_of_svs, sizeof(unsigned));
  Allen::malloc_host((void**) &host_muon_total_number_of_hits, sizeof(unsigned));
  Allen::malloc_host((void**) &host_number_of_sel_rep_words, sizeof(unsigned));
  Allen::malloc_host((void**) &host_selected_events_mf, sizeof(unsigned));

  // Initialize for sequences that don't fill this in.
  host_number_of_events = 0;

  // Buffer for performing GEC on CPU
  Allen::malloc_host((void**) &host_event_list, max_number_of_events * sizeof(unsigned));

  // Buffer for saving dec reports to the host.
  uint32_t* dec_reports = nullptr;
  size_t const dec_reports_size = max_number_of_events * (n_lines + 2) * sizeof(uint32_t);
  Allen::malloc_host((void**) &dec_reports, dec_reports_size);
  ::memset(dec_reports, 0, dec_reports_size);
  host_dec_reports = {dec_reports, dec_reports_size};

  // Buffer for saving sel reports to the host.
  uint32_t* sel_reports = nullptr;
  // SelReports for events selected by the passthrough line have a
  // size of 20. SelReports for almost all events selected by physics
  // lines have a size of 100-200 with almost all <300. So 300 is
  // chosen as a reasonable safe value for the maximum average
  // SelReport size for all events.
  size_t const max_average_sel_report_size = 300;
  size_t const sel_reports_size = max_number_of_events * max_average_sel_report_size * sizeof(uint32_t);
  Allen::malloc_host((void**) &sel_reports, sel_reports_size);
  ::memset(sel_reports, 0, sel_reports_size);
  host_sel_reports = {sel_reports, sel_reports_size};

  uint32_t* sel_report_offsets = nullptr;
  Allen::malloc_host((void**) &sel_report_offsets, (max_number_of_events + 1) * sizeof(uint32_t));
  ::memset(sel_report_offsets, 0, (max_number_of_events + 1) * sizeof(uint32_t));
  host_sel_report_offsets = {sel_report_offsets, (max_number_of_events + 1) * sizeof(uint32_t)};

  // Buffer for saving events passing Hlt1 selections.
  Allen::malloc_host((void**) &host_passing_event_list, max_number_of_events * sizeof(bool));
  ::memset(host_passing_event_list, 0, max_number_of_events * sizeof(bool));

  // Buffer for performing prefix sum
  // Note: If it is of insufficient space, it will get reallocated
  host_allocated_prefix_sum_space = 10000000;
  Allen::malloc_host((void**) &host_prefix_sum_buffer, host_allocated_prefix_sum_space * sizeof(unsigned));

  // Needed for track monitoring
  auto const atomics_scifi_size = max_number_of_events * SciFi::num_atomics * sizeof(int);
  Allen::malloc_host((void**) &host_atomics_scifi, atomics_scifi_size);
  Allen::malloc_host(
    (void**) &host_kf_tracks,
    max_number_of_events * SciFi::Constants::max_tracks * sizeof(ParKalmanFilter::FittedTrack));
  ::memset(host_atomics_scifi, 0, atomics_scifi_size);

  // Needed for PV monitoring
  Allen::malloc_host(
    (void**) &host_reconstructed_multi_pvs, max_number_of_events * PV::max_number_vertices * sizeof(PV::Vertex));
  Allen::malloc_host((void**) &host_number_of_multivertex, max_number_of_events * sizeof(int));
  ::memset(host_number_of_multivertex, 0, max_number_of_events * sizeof(int));

  // Needed for SV monitoring,
  // FIXME: 500 was estimated as a sane starting value from a sample
  // of 5000 BsPhiPhi events. The monitoring of secondary vertices
  // should be improved to not require allocating a large chunk of
  // memory and an initial estimate of the number of vertices
  host_secondary_vertices_size = max_number_of_events * 500 * sizeof(VertexFit::TrackMVAVertex);
  Allen::malloc_host((void**) &host_secondary_vertices, host_secondary_vertices_size);
  auto const sv_offsets_size = (max_number_of_events + 1) * sizeof(unsigned);
  Allen::malloc_host((void**) &host_sv_offsets, sv_offsets_size);
  ::memset(host_sv_offsets, 0, sv_offsets_size);

  // Datatypes to be reserved only if checking is on
  // Note: These datatypes in principle do not require to be pinned
  host_atomics_velo =
    reinterpret_cast<decltype(host_atomics_velo)>(malloc((2 * max_number_of_events + 1) * sizeof(int)));
  host_velo_track_hit_number = reinterpret_cast<decltype(host_velo_track_hit_number)>(
    malloc(max_number_of_events * host_buffers_max_velo_tracks * sizeof(unsigned)));
  host_velo_track_hits = reinterpret_cast<decltype(host_velo_track_hits)>(
    malloc(max_number_of_events * host_buffers_max_velo_tracks * Velo::Constants::max_track_size * sizeof(Velo::Hit)));
  host_velo_kalman_beamline_states = reinterpret_cast<decltype(host_velo_kalman_beamline_states)>(
    malloc(max_number_of_events * host_buffers_max_velo_tracks * Velo::Consolidated::States::size));
  host_velo_kalman_endvelo_states = reinterpret_cast<decltype(host_velo_kalman_endvelo_states)>(
    malloc(max_number_of_events * host_buffers_max_velo_tracks * Velo::Consolidated::States::size));

  host_atomics_ut =
    reinterpret_cast<decltype(host_atomics_ut)>(malloc(UT::num_atomics * (max_number_of_events + 1) * sizeof(int)));
  host_ut_tracks = reinterpret_cast<decltype(host_ut_tracks)>(
    malloc(max_number_of_events * UT::Constants::max_num_tracks * sizeof(UT::TrackHits)));
  host_ut_track_hit_number = reinterpret_cast<decltype(host_ut_track_hit_number)>(
    malloc(max_number_of_events * UT::Constants::max_num_tracks * sizeof(unsigned)));

  const size_t host_ut_track_hits_sz =
    max_number_of_events * UT::Constants::max_num_tracks * UT::Constants::max_track_size * sizeof(UT::Hit);
  host_ut_track_hits = gsl::span<char> {reinterpret_cast<char*>(malloc(host_ut_track_hits_sz)), host_ut_track_hits_sz};

  host_ut_qop = reinterpret_cast<decltype(host_ut_qop)>(
    malloc(max_number_of_events * UT::Constants::max_num_tracks * sizeof(float)));
  host_ut_track_velo_indices = reinterpret_cast<decltype(host_ut_track_velo_indices)>(
    malloc(max_number_of_events * UT::Constants::max_num_tracks * sizeof(int)));
  host_scifi_tracks = reinterpret_cast<decltype(host_scifi_tracks)>(malloc(
    max_number_of_events * UT::Constants::max_num_tracks * LookingForward::maximum_number_of_candidates_per_ut_track *
    sizeof(SciFi::TrackHits)));
  host_scifi_track_hit_number = reinterpret_cast<decltype(host_scifi_track_hit_number)>(malloc(
    max_number_of_events * UT::Constants::max_num_tracks * LookingForward::maximum_number_of_candidates_per_ut_track *
    sizeof(unsigned)));
  host_scifi_track_hits = reinterpret_cast<decltype(host_scifi_track_hits)>(malloc(
    max_number_of_events * UT::Constants::max_num_tracks * LookingForward::maximum_number_of_candidates_per_ut_track *
    SciFi::Constants::max_track_size * sizeof(SciFi::Hit)));
  host_scifi_qop = reinterpret_cast<decltype(host_scifi_qop)>(malloc(
    max_number_of_events * UT::Constants::max_num_tracks * LookingForward::maximum_number_of_candidates_per_ut_track *
    sizeof(float)));
  host_scifi_states = reinterpret_cast<decltype(host_scifi_states)>(malloc(
    max_number_of_events * UT::Constants::max_num_tracks * LookingForward::maximum_number_of_candidates_per_ut_track *
    sizeof(MiniState)));
  host_scifi_track_ut_indices = reinterpret_cast<decltype(host_scifi_track_ut_indices)>(malloc(
    max_number_of_events * UT::Constants::max_num_tracks * LookingForward::maximum_number_of_candidates_per_ut_track *
    sizeof(unsigned)));

  host_reconstructed_pvs = reinterpret_cast<decltype(host_reconstructed_pvs)>(
    malloc(max_number_of_events * PV::max_number_vertices * sizeof(PV::Vertex)));
  host_number_of_vertex = reinterpret_cast<decltype(host_number_of_vertex)>(malloc(max_number_of_events * sizeof(int)));
  host_number_of_seeds = reinterpret_cast<decltype(host_number_of_seeds)>(malloc(max_number_of_events * sizeof(int)));
  host_zhisto = reinterpret_cast<decltype(host_zhisto)>(malloc(
    max_number_of_events * sizeof(float) * (BeamlinePVConstants::Common::zmax - BeamlinePVConstants::Common::zmin) /
    BeamlinePVConstants::Common::dz));
  host_peaks =
    reinterpret_cast<decltype(host_peaks)>(malloc(max_number_of_events * sizeof(float) * PV::max_number_vertices));
  host_number_of_peaks =
    reinterpret_cast<decltype(host_number_of_peaks)>(malloc(max_number_of_events * sizeof(unsigned)));

  host_muon_catboost_output = reinterpret_cast<decltype(host_muon_catboost_output)>(
    malloc(max_number_of_events * SciFi::Constants::max_tracks * sizeof(float)));
  host_is_muon = reinterpret_cast<decltype(host_is_muon)>(
    malloc(max_number_of_events * SciFi::Constants::max_tracks * sizeof(bool)));
  host_event_list_mf = reinterpret_cast<decltype(host_event_list_mf)>(malloc(max_number_of_events * sizeof(unsigned)));
  host_match_upstream_muon = reinterpret_cast<decltype(host_match_upstream_muon)>(
    malloc(max_number_of_events * UT::Constants::max_num_tracks * sizeof(bool)));

  host_sv_atomics =
    reinterpret_cast<decltype(host_sv_atomics)>(malloc((2 * max_number_of_events + 1) * sizeof(unsigned)));
}
