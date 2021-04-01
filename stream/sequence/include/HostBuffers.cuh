/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <vector>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <gsl/gsl>
#include "BackendCommon.h"

// Forward declarations
namespace PV {
  class Vertex;
}
namespace UT {
  struct TrackHits;
}
namespace SciFi {
  struct TrackHits;
}
struct MiniState;
namespace ParKalmanFilter {
  struct FittedTrack;
}
namespace VertexFit {
  struct TrackMVAVertex;
}

struct HostBuffers {
  // Pinned host datatypes
  unsigned host_number_of_events;
  unsigned host_number_of_selected_events;
  unsigned* host_event_list;
  unsigned* host_prefix_sum_buffer;
  bool* host_passing_event_list;
  uint32_t* host_sel_rep_raw_banks;
  unsigned host_sel_rep_raw_banks_size;
  size_t host_allocated_prefix_sum_space;
  unsigned* host_sel_rep_offsets;
  unsigned* host_number_of_sel_rep_words;

  // Velo
  unsigned* host_atomics_velo;
  unsigned* host_velo_track_hit_number;
  char* host_velo_track_hits;
  unsigned* host_total_number_of_velo_clusters;
  unsigned* host_number_of_reconstructed_velo_tracks;
  unsigned* host_accumulated_number_of_hits_in_velo_tracks;
  char* host_velo_kalman_beamline_states;
  char* host_velo_kalman_endvelo_states;
  PV::Vertex* host_reconstructed_pvs;
  int* host_number_of_vertex;
  int* host_number_of_seeds;
  unsigned* host_n_scifi_tracks;
  float* host_zhisto;
  float* host_peaks;
  unsigned* host_number_of_peaks;
  PV::Vertex* host_reconstructed_multi_pvs;
  int* host_number_of_multivertex;
  std::vector<unsigned> velo_clusters_offsets;
  std::vector<unsigned> velo_module_clusters_num;
  std::vector<char> velo_clusters;

  // UT
  unsigned* host_atomics_ut;
  UT::TrackHits* host_ut_tracks;
  unsigned* host_number_of_reconstructed_ut_tracks;
  unsigned* host_accumulated_number_of_ut_hits;
  unsigned* host_accumulated_number_of_hits_in_ut_tracks;
  unsigned* host_ut_track_hit_number;
  char* host_ut_track_hits;
  float* host_ut_qop;
  float* host_ut_x;
  float* host_ut_tx;
  float* host_ut_z;
  unsigned* host_ut_track_velo_indices;

  // SciFi
  unsigned* host_accumulated_number_of_scifi_hits;
  unsigned* host_number_of_reconstructed_scifi_tracks;
  SciFi::TrackHits* host_scifi_tracks;
  unsigned* host_atomics_scifi;
  unsigned* host_accumulated_number_of_hits_in_scifi_tracks;
  unsigned* host_scifi_track_hit_number;
  char* host_scifi_track_hits;
  float* host_scifi_qop;
  MiniState* host_scifi_states;
  unsigned* host_scifi_track_ut_indices;
  unsigned* host_lf_total_size_first_window_layer;
  unsigned* host_lf_total_number_of_candidates;

  // Kalman
  ParKalmanFilter::FittedTrack* host_kf_tracks;

  // Muon
  float* host_muon_catboost_output;
  bool* host_is_muon;
  unsigned* host_muon_total_number_of_tiles;
  unsigned* host_muon_total_number_of_hits;
  unsigned* host_selected_events_mf;
  unsigned* host_event_list_mf;
  bool* host_match_upstream_muon;

  // Calo
  gsl::span<unsigned> host_ecal_digits_offsets = {};
  gsl::span<unsigned> host_hcal_digits_offsets = {};
  gsl::span<char> host_ecal_digits = {};
  gsl::span<char> host_hcal_digits = {};
  gsl::span<unsigned> host_ecal_cluster_offsets = {};
  gsl::span<unsigned> host_hcal_cluster_offsets = {};
  gsl::span<char> host_ecal_clusters = {};
  gsl::span<char> host_hcal_clusters = {};

  // Secondary vertices
  unsigned* host_number_of_svs;
  unsigned* host_sv_offsets;
  unsigned* host_sv_atomics;
  unsigned* host_mf_sv_offsets;

  unsigned host_secondary_vertices_size;
  unsigned host_mf_secondary_vertices_size;
  VertexFit::TrackMVAVertex* host_secondary_vertices;
  VertexFit::TrackMVAVertex* host_mf_secondary_vertices;

  // Selections
  std::string host_names_of_lines;
  unsigned host_number_of_lines;
  gsl::span<uint32_t> host_dec_reports = {};
  gsl::span<bool> host_selections = {};
  gsl::span<unsigned> host_selections_offsets = {};

  // Non pinned datatypes: CPU algorithms
  std::vector<char> host_velo_states;
  std::vector<std::vector<std::vector<uint32_t>>> scifi_ids_ut_tracks;
  std::vector<unsigned> host_scifi_hits;
  std::vector<unsigned> host_scifi_hit_count;

  /**
   * @brief Reserves all host buffers.
   */
  void reserve(const unsigned max_number_of_events, const size_t n_lines, const bool do_check);

  // /**
  //  * @brief Frees all host buffers.
  //  */
  // Allen::error free(const bool do_check);

  /**
   * @brief Returns total number of velo track hits.
   */
  size_t velo_track_hit_number_size() const;

  /**
   * @brief Returns total number of UT track hits.
   */
  size_t ut_track_hit_number_size() const;

  /**
   * @brief Returns total number of SciFi track hits.
   */
  size_t scifi_track_hit_number_size() const;

  /**
   * @brief Retrieve total number of hit uints.
   */
  uint32_t scifi_hits_uints() const;
};
