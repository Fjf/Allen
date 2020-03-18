#pragma once

#include <tuple>
#include "../../cuda/selections/Hlt1/include/LineTraverser.cuh"
#include "../../cuda/raw_banks/include/PopulateOdinBanks.cuh"
#include "../../x86/global_event_cut/include/HostGlobalEventCut.h"
#include "../../cuda/velo/mask_clustering/include/VeloCalculateNumberOfCandidates.cuh"
#include "../../x86/prefix_sum/include/HostPrefixSum.h"
#include "../../cuda/velo/mask_clustering/include/EstimateInputSize.cuh"
#include "../../x86/prefix_sum/include/HostPrefixSum.h"
#include "../../cuda/velo/mask_clustering/include/MaskedVeloClustering.cuh"
#include "../../cuda/velo/calculate_phi_and_sort/include/CalculatePhiAndSort.cuh"
#include "../../cuda/velo/search_by_triplet/include/FillCandidates.cuh"
#include "../../cuda/velo/search_by_triplet/include/SearchByTriplet.cuh"
#include "../../x86/prefix_sum/include/HostPrefixSum.h"
#include "../../cuda/velo/search_by_triplet/include/ThreeHitTracksFilter.cuh"
#include "../../x86/prefix_sum/include/HostPrefixSum.h"
#include "../../cuda/velo/consolidate_tracks/include/VeloCopyTrackHitNumber.cuh"
#include "../../x86/prefix_sum/include/HostPrefixSum.h"
#include "../../cuda/velo/consolidate_tracks/include/VeloConsolidateTracks.cuh"
#include "../../cuda/velo/simplified_kalman_filter/include/VeloKalmanFilter.cuh"
#include "../../cuda/PV/beamlinePV/include/pv_beamline_extrapolate.cuh"
#include "../../cuda/PV/beamlinePV/include/pv_beamline_histo.cuh"
#include "../../cuda/PV/beamlinePV/include/pv_beamline_peak.cuh"
#include "../../cuda/PV/beamlinePV/include/pv_beamline_calculate_denom.cuh"
#include "../../cuda/PV/beamlinePV/include/pv_beamline_multi_fitter.cuh"
#include "../../cuda/PV/beamlinePV/include/pv_beamline_cleanup.cuh"

struct dev_odin_raw_input_t : populate_odin_banks::Parameters::dev_odin_raw_input_t {
  constexpr static auto name {"dev_odin_raw_input_t"};
  size_t size;
  char* offset;
};
struct dev_odin_raw_input_offsets_t : populate_odin_banks::Parameters::dev_odin_raw_input_offsets_t {
  constexpr static auto name {"dev_odin_raw_input_offsets_t"};
  size_t size;
  char* offset;
};
struct host_total_number_of_events_t : host_global_event_cut::Parameters::host_total_number_of_events_t {
  constexpr static auto name {"host_total_number_of_events_t"};
  size_t size;
  char* offset;
};
struct host_event_list_t : host_global_event_cut::Parameters::host_event_list_t {
  constexpr static auto name {"host_event_list_t"};
  size_t size;
  char* offset;
};
struct host_number_of_selected_events_t
  : pv_beamline_extrapolate::Parameters::host_number_of_selected_events_t,
    host_global_event_cut::Parameters::host_number_of_selected_events_t,
    velo_kalman_filter::Parameters::host_number_of_selected_events_t,
    velo_estimate_input_size::Parameters::host_number_of_selected_events_t,
    pv_beamline_multi_fitter::Parameters::host_number_of_selected_events_t,
    velo_fill_candidates::Parameters::host_number_of_selected_events_t,
    velo_calculate_number_of_candidates::Parameters::host_number_of_selected_events_t,
    velo_three_hit_tracks_filter::Parameters::host_number_of_selected_events_t,
    velo_search_by_triplet::Parameters::host_number_of_selected_events_t,
    pv_beamline_histo::Parameters::host_number_of_selected_events_t,
    velo_consolidate_tracks::Parameters::host_number_of_selected_events_t,
    pv_beamline_calculate_denom::Parameters::host_number_of_selected_events_t,
    velo_copy_track_hit_number::Parameters::host_number_of_selected_events_t,
    velo_masked_clustering::Parameters::host_number_of_selected_events_t,
    pv_beamline_cleanup::Parameters::host_number_of_selected_events_t,
    pv_beamline_peak::Parameters::host_number_of_selected_events_t,
    velo_calculate_phi_and_sort::Parameters::host_number_of_selected_events_t {
  constexpr static auto name {"host_number_of_selected_events_t"};
  size_t size;
  char* offset;
};
struct dev_event_list_t : velo_masked_clustering::Parameters::dev_event_list_t,
                          host_global_event_cut::Parameters::dev_event_list_t,
                          velo_estimate_input_size::Parameters::dev_event_list_t,
                          velo_calculate_number_of_candidates::Parameters::dev_event_list_t {
  constexpr static auto name {"dev_event_list_t"};
  size_t size;
  char* offset;
};
struct dev_velo_raw_input_t : velo_calculate_number_of_candidates::Parameters::dev_velo_raw_input_t,
                              velo_estimate_input_size::Parameters::dev_velo_raw_input_t,
                              velo_masked_clustering::Parameters::dev_velo_raw_input_t {
  constexpr static auto name {"dev_velo_raw_input_t"};
  size_t size;
  char* offset;
};
struct dev_velo_raw_input_offsets_t : velo_calculate_number_of_candidates::Parameters::dev_velo_raw_input_offsets_t,
                                      velo_estimate_input_size::Parameters::dev_velo_raw_input_offsets_t,
                                      velo_masked_clustering::Parameters::dev_velo_raw_input_offsets_t {
  constexpr static auto name {"dev_velo_raw_input_offsets_t"};
  size_t size;
  char* offset;
};
struct dev_number_of_candidates_t : velo_calculate_number_of_candidates::Parameters::dev_number_of_candidates_t,
                                    host_prefix_sum::Parameters::dev_input_buffer_t {
  constexpr static auto name {"dev_number_of_candidates_t"};
  size_t size;
  char* offset;
};
struct host_number_of_cluster_candidates_t : velo_estimate_input_size::Parameters::host_number_of_cluster_candidates_t,
                                             host_prefix_sum::Parameters::host_total_sum_holder_t {
  constexpr static auto name {"host_number_of_cluster_candidates_t"};
  size_t size;
  char* offset;
};
struct dev_candidates_offsets_t : host_prefix_sum::Parameters::dev_output_buffer_t,
                                  velo_masked_clustering::Parameters::dev_candidates_offsets_t,
                                  velo_estimate_input_size::Parameters::dev_candidates_offsets_t {
  constexpr static auto name {"dev_candidates_offsets_t"};
  size_t size;
  char* offset;
};
struct dev_estimated_input_size_t : host_prefix_sum::Parameters::dev_input_buffer_t,
                                    velo_estimate_input_size::Parameters::dev_estimated_input_size_t {
  constexpr static auto name {"dev_estimated_input_size_t"};
  size_t size;
  char* offset;
};
struct dev_module_candidate_num_t : velo_estimate_input_size::Parameters::dev_module_candidate_num_t,
                                    velo_masked_clustering::Parameters::dev_module_candidate_num_t {
  constexpr static auto name {"dev_module_candidate_num_t"};
  size_t size;
  char* offset;
};
struct dev_cluster_candidates_t : velo_estimate_input_size::Parameters::dev_cluster_candidates_t,
                                  velo_masked_clustering::Parameters::dev_cluster_candidates_t {
  constexpr static auto name {"dev_cluster_candidates_t"};
  size_t size;
  char* offset;
};
struct host_total_number_of_velo_clusters_t
  : velo_search_by_triplet::Parameters::host_total_number_of_velo_clusters_t,
    velo_fill_candidates::Parameters::host_total_number_of_velo_clusters_t,
    velo_masked_clustering::Parameters::host_total_number_of_velo_clusters_t,
    host_prefix_sum::Parameters::host_total_sum_holder_t,
    velo_calculate_phi_and_sort::Parameters::host_total_number_of_velo_clusters_t {
  constexpr static auto name {"host_total_number_of_velo_clusters_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_estimated_input_size_t
  : host_prefix_sum::Parameters::dev_output_buffer_t,
    velo_three_hit_tracks_filter::Parameters::dev_offsets_estimated_input_size_t,
    velo_fill_candidates::Parameters::dev_offsets_estimated_input_size_t,
    velo_search_by_triplet::Parameters::dev_offsets_estimated_input_size_t,
    velo_masked_clustering::Parameters::dev_offsets_estimated_input_size_t,
    velo_calculate_phi_and_sort::Parameters::dev_offsets_estimated_input_size_t,
    velo_consolidate_tracks::Parameters::dev_offsets_estimated_input_size_t {
  constexpr static auto name {"dev_offsets_estimated_input_size_t"};
  size_t size;
  char* offset;
};
struct dev_module_cluster_num_t : velo_search_by_triplet::Parameters::dev_module_cluster_num_t,
                                  velo_masked_clustering::Parameters::dev_module_cluster_num_t,
                                  velo_fill_candidates::Parameters::dev_module_cluster_num_t,
                                  velo_calculate_phi_and_sort::Parameters::dev_module_cluster_num_t {
  constexpr static auto name {"dev_module_cluster_num_t"};
  size_t size;
  char* offset;
};
struct dev_velo_cluster_container_t : velo_calculate_phi_and_sort::Parameters::dev_velo_cluster_container_t,
                                      velo_masked_clustering::Parameters::dev_velo_cluster_container_t {
  constexpr static auto name {"dev_velo_cluster_container_t"};
  size_t size;
  char* offset;
};
struct dev_sorted_velo_cluster_container_t
  : velo_fill_candidates::Parameters::dev_sorted_velo_cluster_container_t,
    velo_consolidate_tracks::Parameters::dev_sorted_velo_cluster_container_t,
    velo_calculate_phi_and_sort::Parameters::dev_sorted_velo_cluster_container_t,
    velo_three_hit_tracks_filter::Parameters::dev_sorted_velo_cluster_container_t,
    velo_search_by_triplet::Parameters::dev_sorted_velo_cluster_container_t {
  constexpr static auto name {"dev_sorted_velo_cluster_container_t"};
  size_t size;
  char* offset;
};
struct dev_hit_permutation_t : velo_calculate_phi_and_sort::Parameters::dev_hit_permutation_t {
  constexpr static auto name {"dev_hit_permutation_t"};
  size_t size;
  char* offset;
};
struct dev_hit_phi_t : velo_fill_candidates::Parameters::dev_hit_phi_t,
                       velo_calculate_phi_and_sort::Parameters::dev_hit_phi_t,
                       velo_search_by_triplet::Parameters::dev_hit_phi_t {
  constexpr static auto name {"dev_hit_phi_t"};
  size_t size;
  char* offset;
};
struct dev_h0_candidates_t : velo_fill_candidates::Parameters::dev_h0_candidates_t,
                             velo_search_by_triplet::Parameters::dev_h0_candidates_t {
  constexpr static auto name {"dev_h0_candidates_t"};
  size_t size;
  char* offset;
};
struct dev_h2_candidates_t : velo_search_by_triplet::Parameters::dev_h2_candidates_t,
                             velo_fill_candidates::Parameters::dev_h2_candidates_t {
  constexpr static auto name {"dev_h2_candidates_t"};
  size_t size;
  char* offset;
};
struct dev_tracks_t : velo_consolidate_tracks::Parameters::dev_tracks_t,
                      velo_search_by_triplet::Parameters::dev_tracks_t,
                      velo_copy_track_hit_number::Parameters::dev_tracks_t {
  constexpr static auto name {"dev_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_tracklets_t : velo_search_by_triplet::Parameters::dev_tracklets_t {
  constexpr static auto name {"dev_tracklets_t"};
  size_t size;
  char* offset;
};
struct dev_tracks_to_follow_t : velo_search_by_triplet::Parameters::dev_tracks_to_follow_t {
  constexpr static auto name {"dev_tracks_to_follow_t"};
  size_t size;
  char* offset;
};
struct dev_three_hit_tracks_t : velo_three_hit_tracks_filter::Parameters::dev_three_hit_tracks_input_t,
                                velo_search_by_triplet::Parameters::dev_three_hit_tracks_t {
  constexpr static auto name {"dev_three_hit_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_hit_used_t : velo_search_by_triplet::Parameters::dev_hit_used_t,
                        velo_three_hit_tracks_filter::Parameters::dev_hit_used_t {
  constexpr static auto name {"dev_hit_used_t"};
  size_t size;
  char* offset;
};
struct dev_atomics_velo_t : velo_search_by_triplet::Parameters::dev_atomics_velo_t,
                            velo_three_hit_tracks_filter::Parameters::dev_atomics_velo_t {
  constexpr static auto name {"dev_atomics_velo_t"};
  size_t size;
  char* offset;
};
struct dev_rel_indices_t : velo_search_by_triplet::Parameters::dev_rel_indices_t {
  constexpr static auto name {"dev_rel_indices_t"};
  size_t size;
  char* offset;
};
struct dev_number_of_velo_tracks_t : velo_search_by_triplet::Parameters::dev_number_of_velo_tracks_t,
                                     host_prefix_sum::Parameters::dev_input_buffer_t {
  constexpr static auto name {"dev_number_of_velo_tracks_t"};
  size_t size;
  char* offset;
};
struct host_number_of_velo_tracks_at_least_four_hits_t
  : host_prefix_sum::Parameters::host_total_sum_holder_t,
    velo_copy_track_hit_number::Parameters::host_number_of_velo_tracks_at_least_four_hits_t {
  constexpr static auto name {"host_number_of_velo_tracks_at_least_four_hits_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_velo_tracks_t : host_prefix_sum::Parameters::dev_output_buffer_t,
                                   velo_copy_track_hit_number::Parameters::dev_offsets_velo_tracks_t {
  constexpr static auto name {"dev_offsets_velo_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_three_hit_tracks_output_t : velo_consolidate_tracks::Parameters::dev_three_hit_tracks_output_t,
                                       velo_three_hit_tracks_filter::Parameters::dev_three_hit_tracks_output_t {
  constexpr static auto name {"dev_three_hit_tracks_output_t"};
  size_t size;
  char* offset;
};
struct dev_number_of_three_hit_tracks_output_t
  : velo_three_hit_tracks_filter::Parameters::dev_number_of_three_hit_tracks_output_t,
    host_prefix_sum::Parameters::dev_input_buffer_t {
  constexpr static auto name {"dev_number_of_three_hit_tracks_output_t"};
  size_t size;
  char* offset;
};
struct host_number_of_three_hit_tracks_filtered_t
  : host_prefix_sum::Parameters::host_total_sum_holder_t,
    velo_consolidate_tracks::Parameters::host_number_of_three_hit_tracks_filtered_t,
    velo_copy_track_hit_number::Parameters::host_number_of_three_hit_tracks_filtered_t {
  constexpr static auto name {"host_number_of_three_hit_tracks_filtered_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_number_of_three_hit_tracks_filtered_t
  : host_prefix_sum::Parameters::dev_output_buffer_t,
    velo_consolidate_tracks::Parameters::dev_offsets_number_of_three_hit_tracks_filtered_t,
    velo_copy_track_hit_number::Parameters::dev_offsets_number_of_three_hit_tracks_filtered_t {
  constexpr static auto name {"dev_offsets_number_of_three_hit_tracks_filtered_t"};
  size_t size;
  char* offset;
};
struct host_number_of_reconstructed_velo_tracks_t
  : velo_copy_track_hit_number::Parameters::host_number_of_reconstructed_velo_tracks_t,
    velo_kalman_filter::Parameters::host_number_of_reconstructed_velo_tracks_t,
    pv_beamline_calculate_denom::Parameters::host_number_of_reconstructed_velo_tracks_t,
    pv_beamline_extrapolate::Parameters::host_number_of_reconstructed_velo_tracks_t,
    pv_beamline_multi_fitter::Parameters::host_number_of_reconstructed_velo_tracks_t,
    velo_consolidate_tracks::Parameters::host_number_of_reconstructed_velo_tracks_t {
  constexpr static auto name {"host_number_of_reconstructed_velo_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_velo_track_hit_number_t : velo_copy_track_hit_number::Parameters::dev_velo_track_hit_number_t,
                                     host_prefix_sum::Parameters::dev_input_buffer_t {
  constexpr static auto name {"dev_velo_track_hit_number_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_all_velo_tracks_t : pv_beamline_histo::Parameters::dev_offsets_all_velo_tracks_t,
                                       velo_consolidate_tracks::Parameters::dev_offsets_all_velo_tracks_t,
                                       pv_beamline_extrapolate::Parameters::dev_offsets_all_velo_tracks_t,
                                       velo_kalman_filter::Parameters::dev_offsets_all_velo_tracks_t,
                                       pv_beamline_calculate_denom::Parameters::dev_offsets_all_velo_tracks_t,
                                       pv_beamline_multi_fitter::Parameters::dev_offsets_all_velo_tracks_t,
                                       velo_copy_track_hit_number::Parameters::dev_offsets_all_velo_tracks_t {
  constexpr static auto name {"dev_offsets_all_velo_tracks_t"};
  size_t size;
  char* offset;
};
struct host_accumulated_number_of_hits_in_velo_tracks_t
  : host_prefix_sum::Parameters::host_total_sum_holder_t,
    velo_consolidate_tracks::Parameters::host_accumulated_number_of_hits_in_velo_tracks_t {
  constexpr static auto name {"host_accumulated_number_of_hits_in_velo_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_velo_track_hit_number_t
  : host_prefix_sum::Parameters::dev_output_buffer_t,
    velo_kalman_filter::Parameters::dev_offsets_velo_track_hit_number_t,
    velo_consolidate_tracks::Parameters::dev_offsets_velo_track_hit_number_t,
    pv_beamline_calculate_denom::Parameters::dev_offsets_velo_track_hit_number_t,
    pv_beamline_extrapolate::Parameters::dev_offsets_velo_track_hit_number_t,
    pv_beamline_histo::Parameters::dev_offsets_velo_track_hit_number_t,
    pv_beamline_multi_fitter::Parameters::dev_offsets_velo_track_hit_number_t {
  constexpr static auto name {"dev_offsets_velo_track_hit_number_t"};
  size_t size;
  char* offset;
};
struct dev_accepted_velo_tracks_t : velo_consolidate_tracks::Parameters::dev_accepted_velo_tracks_t {
  constexpr static auto name {"dev_accepted_velo_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_velo_states_t : velo_consolidate_tracks::Parameters::dev_velo_states_t,
                           velo_kalman_filter::Parameters::dev_velo_states_t {
  constexpr static auto name {"dev_velo_states_t"};
  size_t size;
  char* offset;
};
struct dev_velo_track_hits_t : velo_consolidate_tracks::Parameters::dev_velo_track_hits_t,
                               velo_kalman_filter::Parameters::dev_velo_track_hits_t {
  constexpr static auto name {"dev_velo_track_hits_t"};
  size_t size;
  char* offset;
};
struct dev_velo_kalman_beamline_states_t : pv_beamline_extrapolate::Parameters::dev_velo_kalman_beamline_states_t,
                                           velo_kalman_filter::Parameters::dev_velo_kalman_beamline_states_t {
  constexpr static auto name {"dev_velo_kalman_beamline_states_t"};
  size_t size;
  char* offset;
};
struct dev_pvtracks_t : pv_beamline_extrapolate::Parameters::dev_pvtracks_t,
                        pv_beamline_calculate_denom::Parameters::dev_pvtracks_t,
                        pv_beamline_multi_fitter::Parameters::dev_pvtracks_t,
                        pv_beamline_histo::Parameters::dev_pvtracks_t {
  constexpr static auto name {"dev_pvtracks_t"};
  size_t size;
  char* offset;
};
struct dev_pvtrack_z_t : pv_beamline_extrapolate::Parameters::dev_pvtrack_z_t,
                         pv_beamline_multi_fitter::Parameters::dev_pvtrack_z_t {
  constexpr static auto name {"dev_pvtrack_z_t"};
  size_t size;
  char* offset;
};
struct dev_zhisto_t : pv_beamline_peak::Parameters::dev_zhisto_t, pv_beamline_histo::Parameters::dev_zhisto_t {
  constexpr static auto name {"dev_zhisto_t"};
  size_t size;
  char* offset;
};
struct dev_zpeaks_t : pv_beamline_multi_fitter::Parameters::dev_zpeaks_t,
                      pv_beamline_calculate_denom::Parameters::dev_zpeaks_t,
                      pv_beamline_peak::Parameters::dev_zpeaks_t {
  constexpr static auto name {"dev_zpeaks_t"};
  size_t size;
  char* offset;
};
struct dev_number_of_zpeaks_t : pv_beamline_peak::Parameters::dev_number_of_zpeaks_t,
                                pv_beamline_multi_fitter::Parameters::dev_number_of_zpeaks_t,
                                pv_beamline_calculate_denom::Parameters::dev_number_of_zpeaks_t {
  constexpr static auto name {"dev_number_of_zpeaks_t"};
  size_t size;
  char* offset;
};
struct dev_pvtracks_denom_t : pv_beamline_multi_fitter::Parameters::dev_pvtracks_denom_t,
                              pv_beamline_calculate_denom::Parameters::dev_pvtracks_denom_t {
  constexpr static auto name {"dev_pvtracks_denom_t"};
  size_t size;
  char* offset;
};
struct dev_multi_fit_vertices_t : pv_beamline_cleanup::Parameters::dev_multi_fit_vertices_t,
                                  pv_beamline_multi_fitter::Parameters::dev_multi_fit_vertices_t {
  constexpr static auto name {"dev_multi_fit_vertices_t"};
  size_t size;
  char* offset;
};
struct dev_number_of_multi_fit_vertices_t : pv_beamline_cleanup::Parameters::dev_number_of_multi_fit_vertices_t,
                                            pv_beamline_multi_fitter::Parameters::dev_number_of_multi_fit_vertices_t {
  constexpr static auto name {"dev_number_of_multi_fit_vertices_t"};
  size_t size;
  char* offset;
};
struct dev_multi_final_vertices_t : pv_beamline_cleanup::Parameters::dev_multi_final_vertices_t {
  constexpr static auto name {"dev_multi_final_vertices_t"};
  size_t size;
  char* offset;
};
struct dev_number_of_multi_final_vertices_t : pv_beamline_cleanup::Parameters::dev_number_of_multi_final_vertices_t {
  constexpr static auto name {"dev_number_of_multi_final_vertices_t"};
  size_t size;
  char* offset;
};

using configured_lines_t = std::tuple<>;

using configured_sequence_t = std::tuple<
  populate_odin_banks::populate_odin_banks_t<
    std::tuple<dev_odin_raw_input_t, dev_odin_raw_input_offsets_t>,
    configured_lines_t,
    'p',
    'o',
    'p',
    'u',
    'l',
    'a',
    't',
    'e',
    '_',
    'o',
    'd',
    'i',
    'n',
    '_',
    'b',
    'a',
    'n',
    'k',
    's',
    '_',
    't'>,
  host_global_event_cut::host_global_event_cut_t<
    std::tuple<host_total_number_of_events_t, host_event_list_t, host_number_of_selected_events_t, dev_event_list_t>,
    'h',
    'o',
    's',
    't',
    '_',
    'g',
    'l',
    'o',
    'b',
    'a',
    'l',
    '_',
    'e',
    'v',
    'e',
    'n',
    't',
    '_',
    'c',
    'u',
    't',
    '_',
    't'>,
  velo_calculate_number_of_candidates::velo_calculate_number_of_candidates_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_event_list_t,
      dev_velo_raw_input_t,
      dev_velo_raw_input_offsets_t,
      dev_number_of_candidates_t>,
    'v',
    'e',
    'l',
    'o',
    '_',
    'c',
    'a',
    'l',
    'c',
    'u',
    'l',
    'a',
    't',
    'e',
    '_',
    'n',
    'u',
    'm',
    'b',
    'e',
    'r',
    '_',
    'o',
    'f',
    '_',
    'c',
    'a',
    'n',
    'd',
    'i',
    'd',
    'a',
    't',
    'e',
    's',
    '_',
    't'>,
  host_prefix_sum::host_prefix_sum_t<
    std::tuple<host_number_of_cluster_candidates_t, dev_number_of_candidates_t, dev_candidates_offsets_t>,
    'p',
    'r',
    'e',
    'f',
    'i',
    'x',
    '_',
    's',
    'u',
    'm',
    '_',
    'o',
    'f',
    'f',
    's',
    'e',
    't',
    's',
    '_',
    'v',
    'e',
    'l',
    'o',
    '_',
    'c',
    'a',
    'n',
    'd',
    'i',
    'd',
    'a',
    't',
    'e',
    's'>,
  velo_estimate_input_size::velo_estimate_input_size_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_cluster_candidates_t,
      dev_event_list_t,
      dev_candidates_offsets_t,
      dev_velo_raw_input_t,
      dev_velo_raw_input_offsets_t,
      dev_estimated_input_size_t,
      dev_module_candidate_num_t,
      dev_cluster_candidates_t>,
    'v',
    'e',
    'l',
    'o',
    '_',
    'e',
    's',
    't',
    'i',
    'm',
    'a',
    't',
    'e',
    '_',
    'i',
    'n',
    'p',
    'u',
    't',
    '_',
    's',
    'i',
    'z',
    'e',
    '_',
    't'>,
  host_prefix_sum::host_prefix_sum_t<
    std::tuple<host_total_number_of_velo_clusters_t, dev_estimated_input_size_t, dev_offsets_estimated_input_size_t>,
    'p',
    'r',
    'e',
    'f',
    'i',
    'x',
    '_',
    's',
    'u',
    'm',
    '_',
    'o',
    'f',
    'f',
    's',
    'e',
    't',
    's',
    '_',
    'e',
    's',
    't',
    'i',
    'm',
    'a',
    't',
    'e',
    'd',
    '_',
    'i',
    'n',
    'p',
    'u',
    't',
    '_',
    's',
    'i',
    'z',
    'e'>,
  velo_masked_clustering::velo_masked_clustering_t<
    std::tuple<
      host_total_number_of_velo_clusters_t,
      host_number_of_selected_events_t,
      dev_velo_raw_input_t,
      dev_velo_raw_input_offsets_t,
      dev_offsets_estimated_input_size_t,
      dev_module_candidate_num_t,
      dev_cluster_candidates_t,
      dev_event_list_t,
      dev_candidates_offsets_t,
      dev_module_cluster_num_t,
      dev_velo_cluster_container_t>,
    'v',
    'e',
    'l',
    'o',
    '_',
    'm',
    'a',
    's',
    'k',
    'e',
    'd',
    '_',
    'c',
    'l',
    'u',
    's',
    't',
    'e',
    'r',
    'i',
    'n',
    'g',
    '_',
    't'>,
  velo_calculate_phi_and_sort::velo_calculate_phi_and_sort_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_total_number_of_velo_clusters_t,
      dev_offsets_estimated_input_size_t,
      dev_module_cluster_num_t,
      dev_velo_cluster_container_t,
      dev_sorted_velo_cluster_container_t,
      dev_hit_permutation_t,
      dev_hit_phi_t>,
    'v',
    'e',
    'l',
    'o',
    '_',
    'c',
    'a',
    'l',
    'c',
    'u',
    'l',
    'a',
    't',
    'e',
    '_',
    'p',
    'h',
    'i',
    '_',
    'a',
    'n',
    'd',
    '_',
    's',
    'o',
    'r',
    't',
    '_',
    't'>,
  velo_fill_candidates::velo_fill_candidates_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_total_number_of_velo_clusters_t,
      dev_sorted_velo_cluster_container_t,
      dev_offsets_estimated_input_size_t,
      dev_module_cluster_num_t,
      dev_hit_phi_t,
      dev_h0_candidates_t,
      dev_h2_candidates_t>,
    'v',
    'e',
    'l',
    'o',
    '_',
    'f',
    'i',
    'l',
    'l',
    '_',
    'c',
    'a',
    'n',
    'd',
    'i',
    'd',
    'a',
    't',
    'e',
    's',
    '_',
    't'>,
  velo_search_by_triplet::velo_search_by_triplet_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_total_number_of_velo_clusters_t,
      dev_sorted_velo_cluster_container_t,
      dev_offsets_estimated_input_size_t,
      dev_module_cluster_num_t,
      dev_h0_candidates_t,
      dev_h2_candidates_t,
      dev_hit_phi_t,
      dev_tracks_t,
      dev_tracklets_t,
      dev_tracks_to_follow_t,
      dev_three_hit_tracks_t,
      dev_hit_used_t,
      dev_atomics_velo_t,
      dev_rel_indices_t,
      dev_number_of_velo_tracks_t>,
    'v',
    'e',
    'l',
    'o',
    '_',
    's',
    'e',
    'a',
    'r',
    'c',
    'h',
    '_',
    'b',
    'y',
    '_',
    't',
    'r',
    'i',
    'p',
    'l',
    'e',
    't',
    '_',
    't'>,
  host_prefix_sum::host_prefix_sum_t<
    std::tuple<host_number_of_velo_tracks_at_least_four_hits_t, dev_number_of_velo_tracks_t, dev_offsets_velo_tracks_t>,
    'p',
    'r',
    'e',
    'f',
    'i',
    'x',
    '_',
    's',
    'u',
    'm',
    '_',
    'o',
    'f',
    'f',
    's',
    'e',
    't',
    's',
    '_',
    'v',
    'e',
    'l',
    'o',
    '_',
    't',
    'r',
    'a',
    'c',
    'k',
    's'>,
  velo_three_hit_tracks_filter::velo_three_hit_tracks_filter_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_sorted_velo_cluster_container_t,
      dev_offsets_estimated_input_size_t,
      dev_three_hit_tracks_t,
      dev_atomics_velo_t,
      dev_hit_used_t,
      dev_three_hit_tracks_output_t,
      dev_number_of_three_hit_tracks_output_t>,
    'v',
    'e',
    'l',
    'o',
    '_',
    't',
    'h',
    'r',
    'e',
    'e',
    '_',
    'h',
    'i',
    't',
    '_',
    't',
    'r',
    'a',
    'c',
    'k',
    's',
    '_',
    'f',
    'i',
    'l',
    't',
    'e',
    'r',
    '_',
    't'>,
  host_prefix_sum::host_prefix_sum_t<
    std::tuple<
      host_number_of_three_hit_tracks_filtered_t,
      dev_number_of_three_hit_tracks_output_t,
      dev_offsets_number_of_three_hit_tracks_filtered_t>,
    'p',
    'r',
    'e',
    'f',
    'i',
    'x',
    '_',
    's',
    'u',
    'm',
    '_',
    'o',
    'f',
    'f',
    's',
    'e',
    't',
    's',
    '_',
    'n',
    'u',
    'm',
    'b',
    'e',
    'r',
    '_',
    'o',
    'f',
    '_',
    't',
    'h',
    'r',
    'e',
    'e',
    '_',
    'h',
    'i',
    't',
    '_',
    't',
    'r',
    'a',
    'c',
    'k',
    's',
    '_',
    'f',
    'i',
    'l',
    't',
    'e',
    'r',
    'e',
    'd'>,
  velo_copy_track_hit_number::velo_copy_track_hit_number_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_velo_tracks_at_least_four_hits_t,
      host_number_of_three_hit_tracks_filtered_t,
      host_number_of_reconstructed_velo_tracks_t,
      dev_tracks_t,
      dev_offsets_velo_tracks_t,
      dev_offsets_number_of_three_hit_tracks_filtered_t,
      dev_velo_track_hit_number_t,
      dev_offsets_all_velo_tracks_t>,
    'v',
    'e',
    'l',
    'o',
    '_',
    'c',
    'o',
    'p',
    'y',
    '_',
    't',
    'r',
    'a',
    'c',
    'k',
    '_',
    'h',
    'i',
    't',
    '_',
    'n',
    'u',
    'm',
    'b',
    'e',
    'r',
    '_',
    't'>,
  host_prefix_sum::host_prefix_sum_t<
    std::tuple<
      host_accumulated_number_of_hits_in_velo_tracks_t,
      dev_velo_track_hit_number_t,
      dev_offsets_velo_track_hit_number_t>,
    'p',
    'r',
    'e',
    'f',
    'i',
    'x',
    '_',
    's',
    'u',
    'm',
    '_',
    'o',
    'f',
    'f',
    's',
    'e',
    't',
    's',
    '_',
    'v',
    'e',
    'l',
    'o',
    '_',
    't',
    'r',
    'a',
    'c',
    'k',
    '_',
    'h',
    'i',
    't',
    '_',
    'n',
    'u',
    'm',
    'b',
    'e',
    'r'>,
  velo_consolidate_tracks::velo_consolidate_tracks_t<
    std::tuple<
      host_accumulated_number_of_hits_in_velo_tracks_t,
      host_number_of_reconstructed_velo_tracks_t,
      host_number_of_three_hit_tracks_filtered_t,
      host_number_of_selected_events_t,
      dev_accepted_velo_tracks_t,
      dev_offsets_all_velo_tracks_t,
      dev_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_sorted_velo_cluster_container_t,
      dev_offsets_estimated_input_size_t,
      dev_velo_states_t,
      dev_three_hit_tracks_output_t,
      dev_offsets_number_of_three_hit_tracks_filtered_t,
      dev_velo_track_hits_t>,
    'v',
    'e',
    'l',
    'o',
    '_',
    'c',
    'o',
    'n',
    's',
    'o',
    'l',
    'i',
    'd',
    'a',
    't',
    'e',
    '_',
    't',
    'r',
    'a',
    'c',
    'k',
    's',
    '_',
    't'>,
  velo_kalman_filter::velo_kalman_filter_t<
    std::tuple<
      host_number_of_reconstructed_velo_tracks_t,
      host_number_of_selected_events_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_velo_track_hits_t,
      dev_velo_states_t,
      dev_velo_kalman_beamline_states_t>,
    'v',
    'e',
    'l',
    'o',
    '_',
    'k',
    'a',
    'l',
    'm',
    'a',
    'n',
    '_',
    'f',
    'i',
    'l',
    't',
    'e',
    'r',
    '_',
    't'>,
  pv_beamline_extrapolate::pv_beamline_extrapolate_t<
    std::tuple<
      host_number_of_reconstructed_velo_tracks_t,
      host_number_of_selected_events_t,
      dev_velo_kalman_beamline_states_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_pvtracks_t,
      dev_pvtrack_z_t>,
    'p',
    'v',
    '_',
    'b',
    'e',
    'a',
    'm',
    'l',
    'i',
    'n',
    'e',
    '_',
    'e',
    'x',
    't',
    'r',
    'a',
    'p',
    'o',
    'l',
    'a',
    't',
    'e',
    '_',
    't'>,
  pv_beamline_histo::pv_beamline_histo_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_pvtracks_t,
      dev_zhisto_t>,
    'p',
    'v',
    '_',
    'b',
    'e',
    'a',
    'm',
    'l',
    'i',
    'n',
    'e',
    '_',
    'h',
    'i',
    's',
    't',
    'o',
    '_',
    't'>,
  pv_beamline_peak::pv_beamline_peak_t<
    std::tuple<host_number_of_selected_events_t, dev_zhisto_t, dev_zpeaks_t, dev_number_of_zpeaks_t>,
    'p',
    'v',
    '_',
    'b',
    'e',
    'a',
    'm',
    'l',
    'i',
    'n',
    'e',
    '_',
    'p',
    'e',
    'a',
    'k',
    '_',
    't'>,
  pv_beamline_calculate_denom::pv_beamline_calculate_denom_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_velo_tracks_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_pvtracks_t,
      dev_pvtracks_denom_t,
      dev_zpeaks_t,
      dev_number_of_zpeaks_t>,
    'p',
    'v',
    '_',
    'b',
    'e',
    'a',
    'm',
    'l',
    'i',
    'n',
    'e',
    '_',
    'c',
    'a',
    'l',
    'c',
    'u',
    'l',
    'a',
    't',
    'e',
    '_',
    'd',
    'e',
    'n',
    'o',
    'm',
    '_',
    't'>,
  pv_beamline_multi_fitter::pv_beamline_multi_fitter_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_velo_tracks_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_pvtracks_t,
      dev_pvtracks_denom_t,
      dev_zpeaks_t,
      dev_number_of_zpeaks_t,
      dev_multi_fit_vertices_t,
      dev_number_of_multi_fit_vertices_t,
      dev_pvtrack_z_t>,
    'p',
    'v',
    '_',
    'b',
    'e',
    'a',
    'm',
    'l',
    'i',
    'n',
    'e',
    '_',
    'm',
    'u',
    'l',
    't',
    'i',
    '_',
    'f',
    'i',
    't',
    't',
    'e',
    'r',
    '_',
    't'>,
  pv_beamline_cleanup::pv_beamline_cleanup_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_multi_fit_vertices_t,
      dev_number_of_multi_fit_vertices_t,
      dev_multi_final_vertices_t,
      dev_number_of_multi_final_vertices_t>,
    'p',
    'v',
    '_',
    'b',
    'e',
    'a',
    'm',
    'l',
    'i',
    'n',
    'e',
    '_',
    'c',
    'l',
    'e',
    'a',
    'n',
    'u',
    'p',
    '_',
    't'>>;
