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
#include "../../cuda/UT/UTDecoding/include/UTCalculateNumberOfHits.cuh"
#include "../../x86/prefix_sum/include/HostPrefixSum.h"
#include "../../cuda/UT/UTDecoding/include/UTPreDecode.cuh"
#include "../../cuda/UT/UTDecoding/include/UTFindPermutation.cuh"
#include "../../cuda/UT/UTDecoding/include/UTDecodeRawBanksInOrder.cuh"
#include "../../cuda/UT/compassUT/include/UTSelectVeloTracks.cuh"
#include "../../cuda/UT/compassUT/include/SearchWindows.cuh"
#include "../../cuda/UT/compassUT/include/UTSelectVeloTracksWithWindows.cuh"
#include "../../cuda/UT/compassUT/include/CompassUT.cuh"
#include "../../x86/prefix_sum/include/HostPrefixSum.h"
#include "../../cuda/UT/consolidate/include/UTCopyTrackHitNumber.cuh"
#include "../../x86/prefix_sum/include/HostPrefixSum.h"
#include "../../cuda/UT/consolidate/include/ConsolidateUT.cuh"
#include "../../cuda/SciFi/preprocessing/include/SciFiCalculateClusterCountV4.cuh"
#include "../../x86/prefix_sum/include/HostPrefixSum.h"
#include "../../cuda/SciFi/preprocessing/include/SciFiPreDecodeV4.cuh"
#include "../../cuda/SciFi/preprocessing/include/SciFiRawBankDecoderV4.cuh"
#include "../../cuda/SciFi/preprocessing/include/SciFiDirectDecoderV4.cuh"
#include "../../cuda/SciFi/looking_forward/search_initial_windows/include/LFSearchInitialWindows.cuh"
#include "../../cuda/SciFi/looking_forward/triplet_seeding/include/LFTripletSeeding.cuh"
#include "../../cuda/SciFi/looking_forward/triplet_keep_best/include/LFTripletKeepBest.cuh"
#include "../../cuda/SciFi/looking_forward/calculate_parametrization/include/LFCalculateParametrization.cuh"
#include "../../cuda/SciFi/looking_forward/extend_tracks_x/include/LFExtendTracksX.cuh"
#include "../../cuda/SciFi/looking_forward/extend_tracks_uv/include/LFExtendTracksUV.cuh"
#include "../../cuda/SciFi/looking_forward/quality_filters/include/LFQualityFilterLength.cuh"
#include "../../cuda/SciFi/looking_forward/quality_filters/include/LFQualityFilter.cuh"
#include "../../x86/prefix_sum/include/HostPrefixSum.h"
#include "../../cuda/SciFi/consolidate/include/SciFiCopyTrackHitNumber.cuh"
#include "../../x86/prefix_sum/include/HostPrefixSum.h"
#include "../../cuda/SciFi/consolidate/include/ConsolidateSciFi.cuh"

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
  : ut_select_velo_tracks::Parameters::host_number_of_selected_events_t,
    velo_consolidate_tracks::Parameters::host_number_of_selected_events_t,
    pv_beamline_calculate_denom::Parameters::host_number_of_selected_events_t,
    velo_three_hit_tracks_filter::Parameters::host_number_of_selected_events_t,
    velo_fill_candidates::Parameters::host_number_of_selected_events_t,
    scifi_raw_bank_decoder_v4::Parameters::host_number_of_selected_events_t,
    lf_extend_tracks_uv::Parameters::host_number_of_selected_events_t,
    velo_masked_clustering::Parameters::host_number_of_selected_events_t,
    velo_search_by_triplet::Parameters::host_number_of_selected_events_t,
    velo_calculate_number_of_candidates::Parameters::host_number_of_selected_events_t,
    host_global_event_cut::Parameters::host_number_of_selected_events_t,
    lf_search_initial_windows::Parameters::host_number_of_selected_events_t,
    velo_estimate_input_size::Parameters::host_number_of_selected_events_t,
    lf_extend_tracks_x::Parameters::host_number_of_selected_events_t,
    compass_ut::Parameters::host_number_of_selected_events_t,
    velo_calculate_phi_and_sort::Parameters::host_number_of_selected_events_t,
    pv_beamline_cleanup::Parameters::host_number_of_selected_events_t,
    lf_calculate_parametrization::Parameters::host_number_of_selected_events_t,
    ut_select_velo_tracks_with_windows::Parameters::host_number_of_selected_events_t,
    ut_decode_raw_banks_in_order::Parameters::host_number_of_selected_events_t,
    ut_calculate_number_of_hits::Parameters::host_number_of_selected_events_t,
    velo_kalman_filter::Parameters::host_number_of_selected_events_t,
    scifi_copy_track_hit_number::Parameters::host_number_of_selected_events_t,
    pv_beamline_peak::Parameters::host_number_of_selected_events_t,
    scifi_pre_decode_v4::Parameters::host_number_of_selected_events_t,
    pv_beamline_extrapolate::Parameters::host_number_of_selected_events_t,
    ut_consolidate_tracks::Parameters::host_number_of_selected_events_t,
    pv_beamline_multi_fitter::Parameters::host_number_of_selected_events_t,
    scifi_direct_decoder_v4::Parameters::host_number_of_selected_events_t,
    pv_beamline_histo::Parameters::host_number_of_selected_events_t,
    scifi_calculate_cluster_count_v4::Parameters::host_number_of_selected_events_t,
    lf_triplet_seeding::Parameters::host_number_of_selected_events_t,
    scifi_consolidate_tracks::Parameters::host_number_of_selected_events_t,
    velo_copy_track_hit_number::Parameters::host_number_of_selected_events_t,
    ut_pre_decode::Parameters::host_number_of_selected_events_t,
    lf_quality_filter::Parameters::host_number_of_selected_events_t,
    ut_copy_track_hit_number::Parameters::host_number_of_selected_events_t,
    ut_find_permutation::Parameters::host_number_of_selected_events_t,
    ut_search_windows::Parameters::host_number_of_selected_events_t,
    lf_triplet_keep_best::Parameters::host_number_of_selected_events_t,
    lf_quality_filter_length::Parameters::host_number_of_selected_events_t {
  constexpr static auto name {"host_number_of_selected_events_t"};
  size_t size;
  char* offset;
};
struct dev_event_list_t : velo_estimate_input_size::Parameters::dev_event_list_t,
                          scifi_direct_decoder_v4::Parameters::dev_event_list_t,
                          velo_masked_clustering::Parameters::dev_event_list_t,
                          ut_decode_raw_banks_in_order::Parameters::dev_event_list_t,
                          scifi_raw_bank_decoder_v4::Parameters::dev_event_list_t,
                          ut_calculate_number_of_hits::Parameters::dev_event_list_t,
                          scifi_pre_decode_v4::Parameters::dev_event_list_t,
                          ut_pre_decode::Parameters::dev_event_list_t,
                          host_global_event_cut::Parameters::dev_event_list_t,
                          velo_calculate_number_of_candidates::Parameters::dev_event_list_t,
                          scifi_calculate_cluster_count_v4::Parameters::dev_event_list_t {
  constexpr static auto name {"dev_event_list_t"};
  size_t size;
  char* offset;
};
struct dev_velo_raw_input_t : velo_masked_clustering::Parameters::dev_velo_raw_input_t,
                              velo_calculate_number_of_candidates::Parameters::dev_velo_raw_input_t,
                              velo_estimate_input_size::Parameters::dev_velo_raw_input_t {
  constexpr static auto name {"dev_velo_raw_input_t"};
  size_t size;
  char* offset;
};
struct dev_velo_raw_input_offsets_t : velo_estimate_input_size::Parameters::dev_velo_raw_input_offsets_t,
                                      velo_calculate_number_of_candidates::Parameters::dev_velo_raw_input_offsets_t,
                                      velo_masked_clustering::Parameters::dev_velo_raw_input_offsets_t {
  constexpr static auto name {"dev_velo_raw_input_offsets_t"};
  size_t size;
  char* offset;
};
struct dev_number_of_candidates_t : host_prefix_sum::Parameters::dev_input_buffer_t,
                                    velo_calculate_number_of_candidates::Parameters::dev_number_of_candidates_t {
  constexpr static auto name {"dev_number_of_candidates_t"};
  size_t size;
  char* offset;
};
struct host_number_of_cluster_candidates_t : host_prefix_sum::Parameters::host_total_sum_holder_t,
                                             velo_estimate_input_size::Parameters::host_number_of_cluster_candidates_t {
  constexpr static auto name {"host_number_of_cluster_candidates_t"};
  size_t size;
  char* offset;
};
struct dev_candidates_offsets_t : velo_estimate_input_size::Parameters::dev_candidates_offsets_t,
                                  host_prefix_sum::Parameters::dev_output_buffer_t,
                                  velo_masked_clustering::Parameters::dev_candidates_offsets_t {
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
struct dev_cluster_candidates_t : velo_masked_clustering::Parameters::dev_cluster_candidates_t,
                                  velo_estimate_input_size::Parameters::dev_cluster_candidates_t {
  constexpr static auto name {"dev_cluster_candidates_t"};
  size_t size;
  char* offset;
};
struct host_total_number_of_velo_clusters_t
  : velo_calculate_phi_and_sort::Parameters::host_total_number_of_velo_clusters_t,
    host_prefix_sum::Parameters::host_total_sum_holder_t,
    velo_search_by_triplet::Parameters::host_total_number_of_velo_clusters_t,
    velo_masked_clustering::Parameters::host_total_number_of_velo_clusters_t,
    velo_fill_candidates::Parameters::host_total_number_of_velo_clusters_t {
  constexpr static auto name {"host_total_number_of_velo_clusters_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_estimated_input_size_t
  : velo_consolidate_tracks::Parameters::dev_offsets_estimated_input_size_t,
    velo_masked_clustering::Parameters::dev_offsets_estimated_input_size_t,
    velo_calculate_phi_and_sort::Parameters::dev_offsets_estimated_input_size_t,
    velo_three_hit_tracks_filter::Parameters::dev_offsets_estimated_input_size_t,
    host_prefix_sum::Parameters::dev_output_buffer_t,
    velo_search_by_triplet::Parameters::dev_offsets_estimated_input_size_t,
    velo_fill_candidates::Parameters::dev_offsets_estimated_input_size_t {
  constexpr static auto name {"dev_offsets_estimated_input_size_t"};
  size_t size;
  char* offset;
};
struct dev_module_cluster_num_t : velo_calculate_phi_and_sort::Parameters::dev_module_cluster_num_t,
                                  velo_fill_candidates::Parameters::dev_module_cluster_num_t,
                                  velo_masked_clustering::Parameters::dev_module_cluster_num_t,
                                  velo_search_by_triplet::Parameters::dev_module_cluster_num_t {
  constexpr static auto name {"dev_module_cluster_num_t"};
  size_t size;
  char* offset;
};
struct dev_velo_cluster_container_t : velo_masked_clustering::Parameters::dev_velo_cluster_container_t,
                                      velo_calculate_phi_and_sort::Parameters::dev_velo_cluster_container_t {
  constexpr static auto name {"dev_velo_cluster_container_t"};
  size_t size;
  char* offset;
};
struct dev_sorted_velo_cluster_container_t
  : velo_search_by_triplet::Parameters::dev_sorted_velo_cluster_container_t,
    velo_three_hit_tracks_filter::Parameters::dev_sorted_velo_cluster_container_t,
    velo_consolidate_tracks::Parameters::dev_sorted_velo_cluster_container_t,
    velo_calculate_phi_and_sort::Parameters::dev_sorted_velo_cluster_container_t,
    velo_fill_candidates::Parameters::dev_sorted_velo_cluster_container_t {
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
                       velo_search_by_triplet::Parameters::dev_hit_phi_t,
                       velo_calculate_phi_and_sort::Parameters::dev_hit_phi_t {
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
struct dev_tracks_t : velo_search_by_triplet::Parameters::dev_tracks_t,
                      velo_consolidate_tracks::Parameters::dev_tracks_t,
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
  : velo_copy_track_hit_number::Parameters::host_number_of_velo_tracks_at_least_four_hits_t,
    host_prefix_sum::Parameters::host_total_sum_holder_t {
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
struct dev_three_hit_tracks_output_t : velo_three_hit_tracks_filter::Parameters::dev_three_hit_tracks_output_t,
                                       velo_consolidate_tracks::Parameters::dev_three_hit_tracks_output_t {
  constexpr static auto name {"dev_three_hit_tracks_output_t"};
  size_t size;
  char* offset;
};
struct dev_number_of_three_hit_tracks_output_t
  : host_prefix_sum::Parameters::dev_input_buffer_t,
    velo_three_hit_tracks_filter::Parameters::dev_number_of_three_hit_tracks_output_t {
  constexpr static auto name {"dev_number_of_three_hit_tracks_output_t"};
  size_t size;
  char* offset;
};
struct host_number_of_three_hit_tracks_filtered_t
  : velo_consolidate_tracks::Parameters::host_number_of_three_hit_tracks_filtered_t,
    host_prefix_sum::Parameters::host_total_sum_holder_t,
    velo_copy_track_hit_number::Parameters::host_number_of_three_hit_tracks_filtered_t {
  constexpr static auto name {"host_number_of_three_hit_tracks_filtered_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_number_of_three_hit_tracks_filtered_t
  : host_prefix_sum::Parameters::dev_output_buffer_t,
    velo_copy_track_hit_number::Parameters::dev_offsets_number_of_three_hit_tracks_filtered_t,
    velo_consolidate_tracks::Parameters::dev_offsets_number_of_three_hit_tracks_filtered_t {
  constexpr static auto name {"dev_offsets_number_of_three_hit_tracks_filtered_t"};
  size_t size;
  char* offset;
};
struct host_number_of_reconstructed_velo_tracks_t
  : velo_copy_track_hit_number::Parameters::host_number_of_reconstructed_velo_tracks_t,
    ut_select_velo_tracks::Parameters::host_number_of_reconstructed_velo_tracks_t,
    velo_consolidate_tracks::Parameters::host_number_of_reconstructed_velo_tracks_t,
    ut_select_velo_tracks_with_windows::Parameters::host_number_of_reconstructed_velo_tracks_t,
    pv_beamline_extrapolate::Parameters::host_number_of_reconstructed_velo_tracks_t,
    pv_beamline_multi_fitter::Parameters::host_number_of_reconstructed_velo_tracks_t,
    velo_kalman_filter::Parameters::host_number_of_reconstructed_velo_tracks_t,
    ut_search_windows::Parameters::host_number_of_reconstructed_velo_tracks_t,
    pv_beamline_calculate_denom::Parameters::host_number_of_reconstructed_velo_tracks_t {
  constexpr static auto name {"host_number_of_reconstructed_velo_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_velo_track_hit_number_t : host_prefix_sum::Parameters::dev_input_buffer_t,
                                     velo_copy_track_hit_number::Parameters::dev_velo_track_hit_number_t {
  constexpr static auto name {"dev_velo_track_hit_number_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_all_velo_tracks_t : velo_kalman_filter::Parameters::dev_offsets_all_velo_tracks_t,
                                       compass_ut::Parameters::dev_offsets_all_velo_tracks_t,
                                       lf_calculate_parametrization::Parameters::dev_offsets_all_velo_tracks_t,
                                       lf_quality_filter::Parameters::dev_offsets_all_velo_tracks_t,
                                       velo_copy_track_hit_number::Parameters::dev_offsets_all_velo_tracks_t,
                                       pv_beamline_extrapolate::Parameters::dev_offsets_all_velo_tracks_t,
                                       pv_beamline_calculate_denom::Parameters::dev_offsets_all_velo_tracks_t,
                                       ut_select_velo_tracks::Parameters::dev_offsets_all_velo_tracks_t,
                                       pv_beamline_multi_fitter::Parameters::dev_offsets_all_velo_tracks_t,
                                       lf_triplet_seeding::Parameters::dev_offsets_all_velo_tracks_t,
                                       lf_search_initial_windows::Parameters::dev_offsets_all_velo_tracks_t,
                                       velo_consolidate_tracks::Parameters::dev_offsets_all_velo_tracks_t,
                                       pv_beamline_histo::Parameters::dev_offsets_all_velo_tracks_t,
                                       ut_select_velo_tracks_with_windows::Parameters::dev_offsets_all_velo_tracks_t,
                                       ut_search_windows::Parameters::dev_offsets_all_velo_tracks_t {
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
  : pv_beamline_histo::Parameters::dev_offsets_velo_track_hit_number_t,
    pv_beamline_multi_fitter::Parameters::dev_offsets_velo_track_hit_number_t,
    ut_search_windows::Parameters::dev_offsets_velo_track_hit_number_t,
    ut_select_velo_tracks_with_windows::Parameters::dev_offsets_velo_track_hit_number_t,
    compass_ut::Parameters::dev_offsets_velo_track_hit_number_t,
    velo_kalman_filter::Parameters::dev_offsets_velo_track_hit_number_t,
    pv_beamline_calculate_denom::Parameters::dev_offsets_velo_track_hit_number_t,
    ut_select_velo_tracks::Parameters::dev_offsets_velo_track_hit_number_t,
    pv_beamline_extrapolate::Parameters::dev_offsets_velo_track_hit_number_t,
    lf_search_initial_windows::Parameters::dev_offsets_velo_track_hit_number_t,
    lf_quality_filter::Parameters::dev_offsets_velo_track_hit_number_t,
    host_prefix_sum::Parameters::dev_output_buffer_t,
    velo_consolidate_tracks::Parameters::dev_offsets_velo_track_hit_number_t,
    lf_calculate_parametrization::Parameters::dev_offsets_velo_track_hit_number_t {
  constexpr static auto name {"dev_offsets_velo_track_hit_number_t"};
  size_t size;
  char* offset;
};
struct dev_accepted_velo_tracks_t : ut_select_velo_tracks::Parameters::dev_accepted_velo_tracks_t,
                                    velo_consolidate_tracks::Parameters::dev_accepted_velo_tracks_t,
                                    ut_select_velo_tracks_with_windows::Parameters::dev_accepted_velo_tracks_t {
  constexpr static auto name {"dev_accepted_velo_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_velo_states_t : velo_consolidate_tracks::Parameters::dev_velo_states_t,
                           lf_quality_filter::Parameters::dev_velo_states_t,
                           lf_search_initial_windows::Parameters::dev_velo_states_t,
                           ut_select_velo_tracks_with_windows::Parameters::dev_velo_states_t,
                           ut_select_velo_tracks::Parameters::dev_velo_states_t,
                           velo_kalman_filter::Parameters::dev_velo_states_t,
                           ut_search_windows::Parameters::dev_velo_states_t,
                           compass_ut::Parameters::dev_velo_states_t,
                           lf_triplet_seeding::Parameters::dev_velo_states_t,
                           lf_calculate_parametrization::Parameters::dev_velo_states_t {
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
struct dev_velo_kalman_beamline_states_t : velo_kalman_filter::Parameters::dev_velo_kalman_beamline_states_t,
                                           pv_beamline_extrapolate::Parameters::dev_velo_kalman_beamline_states_t {
  constexpr static auto name {"dev_velo_kalman_beamline_states_t"};
  size_t size;
  char* offset;
};
struct dev_pvtracks_t : pv_beamline_multi_fitter::Parameters::dev_pvtracks_t,
                        pv_beamline_extrapolate::Parameters::dev_pvtracks_t,
                        pv_beamline_calculate_denom::Parameters::dev_pvtracks_t,
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
struct dev_zpeaks_t : pv_beamline_peak::Parameters::dev_zpeaks_t,
                      pv_beamline_calculate_denom::Parameters::dev_zpeaks_t,
                      pv_beamline_multi_fitter::Parameters::dev_zpeaks_t {
  constexpr static auto name {"dev_zpeaks_t"};
  size_t size;
  char* offset;
};
struct dev_number_of_zpeaks_t : pv_beamline_peak::Parameters::dev_number_of_zpeaks_t,
                                pv_beamline_calculate_denom::Parameters::dev_number_of_zpeaks_t,
                                pv_beamline_multi_fitter::Parameters::dev_number_of_zpeaks_t {
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
struct dev_multi_fit_vertices_t : pv_beamline_multi_fitter::Parameters::dev_multi_fit_vertices_t,
                                  pv_beamline_cleanup::Parameters::dev_multi_fit_vertices_t {
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
struct dev_ut_raw_input_t : ut_decode_raw_banks_in_order::Parameters::dev_ut_raw_input_t,
                            ut_pre_decode::Parameters::dev_ut_raw_input_t,
                            ut_calculate_number_of_hits::Parameters::dev_ut_raw_input_t {
  constexpr static auto name {"dev_ut_raw_input_t"};
  size_t size;
  char* offset;
};
struct dev_ut_raw_input_offsets_t : ut_calculate_number_of_hits::Parameters::dev_ut_raw_input_offsets_t,
                                    ut_pre_decode::Parameters::dev_ut_raw_input_offsets_t,
                                    ut_decode_raw_banks_in_order::Parameters::dev_ut_raw_input_offsets_t {
  constexpr static auto name {"dev_ut_raw_input_offsets_t"};
  size_t size;
  char* offset;
};
struct dev_ut_hit_sizes_t : host_prefix_sum::Parameters::dev_input_buffer_t,
                            ut_calculate_number_of_hits::Parameters::dev_ut_hit_sizes_t {
  constexpr static auto name {"dev_ut_hit_sizes_t"};
  size_t size;
  char* offset;
};
struct host_accumulated_number_of_ut_hits_t : host_prefix_sum::Parameters::host_total_sum_holder_t,
                                              ut_pre_decode::Parameters::host_accumulated_number_of_ut_hits_t,
                                              ut_find_permutation::Parameters::host_accumulated_number_of_ut_hits_t,
                                              ut_consolidate_tracks::Parameters::host_accumulated_number_of_ut_hits_t {
  constexpr static auto name {"host_accumulated_number_of_ut_hits_t"};
  size_t size;
  char* offset;
};
struct dev_ut_hit_offsets_t : ut_pre_decode::Parameters::dev_ut_hit_offsets_t,
                              ut_search_windows::Parameters::dev_ut_hit_offsets_t,
                              compass_ut::Parameters::dev_ut_hit_offsets_t,
                              ut_decode_raw_banks_in_order::Parameters::dev_ut_hit_offsets_t,
                              host_prefix_sum::Parameters::dev_output_buffer_t,
                              ut_find_permutation::Parameters::dev_ut_hit_offsets_t,
                              ut_consolidate_tracks::Parameters::dev_ut_hit_offsets_t {
  constexpr static auto name {"dev_ut_hit_offsets_t"};
  size_t size;
  char* offset;
};
struct dev_ut_hits_t : ut_search_windows::Parameters::dev_ut_hits_t,
                       ut_pre_decode::Parameters::dev_ut_hits_t,
                       ut_find_permutation::Parameters::dev_ut_hits_t,
                       ut_decode_raw_banks_in_order::Parameters::dev_ut_hits_t,
                       ut_consolidate_tracks::Parameters::dev_ut_hits_t,
                       compass_ut::Parameters::dev_ut_hits_t {
  constexpr static auto name {"dev_ut_hits_t"};
  size_t size;
  char* offset;
};
struct dev_ut_hit_count_t : ut_pre_decode::Parameters::dev_ut_hit_count_t {
  constexpr static auto name {"dev_ut_hit_count_t"};
  size_t size;
  char* offset;
};
struct dev_ut_hit_permutations_t : ut_decode_raw_banks_in_order::Parameters::dev_ut_hit_permutations_t,
                                   ut_find_permutation::Parameters::dev_ut_hit_permutations_t {
  constexpr static auto name {"dev_ut_hit_permutations_t"};
  size_t size;
  char* offset;
};
struct dev_ut_number_of_selected_velo_tracks_t
  : ut_select_velo_tracks::Parameters::dev_ut_number_of_selected_velo_tracks_t,
    ut_search_windows::Parameters::dev_ut_number_of_selected_velo_tracks_t,
    ut_select_velo_tracks_with_windows::Parameters::dev_ut_number_of_selected_velo_tracks_t {
  constexpr static auto name {"dev_ut_number_of_selected_velo_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_ut_selected_velo_tracks_t : ut_select_velo_tracks_with_windows::Parameters::dev_ut_selected_velo_tracks_t,
                                       ut_select_velo_tracks::Parameters::dev_ut_selected_velo_tracks_t,
                                       ut_search_windows::Parameters::dev_ut_selected_velo_tracks_t {
  constexpr static auto name {"dev_ut_selected_velo_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_ut_windows_layers_t : compass_ut::Parameters::dev_ut_windows_layers_t,
                                 ut_search_windows::Parameters::dev_ut_windows_layers_t,
                                 ut_select_velo_tracks_with_windows::Parameters::dev_ut_windows_layers_t {
  constexpr static auto name {"dev_ut_windows_layers_t"};
  size_t size;
  char* offset;
};
struct dev_ut_number_of_selected_velo_tracks_with_windows_t
  : compass_ut::Parameters::dev_ut_number_of_selected_velo_tracks_with_windows_t,
    ut_select_velo_tracks_with_windows::Parameters::dev_ut_number_of_selected_velo_tracks_with_windows_t {
  constexpr static auto name {"dev_ut_number_of_selected_velo_tracks_with_windows_t"};
  size_t size;
  char* offset;
};
struct dev_ut_selected_velo_tracks_with_windows_t
  : ut_select_velo_tracks_with_windows::Parameters::dev_ut_selected_velo_tracks_with_windows_t,
    compass_ut::Parameters::dev_ut_selected_velo_tracks_with_windows_t {
  constexpr static auto name {"dev_ut_selected_velo_tracks_with_windows_t"};
  size_t size;
  char* offset;
};
struct dev_ut_tracks_t : compass_ut::Parameters::dev_ut_tracks_t,
                         ut_consolidate_tracks::Parameters::dev_ut_tracks_t,
                         ut_copy_track_hit_number::Parameters::dev_ut_tracks_t {
  constexpr static auto name {"dev_ut_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_atomics_ut_t : host_prefix_sum::Parameters::dev_input_buffer_t, compass_ut::Parameters::dev_atomics_ut_t {
  constexpr static auto name {"dev_atomics_ut_t"};
  size_t size;
  char* offset;
};
struct host_number_of_reconstructed_ut_tracks_t
  : host_prefix_sum::Parameters::host_total_sum_holder_t,
    ut_copy_track_hit_number::Parameters::host_number_of_reconstructed_ut_tracks_t,
    lf_calculate_parametrization::Parameters::host_number_of_reconstructed_ut_tracks_t,
    lf_quality_filter::Parameters::host_number_of_reconstructed_ut_tracks_t,
    lf_triplet_keep_best::Parameters::host_number_of_reconstructed_ut_tracks_t,
    lf_quality_filter_length::Parameters::host_number_of_reconstructed_ut_tracks_t,
    lf_search_initial_windows::Parameters::host_number_of_reconstructed_ut_tracks_t,
    ut_consolidate_tracks::Parameters::host_number_of_reconstructed_ut_tracks_t,
    lf_triplet_seeding::Parameters::host_number_of_reconstructed_ut_tracks_t {
  constexpr static auto name {"host_number_of_reconstructed_ut_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_ut_tracks_t : lf_quality_filter::Parameters::dev_offsets_ut_tracks_t,
                                 scifi_copy_track_hit_number::Parameters::dev_offsets_ut_tracks_t,
                                 lf_search_initial_windows::Parameters::dev_offsets_ut_tracks_t,
                                 ut_copy_track_hit_number::Parameters::dev_offsets_ut_tracks_t,
                                 lf_extend_tracks_uv::Parameters::dev_offsets_ut_tracks_t,
                                 scifi_consolidate_tracks::Parameters::dev_offsets_ut_tracks_t,
                                 lf_triplet_keep_best::Parameters::dev_offsets_ut_tracks_t,
                                 ut_consolidate_tracks::Parameters::dev_offsets_ut_tracks_t,
                                 lf_extend_tracks_x::Parameters::dev_offsets_ut_tracks_t,
                                 lf_quality_filter_length::Parameters::dev_offsets_ut_tracks_t,
                                 host_prefix_sum::Parameters::dev_output_buffer_t,
                                 lf_calculate_parametrization::Parameters::dev_offsets_ut_tracks_t,
                                 lf_triplet_seeding::Parameters::dev_offsets_ut_tracks_t {
  constexpr static auto name {"dev_offsets_ut_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_ut_track_hit_number_t : host_prefix_sum::Parameters::dev_input_buffer_t,
                                   ut_copy_track_hit_number::Parameters::dev_ut_track_hit_number_t {
  constexpr static auto name {"dev_ut_track_hit_number_t"};
  size_t size;
  char* offset;
};
struct host_accumulated_number_of_hits_in_ut_tracks_t
  : ut_consolidate_tracks::Parameters::host_accumulated_number_of_hits_in_ut_tracks_t,
    host_prefix_sum::Parameters::host_total_sum_holder_t {
  constexpr static auto name {"host_accumulated_number_of_hits_in_ut_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_ut_track_hit_number_t : lf_triplet_keep_best::Parameters::dev_offsets_ut_track_hit_number_t,
                                           lf_extend_tracks_uv::Parameters::dev_offsets_ut_track_hit_number_t,
                                           lf_triplet_seeding::Parameters::dev_offsets_ut_track_hit_number_t,
                                           lf_calculate_parametrization::Parameters::dev_offsets_ut_track_hit_number_t,
                                           lf_quality_filter_length::Parameters::dev_offsets_ut_track_hit_number_t,
                                           lf_quality_filter::Parameters::dev_offsets_ut_track_hit_number_t,
                                           lf_search_initial_windows::Parameters::dev_offsets_ut_track_hit_number_t,
                                           scifi_consolidate_tracks::Parameters::dev_offsets_ut_track_hit_number_t,
                                           host_prefix_sum::Parameters::dev_output_buffer_t,
                                           lf_extend_tracks_x::Parameters::dev_offsets_ut_track_hit_number_t,
                                           ut_consolidate_tracks::Parameters::dev_offsets_ut_track_hit_number_t {
  constexpr static auto name {"dev_offsets_ut_track_hit_number_t"};
  size_t size;
  char* offset;
};
struct dev_ut_track_hits_t : ut_consolidate_tracks::Parameters::dev_ut_track_hits_t {
  constexpr static auto name {"dev_ut_track_hits_t"};
  size_t size;
  char* offset;
};
struct dev_ut_qop_t : lf_triplet_seeding::Parameters::dev_ut_qop_t,
                      lf_calculate_parametrization::Parameters::dev_ut_qop_t,
                      lf_search_initial_windows::Parameters::dev_ut_qop_t,
                      ut_consolidate_tracks::Parameters::dev_ut_qop_t {
  constexpr static auto name {"dev_ut_qop_t"};
  size_t size;
  char* offset;
};
struct dev_ut_x_t : lf_search_initial_windows::Parameters::dev_ut_x_t, ut_consolidate_tracks::Parameters::dev_ut_x_t {
  constexpr static auto name {"dev_ut_x_t"};
  size_t size;
  char* offset;
};
struct dev_ut_tx_t : lf_search_initial_windows::Parameters::dev_ut_tx_t,
                     ut_consolidate_tracks::Parameters::dev_ut_tx_t {
  constexpr static auto name {"dev_ut_tx_t"};
  size_t size;
  char* offset;
};
struct dev_ut_z_t : lf_search_initial_windows::Parameters::dev_ut_z_t, ut_consolidate_tracks::Parameters::dev_ut_z_t {
  constexpr static auto name {"dev_ut_z_t"};
  size_t size;
  char* offset;
};
struct dev_ut_track_velo_indices_t : lf_triplet_seeding::Parameters::dev_ut_track_velo_indices_t,
                                     lf_quality_filter::Parameters::dev_ut_track_velo_indices_t,
                                     lf_calculate_parametrization::Parameters::dev_ut_track_velo_indices_t,
                                     ut_consolidate_tracks::Parameters::dev_ut_track_velo_indices_t,
                                     lf_search_initial_windows::Parameters::dev_ut_track_velo_indices_t {
  constexpr static auto name {"dev_ut_track_velo_indices_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_raw_input_t : scifi_raw_bank_decoder_v4::Parameters::dev_scifi_raw_input_t,
                               scifi_direct_decoder_v4::Parameters::dev_scifi_raw_input_t,
                               scifi_pre_decode_v4::Parameters::dev_scifi_raw_input_t,
                               scifi_calculate_cluster_count_v4::Parameters::dev_scifi_raw_input_t {
  constexpr static auto name {"dev_scifi_raw_input_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_raw_input_offsets_t : scifi_calculate_cluster_count_v4::Parameters::dev_scifi_raw_input_offsets_t,
                                       scifi_raw_bank_decoder_v4::Parameters::dev_scifi_raw_input_offsets_t,
                                       scifi_direct_decoder_v4::Parameters::dev_scifi_raw_input_offsets_t,
                                       scifi_pre_decode_v4::Parameters::dev_scifi_raw_input_offsets_t {
  constexpr static auto name {"dev_scifi_raw_input_offsets_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_hit_count_t : host_prefix_sum::Parameters::dev_input_buffer_t,
                               scifi_calculate_cluster_count_v4::Parameters::dev_scifi_hit_count_t {
  constexpr static auto name {"dev_scifi_hit_count_t"};
  size_t size;
  char* offset;
};
struct host_accumulated_number_of_scifi_hits_t
  : host_prefix_sum::Parameters::host_total_sum_holder_t,
    scifi_pre_decode_v4::Parameters::host_accumulated_number_of_scifi_hits_t {
  constexpr static auto name {"host_accumulated_number_of_scifi_hits_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_hit_offsets_t : lf_quality_filter::Parameters::dev_scifi_hit_offsets_t,
                                 scifi_direct_decoder_v4::Parameters::dev_scifi_hit_offsets_t,
                                 scifi_pre_decode_v4::Parameters::dev_scifi_hit_offsets_t,
                                 scifi_raw_bank_decoder_v4::Parameters::dev_scifi_hit_offsets_t,
                                 lf_triplet_seeding::Parameters::dev_scifi_hit_offsets_t,
                                 lf_extend_tracks_uv::Parameters::dev_scifi_hit_offsets_t,
                                 scifi_consolidate_tracks::Parameters::dev_scifi_hit_offsets_t,
                                 lf_search_initial_windows::Parameters::dev_scifi_hit_offsets_t,
                                 lf_calculate_parametrization::Parameters::dev_scifi_hit_offsets_t,
                                 host_prefix_sum::Parameters::dev_output_buffer_t,
                                 lf_extend_tracks_x::Parameters::dev_scifi_hit_offsets_t {
  constexpr static auto name {"dev_scifi_hit_offsets_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_hits_t : lf_quality_filter::Parameters::dev_scifi_hits_t,
                          scifi_consolidate_tracks::Parameters::dev_scifi_hits_t,
                          lf_calculate_parametrization::Parameters::dev_scifi_hits_t,
                          scifi_direct_decoder_v4::Parameters::dev_scifi_hits_t,
                          lf_triplet_seeding::Parameters::dev_scifi_hits_t,
                          lf_extend_tracks_x::Parameters::dev_scifi_hits_t,
                          lf_extend_tracks_uv::Parameters::dev_scifi_hits_t,
                          scifi_raw_bank_decoder_v4::Parameters::dev_scifi_hits_t,
                          scifi_pre_decode_v4::Parameters::dev_scifi_hits_t,
                          lf_search_initial_windows::Parameters::dev_scifi_hits_t {
  constexpr static auto name {"dev_scifi_hits_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_initial_windows_t : lf_extend_tracks_x::Parameters::dev_scifi_lf_initial_windows_t,
                                        lf_extend_tracks_uv::Parameters::dev_scifi_lf_initial_windows_t,
                                        lf_search_initial_windows::Parameters::dev_scifi_lf_initial_windows_t,
                                        lf_triplet_keep_best::Parameters::dev_scifi_lf_initial_windows_t,
                                        lf_triplet_seeding::Parameters::dev_scifi_lf_initial_windows_t {
  constexpr static auto name {"dev_scifi_lf_initial_windows_t"};
  size_t size;
  char* offset;
};
struct dev_ut_states_t : lf_quality_filter::Parameters::dev_ut_states_t,
                         lf_triplet_seeding::Parameters::dev_ut_states_t,
                         lf_extend_tracks_uv::Parameters::dev_ut_states_t,
                         lf_search_initial_windows::Parameters::dev_ut_states_t {
  constexpr static auto name {"dev_ut_states_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_process_track_t : lf_triplet_seeding::Parameters::dev_scifi_lf_process_track_t,
                                      lf_triplet_keep_best::Parameters::dev_scifi_lf_process_track_t,
                                      lf_search_initial_windows::Parameters::dev_scifi_lf_process_track_t {
  constexpr static auto name {"dev_scifi_lf_process_track_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_found_triplets_t : lf_triplet_seeding::Parameters::dev_scifi_lf_found_triplets_t,
                                       lf_triplet_keep_best::Parameters::dev_scifi_lf_found_triplets_t {
  constexpr static auto name {"dev_scifi_lf_found_triplets_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_number_of_found_triplets_t
  : lf_triplet_keep_best::Parameters::dev_scifi_lf_number_of_found_triplets_t,
    lf_triplet_seeding::Parameters::dev_scifi_lf_number_of_found_triplets_t {
  constexpr static auto name {"dev_scifi_lf_number_of_found_triplets_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_tracks_t : lf_triplet_keep_best::Parameters::dev_scifi_lf_tracks_t,
                               lf_calculate_parametrization::Parameters::dev_scifi_lf_tracks_t,
                               lf_quality_filter_length::Parameters::dev_scifi_lf_tracks_t,
                               lf_extend_tracks_uv::Parameters::dev_scifi_lf_tracks_t,
                               lf_extend_tracks_x::Parameters::dev_scifi_lf_tracks_t {
  constexpr static auto name {"dev_scifi_lf_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_atomics_t : lf_extend_tracks_x::Parameters::dev_scifi_lf_atomics_t,
                                lf_triplet_keep_best::Parameters::dev_scifi_lf_atomics_t,
                                lf_quality_filter_length::Parameters::dev_scifi_lf_atomics_t,
                                lf_extend_tracks_uv::Parameters::dev_scifi_lf_atomics_t,
                                lf_calculate_parametrization::Parameters::dev_scifi_lf_atomics_t {
  constexpr static auto name {"dev_scifi_lf_atomics_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_total_number_of_found_triplets_t
  : lf_triplet_keep_best::Parameters::dev_scifi_lf_total_number_of_found_triplets_t {
  constexpr static auto name {"dev_scifi_lf_total_number_of_found_triplets_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_parametrization_t : lf_calculate_parametrization::Parameters::dev_scifi_lf_parametrization_t,
                                        lf_extend_tracks_x::Parameters::dev_scifi_lf_parametrization_t,
                                        lf_quality_filter_length::Parameters::dev_scifi_lf_parametrization_t,
                                        lf_extend_tracks_uv::Parameters::dev_scifi_lf_parametrization_t {
  constexpr static auto name {"dev_scifi_lf_parametrization_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_length_filtered_tracks_t
  : lf_quality_filter::Parameters::dev_scifi_lf_length_filtered_tracks_t,
    lf_quality_filter_length::Parameters::dev_scifi_lf_length_filtered_tracks_t {
  constexpr static auto name {"dev_scifi_lf_length_filtered_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_length_filtered_atomics_t
  : lf_quality_filter::Parameters::dev_scifi_lf_length_filtered_atomics_t,
    lf_quality_filter_length::Parameters::dev_scifi_lf_length_filtered_atomics_t {
  constexpr static auto name {"dev_scifi_lf_length_filtered_atomics_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_parametrization_length_filter_t
  : lf_quality_filter::Parameters::dev_scifi_lf_parametrization_length_filter_t,
    lf_quality_filter_length::Parameters::dev_scifi_lf_parametrization_length_filter_t {
  constexpr static auto name {"dev_scifi_lf_parametrization_length_filter_t"};
  size_t size;
  char* offset;
};
struct dev_atomics_scifi_t : host_prefix_sum::Parameters::dev_input_buffer_t,
                             lf_quality_filter::Parameters::dev_atomics_scifi_t {
  constexpr static auto name {"dev_atomics_scifi_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_tracks_t : scifi_consolidate_tracks::Parameters::dev_scifi_tracks_t,
                            scifi_copy_track_hit_number::Parameters::dev_scifi_tracks_t,
                            lf_quality_filter::Parameters::dev_scifi_tracks_t {
  constexpr static auto name {"dev_scifi_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_y_parametrization_length_filter_t
  : lf_quality_filter::Parameters::dev_scifi_lf_y_parametrization_length_filter_t {
  constexpr static auto name {"dev_scifi_lf_y_parametrization_length_filter_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_lf_parametrization_consolidate_t
  : lf_quality_filter::Parameters::dev_scifi_lf_parametrization_consolidate_t,
    scifi_consolidate_tracks::Parameters::dev_scifi_lf_parametrization_consolidate_t {
  constexpr static auto name {"dev_scifi_lf_parametrization_consolidate_t"};
  size_t size;
  char* offset;
};
struct host_number_of_reconstructed_scifi_tracks_t
  : scifi_copy_track_hit_number::Parameters::host_number_of_reconstructed_scifi_tracks_t,
    host_prefix_sum::Parameters::host_total_sum_holder_t,
    scifi_consolidate_tracks::Parameters::host_number_of_reconstructed_scifi_tracks_t {
  constexpr static auto name {"host_number_of_reconstructed_scifi_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_forward_tracks_t : host_prefix_sum::Parameters::dev_output_buffer_t,
                                      scifi_consolidate_tracks::Parameters::dev_offsets_forward_tracks_t,
                                      scifi_copy_track_hit_number::Parameters::dev_offsets_forward_tracks_t {
  constexpr static auto name {"dev_offsets_forward_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_track_hit_number_t : scifi_copy_track_hit_number::Parameters::dev_scifi_track_hit_number_t,
                                      host_prefix_sum::Parameters::dev_input_buffer_t {
  constexpr static auto name {"dev_scifi_track_hit_number_t"};
  size_t size;
  char* offset;
};
struct host_accumulated_number_of_hits_in_scifi_tracks_t
  : scifi_consolidate_tracks::Parameters::host_accumulated_number_of_hits_in_scifi_tracks_t,
    host_prefix_sum::Parameters::host_total_sum_holder_t {
  constexpr static auto name {"host_accumulated_number_of_hits_in_scifi_tracks_t"};
  size_t size;
  char* offset;
};
struct dev_offsets_scifi_track_hit_number : host_prefix_sum::Parameters::dev_output_buffer_t,
                                            scifi_consolidate_tracks::Parameters::dev_offsets_scifi_track_hit_number {
  constexpr static auto name {"dev_offsets_scifi_track_hit_number"};
  size_t size;
  char* offset;
};
struct dev_scifi_track_hits_t : scifi_consolidate_tracks::Parameters::dev_scifi_track_hits_t {
  constexpr static auto name {"dev_scifi_track_hits_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_qop_t : scifi_consolidate_tracks::Parameters::dev_scifi_qop_t {
  constexpr static auto name {"dev_scifi_qop_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_states_t : scifi_consolidate_tracks::Parameters::dev_scifi_states_t {
  constexpr static auto name {"dev_scifi_states_t"};
  size_t size;
  char* offset;
};
struct dev_scifi_track_ut_indices_t : scifi_consolidate_tracks::Parameters::dev_scifi_track_ut_indices_t {
  constexpr static auto name {"dev_scifi_track_ut_indices_t"};
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
    't'>,
  ut_calculate_number_of_hits::ut_calculate_number_of_hits_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_event_list_t,
      dev_ut_raw_input_t,
      dev_ut_raw_input_offsets_t,
      dev_ut_hit_sizes_t>,
    'u',
    't',
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
    'h',
    'i',
    't',
    's',
    '_',
    't'>,
  host_prefix_sum::host_prefix_sum_t<
    std::tuple<host_accumulated_number_of_ut_hits_t, dev_ut_hit_sizes_t, dev_ut_hit_offsets_t>,
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
    'u',
    't',
    '_',
    'h',
    'i',
    't',
    's'>,
  ut_pre_decode::ut_pre_decode_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_accumulated_number_of_ut_hits_t,
      dev_ut_raw_input_t,
      dev_ut_raw_input_offsets_t,
      dev_event_list_t,
      dev_ut_hit_offsets_t,
      dev_ut_hits_t,
      dev_ut_hit_count_t>,
    'u',
    't',
    '_',
    'p',
    'r',
    'e',
    '_',
    'd',
    'e',
    'c',
    'o',
    'd',
    'e',
    '_',
    't'>,
  ut_find_permutation::ut_find_permutation_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_accumulated_number_of_ut_hits_t,
      dev_ut_hits_t,
      dev_ut_hit_offsets_t,
      dev_ut_hit_permutations_t>,
    'u',
    't',
    '_',
    'f',
    'i',
    'n',
    'd',
    '_',
    'p',
    'e',
    'r',
    'm',
    'u',
    't',
    'a',
    't',
    'i',
    'o',
    'n',
    '_',
    't'>,
  ut_decode_raw_banks_in_order::ut_decode_raw_banks_in_order_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_ut_raw_input_t,
      dev_ut_raw_input_offsets_t,
      dev_event_list_t,
      dev_ut_hit_offsets_t,
      dev_ut_hits_t,
      dev_ut_hit_permutations_t>,
    'u',
    't',
    '_',
    'd',
    'e',
    'c',
    'o',
    'd',
    'e',
    '_',
    'r',
    'a',
    'w',
    '_',
    'b',
    'a',
    'n',
    'k',
    's',
    '_',
    'i',
    'n',
    '_',
    'o',
    'r',
    'd',
    'e',
    'r',
    '_',
    't'>,
  ut_select_velo_tracks::ut_select_velo_tracks_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_velo_tracks_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_velo_states_t,
      dev_accepted_velo_tracks_t,
      dev_ut_number_of_selected_velo_tracks_t,
      dev_ut_selected_velo_tracks_t>,
    'u',
    't',
    '_',
    's',
    'e',
    'l',
    'e',
    'c',
    't',
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
    's',
    '_',
    't'>,
  ut_search_windows::ut_search_windows_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_velo_tracks_t,
      dev_ut_hits_t,
      dev_ut_hit_offsets_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_velo_states_t,
      dev_ut_number_of_selected_velo_tracks_t,
      dev_ut_selected_velo_tracks_t,
      dev_ut_windows_layers_t>,
    'u',
    't',
    '_',
    's',
    'e',
    'a',
    'r',
    'c',
    'h',
    '_',
    'w',
    'i',
    'n',
    'd',
    'o',
    'w',
    's',
    '_',
    't'>,
  ut_select_velo_tracks_with_windows::ut_select_velo_tracks_with_windows_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_velo_tracks_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_velo_states_t,
      dev_accepted_velo_tracks_t,
      dev_ut_number_of_selected_velo_tracks_t,
      dev_ut_selected_velo_tracks_t,
      dev_ut_windows_layers_t,
      dev_ut_number_of_selected_velo_tracks_with_windows_t,
      dev_ut_selected_velo_tracks_with_windows_t>,
    'u',
    't',
    '_',
    's',
    'e',
    'l',
    'e',
    'c',
    't',
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
    's',
    '_',
    'w',
    'i',
    't',
    'h',
    '_',
    'w',
    'i',
    'n',
    'd',
    'o',
    'w',
    's',
    '_',
    't'>,
  compass_ut::compass_ut_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_ut_hits_t,
      dev_ut_hit_offsets_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_velo_states_t,
      dev_ut_tracks_t,
      dev_atomics_ut_t,
      dev_ut_windows_layers_t,
      dev_ut_number_of_selected_velo_tracks_with_windows_t,
      dev_ut_selected_velo_tracks_with_windows_t>,
    'c',
    'o',
    'm',
    'p',
    'a',
    's',
    's',
    '_',
    'u',
    't',
    '_',
    't'>,
  host_prefix_sum::host_prefix_sum_t<
    std::tuple<host_number_of_reconstructed_ut_tracks_t, dev_atomics_ut_t, dev_offsets_ut_tracks_t>,
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
    'u',
    't',
    '_',
    't',
    'r',
    'a',
    'c',
    'k',
    's'>,
  ut_copy_track_hit_number::ut_copy_track_hit_number_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_ut_tracks_t,
      dev_ut_tracks_t,
      dev_offsets_ut_tracks_t,
      dev_ut_track_hit_number_t>,
    'u',
    't',
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
      host_accumulated_number_of_hits_in_ut_tracks_t,
      dev_ut_track_hit_number_t,
      dev_offsets_ut_track_hit_number_t>,
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
    'u',
    't',
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
  ut_consolidate_tracks::ut_consolidate_tracks_t<
    std::tuple<
      host_accumulated_number_of_ut_hits_t,
      host_number_of_reconstructed_ut_tracks_t,
      host_number_of_selected_events_t,
      host_accumulated_number_of_hits_in_ut_tracks_t,
      dev_ut_hits_t,
      dev_ut_hit_offsets_t,
      dev_ut_track_hits_t,
      dev_offsets_ut_tracks_t,
      dev_offsets_ut_track_hit_number_t,
      dev_ut_qop_t,
      dev_ut_x_t,
      dev_ut_tx_t,
      dev_ut_z_t,
      dev_ut_track_velo_indices_t,
      dev_ut_tracks_t>,
    'u',
    't',
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
  scifi_calculate_cluster_count_v4::scifi_calculate_cluster_count_v4_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_scifi_raw_input_t,
      dev_scifi_raw_input_offsets_t,
      dev_scifi_hit_count_t,
      dev_event_list_t>,
    's',
    'c',
    'i',
    'f',
    'i',
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
    'c',
    'l',
    'u',
    's',
    't',
    'e',
    'r',
    '_',
    'c',
    'o',
    'u',
    'n',
    't',
    '_',
    'v',
    '4',
    '_',
    't'>,
  host_prefix_sum::host_prefix_sum_t<
    std::tuple<host_accumulated_number_of_scifi_hits_t, dev_scifi_hit_count_t, dev_scifi_hit_offsets_t>,
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
    's',
    'c',
    'i',
    'f',
    'i',
    '_',
    'h',
    'i',
    't',
    's'>,
  scifi_pre_decode_v4::scifi_pre_decode_v4_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_accumulated_number_of_scifi_hits_t,
      dev_scifi_raw_input_t,
      dev_scifi_raw_input_offsets_t,
      dev_event_list_t,
      dev_scifi_hit_offsets_t,
      dev_scifi_hits_t>,
    's',
    'c',
    'i',
    'f',
    'i',
    '_',
    'p',
    'r',
    'e',
    '_',
    'd',
    'e',
    'c',
    'o',
    'd',
    'e',
    '_',
    'v',
    '4',
    '_',
    't'>,
  scifi_raw_bank_decoder_v4::scifi_raw_bank_decoder_v4_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_scifi_raw_input_t,
      dev_scifi_raw_input_offsets_t,
      dev_scifi_hit_offsets_t,
      dev_scifi_hits_t,
      dev_event_list_t>,
    's',
    'c',
    'i',
    'f',
    'i',
    '_',
    'r',
    'a',
    'w',
    '_',
    'b',
    'a',
    'n',
    'k',
    '_',
    'd',
    'e',
    'c',
    'o',
    'd',
    'e',
    'r',
    '_',
    'v',
    '4',
    '_',
    't'>,
  scifi_direct_decoder_v4::scifi_direct_decoder_v4_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_scifi_raw_input_t,
      dev_scifi_raw_input_offsets_t,
      dev_scifi_hit_offsets_t,
      dev_scifi_hits_t,
      dev_event_list_t>,
    's',
    'c',
    'i',
    'f',
    'i',
    '_',
    'd',
    'i',
    'r',
    'e',
    'c',
    't',
    '_',
    'd',
    'e',
    'c',
    'o',
    'd',
    'e',
    'r',
    '_',
    'v',
    '4',
    '_',
    't'>,
  lf_search_initial_windows::lf_search_initial_windows_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_ut_tracks_t,
      dev_scifi_hits_t,
      dev_scifi_hit_offsets_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_velo_states_t,
      dev_offsets_ut_tracks_t,
      dev_offsets_ut_track_hit_number_t,
      dev_ut_x_t,
      dev_ut_tx_t,
      dev_ut_z_t,
      dev_ut_qop_t,
      dev_ut_track_velo_indices_t,
      dev_scifi_lf_initial_windows_t,
      dev_ut_states_t,
      dev_scifi_lf_process_track_t>,
    'l',
    'f',
    '_',
    's',
    'e',
    'a',
    'r',
    'c',
    'h',
    '_',
    'i',
    'n',
    'i',
    't',
    'i',
    'a',
    'l',
    '_',
    'w',
    'i',
    'n',
    'd',
    'o',
    'w',
    's',
    '_',
    't'>,
  lf_triplet_seeding::lf_triplet_seeding_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_ut_tracks_t,
      dev_scifi_hits_t,
      dev_scifi_hit_offsets_t,
      dev_offsets_all_velo_tracks_t,
      dev_velo_states_t,
      dev_offsets_ut_tracks_t,
      dev_offsets_ut_track_hit_number_t,
      dev_ut_track_velo_indices_t,
      dev_ut_qop_t,
      dev_scifi_lf_initial_windows_t,
      dev_ut_states_t,
      dev_scifi_lf_process_track_t,
      dev_scifi_lf_found_triplets_t,
      dev_scifi_lf_number_of_found_triplets_t>,
    'l',
    'f',
    '_',
    't',
    'r',
    'i',
    'p',
    'l',
    'e',
    't',
    '_',
    's',
    'e',
    'e',
    'd',
    'i',
    'n',
    'g',
    '_',
    't'>,
  lf_triplet_keep_best::lf_triplet_keep_best_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_ut_tracks_t,
      dev_offsets_ut_tracks_t,
      dev_offsets_ut_track_hit_number_t,
      dev_scifi_lf_tracks_t,
      dev_scifi_lf_atomics_t,
      dev_scifi_lf_initial_windows_t,
      dev_scifi_lf_process_track_t,
      dev_scifi_lf_found_triplets_t,
      dev_scifi_lf_number_of_found_triplets_t,
      dev_scifi_lf_total_number_of_found_triplets_t>,
    'l',
    'f',
    '_',
    't',
    'r',
    'i',
    'p',
    'l',
    'e',
    't',
    '_',
    'k',
    'e',
    'e',
    'p',
    '_',
    'b',
    'e',
    's',
    't',
    '_',
    't'>,
  lf_calculate_parametrization::lf_calculate_parametrization_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_ut_tracks_t,
      dev_scifi_hits_t,
      dev_scifi_hit_offsets_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_velo_states_t,
      dev_offsets_ut_tracks_t,
      dev_offsets_ut_track_hit_number_t,
      dev_ut_track_velo_indices_t,
      dev_ut_qop_t,
      dev_scifi_lf_tracks_t,
      dev_scifi_lf_atomics_t,
      dev_scifi_lf_parametrization_t>,
    'l',
    'f',
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
    'a',
    'r',
    'a',
    'm',
    'e',
    't',
    'r',
    'i',
    'z',
    'a',
    't',
    'i',
    'o',
    'n',
    '_',
    't'>,
  lf_extend_tracks_x::lf_extend_tracks_x_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_scifi_hits_t,
      dev_scifi_hit_offsets_t,
      dev_offsets_ut_tracks_t,
      dev_offsets_ut_track_hit_number_t,
      dev_scifi_lf_tracks_t,
      dev_scifi_lf_atomics_t,
      dev_scifi_lf_initial_windows_t,
      dev_scifi_lf_parametrization_t>,
    'l',
    'f',
    '_',
    'e',
    'x',
    't',
    'e',
    'n',
    'd',
    '_',
    't',
    'r',
    'a',
    'c',
    'k',
    's',
    '_',
    'x',
    '_',
    't'>,
  lf_extend_tracks_uv::lf_extend_tracks_uv_t<
    std::tuple<
      host_number_of_selected_events_t,
      dev_scifi_hits_t,
      dev_scifi_hit_offsets_t,
      dev_offsets_ut_tracks_t,
      dev_offsets_ut_track_hit_number_t,
      dev_scifi_lf_tracks_t,
      dev_scifi_lf_atomics_t,
      dev_ut_states_t,
      dev_scifi_lf_initial_windows_t,
      dev_scifi_lf_parametrization_t>,
    'l',
    'f',
    '_',
    'e',
    'x',
    't',
    'e',
    'n',
    'd',
    '_',
    't',
    'r',
    'a',
    'c',
    'k',
    's',
    '_',
    'u',
    'v',
    '_',
    't'>,
  lf_quality_filter_length::lf_quality_filter_length_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_ut_tracks_t,
      dev_offsets_ut_tracks_t,
      dev_offsets_ut_track_hit_number_t,
      dev_scifi_lf_tracks_t,
      dev_scifi_lf_atomics_t,
      dev_scifi_lf_length_filtered_tracks_t,
      dev_scifi_lf_length_filtered_atomics_t,
      dev_scifi_lf_parametrization_t,
      dev_scifi_lf_parametrization_length_filter_t>,
    'l',
    'f',
    '_',
    'q',
    'u',
    'a',
    'l',
    'i',
    't',
    'y',
    '_',
    'f',
    'i',
    'l',
    't',
    'e',
    'r',
    '_',
    'l',
    'e',
    'n',
    'g',
    't',
    'h',
    '_',
    't'>,
  lf_quality_filter::lf_quality_filter_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_ut_tracks_t,
      dev_scifi_hits_t,
      dev_scifi_hit_offsets_t,
      dev_offsets_ut_tracks_t,
      dev_offsets_ut_track_hit_number_t,
      dev_scifi_lf_length_filtered_tracks_t,
      dev_scifi_lf_length_filtered_atomics_t,
      dev_atomics_scifi_t,
      dev_scifi_tracks_t,
      dev_scifi_lf_parametrization_length_filter_t,
      dev_scifi_lf_y_parametrization_length_filter_t,
      dev_scifi_lf_parametrization_consolidate_t,
      dev_ut_states_t,
      dev_velo_states_t,
      dev_offsets_all_velo_tracks_t,
      dev_offsets_velo_track_hit_number_t,
      dev_ut_track_velo_indices_t>,
    'l',
    'f',
    '_',
    'q',
    'u',
    'a',
    'l',
    'i',
    't',
    'y',
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
    std::tuple<host_number_of_reconstructed_scifi_tracks_t, dev_atomics_scifi_t, dev_offsets_forward_tracks_t>,
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
    'f',
    'o',
    'r',
    'w',
    'a',
    'r',
    'd',
    '_',
    't',
    'r',
    'a',
    'c',
    'k',
    's'>,
  scifi_copy_track_hit_number::scifi_copy_track_hit_number_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_number_of_reconstructed_scifi_tracks_t,
      dev_offsets_ut_tracks_t,
      dev_scifi_tracks_t,
      dev_offsets_forward_tracks_t,
      dev_scifi_track_hit_number_t>,
    's',
    'c',
    'i',
    'f',
    'i',
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
      host_accumulated_number_of_hits_in_scifi_tracks_t,
      dev_scifi_track_hit_number_t,
      dev_offsets_scifi_track_hit_number>,
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
    's',
    'c',
    'i',
    'f',
    'i',
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
  scifi_consolidate_tracks::scifi_consolidate_tracks_t<
    std::tuple<
      host_number_of_selected_events_t,
      host_accumulated_number_of_hits_in_scifi_tracks_t,
      host_number_of_reconstructed_scifi_tracks_t,
      dev_scifi_hits_t,
      dev_scifi_hit_offsets_t,
      dev_scifi_track_hits_t,
      dev_offsets_forward_tracks_t,
      dev_offsets_scifi_track_hit_number,
      dev_scifi_qop_t,
      dev_scifi_states_t,
      dev_scifi_track_ut_indices_t,
      dev_offsets_ut_tracks_t,
      dev_offsets_ut_track_hit_number_t,
      dev_scifi_tracks_t,
      dev_scifi_lf_parametrization_consolidate_t>,
    's',
    'c',
    'i',
    'f',
    'i',
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
    't'>>;
