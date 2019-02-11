/**
 * Specify here the algorithms to be executed in the sequence,
 * in the expected order of execution.
 */
SEQUENCE_T(
  init_event_list_t,
  global_event_cut_t,
  velo_estimate_input_size_t,
  prefix_sum_velo_clusters_t,
  velo_masked_clustering_t,
  velo_calculate_phi_and_sort_t,
  velo_fill_candidates_t,
  velo_search_by_triplet_t,
  velo_weak_tracks_adder_t,
  copy_and_prefix_sum_single_block_velo_t,
  copy_velo_track_hit_number_t,
  prefix_sum_velo_track_hit_number_t,
  consolidate_velo_tracks_t,
  velo_kalman_fit_t,
  pv_beamline_extrapolate_t,
  pv_beamline_histo_t,
  pv_beamline_peak_t,
  pv_beamline_multi_fitter_t,
  ut_calculate_number_of_hits_t,
  prefix_sum_ut_hits_t,
  ut_pre_decode_t,
  ut_find_permutation_t,
  ut_decode_raw_banks_in_order_t,
  ut_search_windows_t,
  compass_ut_t,
  copy_and_prefix_sum_single_block_ut_t,
  copy_ut_track_hit_number_t,
  prefix_sum_ut_track_hit_number_t,
  consolidate_ut_tracks_t,
  scifi_calculate_cluster_count_v4_t,
  prefix_sum_scifi_hits_t,
  scifi_pre_decode_v4_t,
  scifi_raw_bank_decoder_v4_t,
  scifi_direct_decoder_v4_t,
  // cpu_scifi_pr_forward_t,
  cpu_scifi_momentum_forward_t,
  // copy_and_prefix_sum_single_block_scifi_t,
  // copy_scifi_track_hit_number_t,
  // prefix_sum_scifi_track_hit_number_t,
  // consolidate_scifi_tracks_t
)
