/**
 * Specify here the algorithms to be executed in the sequence,
 * in the expected order of execution.
 */
SEQUENCE_T(
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
  consolidate_ut_tracks_t
)
