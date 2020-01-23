from algorithms import *

def VELO_sequence(validate=True):
  host_global_event_cut = host_global_event_cut_t()
  velo_estimate_input_size = velo_estimate_input_size_t()

  prefix_sum_offsets_estimated_input_size = host_prefix_sum_t("prefix_sum_offsets_estimated_input_size",
    host_total_sum_holder_t="host_total_number_of_velo_clusters_t",
    dev_input_buffer_t=velo_estimate_input_size.dev_estimated_input_size_t(),
    dev_output_buffer_t="dev_offsets_estimated_input_size_t")

  velo_masked_clustering = velo_masked_clustering_t()
  velo_calculate_phi_and_sort = velo_calculate_phi_and_sort_t()
  velo_fill_candidates = velo_fill_candidates_t()
  velo_search_by_triplet = velo_search_by_triplet_t()

  prefix_sum_offsets_velo_tracks = host_prefix_sum_t("prefix_sum_offsets_velo_tracks",
    host_total_sum_holder_t="host_number_of_velo_tracks_at_least_four_hits_t",
    dev_input_buffer_t=velo_search_by_triplet.dev_number_of_velo_tracks_t(),
    dev_output_buffer_t="dev_offsets_velo_tracks_t")

  velo_three_hit_tracks_filter = velo_three_hit_tracks_filter_t(
    dev_three_hit_tracks_input_t=velo_search_by_triplet.dev_three_hit_tracks_t())

  prefix_sum_offsets_number_of_three_hit_tracks_filtered = host_prefix_sum_t("prefix_sum_offsets_number_of_three_hit_tracks_filtered",
    host_total_sum_holder_t="host_number_of_three_hit_tracks_filtered_t",
    dev_input_buffer_t=velo_three_hit_tracks_filter.dev_number_of_three_hit_tracks_output_t(),
    dev_output_buffer_t="dev_offsets_number_of_three_hit_tracks_filtered_t")

  velo_copy_track_hit_number = velo_copy_track_hit_number_t()

  prefix_sum_offsets_velo_track_hit_number = host_prefix_sum_t("prefix_sum_offsets_velo_track_hit_number",
    host_total_sum_holder_t="host_accumulated_number_of_hits_in_velo_tracks_t",
    dev_input_buffer_t=velo_copy_track_hit_number.dev_velo_track_hit_number_t(),
    dev_output_buffer_t="dev_offsets_velo_track_hit_number_t")

  velo_consolidate_tracks = velo_consolidate_tracks_t()

  s = Sequence(host_global_event_cut,
    velo_estimate_input_size,
    prefix_sum_offsets_estimated_input_size,
    velo_masked_clustering,
    velo_calculate_phi_and_sort,
    velo_fill_candidates,
    velo_search_by_triplet,
    prefix_sum_offsets_velo_tracks,
    velo_three_hit_tracks_filter,
    prefix_sum_offsets_number_of_three_hit_tracks_filtered,
    velo_copy_track_hit_number,
    prefix_sum_offsets_velo_track_hit_number,
    velo_consolidate_tracks)

  if validate:
    s.validate()

  return s