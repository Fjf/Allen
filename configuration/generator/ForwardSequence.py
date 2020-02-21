from algorithms import *


def Forward_sequence():
    scifi_calculate_cluster_count_v4 = scifi_calculate_cluster_count_v4_t()

    prefix_sum_scifi_hits = host_prefix_sum_t(
        "prefix_sum_scifi_hits",
        host_total_sum_holder_t="host_accumulated_number_of_scifi_hits_t",
        dev_input_buffer_t=scifi_calculate_cluster_count_v4.
        dev_scifi_hit_count_t(),
        dev_output_buffer_t="dev_scifi_hit_offsets_t")

    scifi_pre_decode_v4 = scifi_pre_decode_v4_t()
    scifi_raw_bank_decoder_v4 = scifi_raw_bank_decoder_v4_t()
    scifi_direct_decoder_v4 = scifi_direct_decoder_v4_t()

    lf_search_initial_windows = lf_search_initial_windows_t()
    lf_triplet_seeding = lf_triplet_seeding_t()
    lf_triplet_keep_best = lf_triplet_keep_best_t()
    lf_calculate_parametrization = lf_calculate_parametrization_t()
    lf_extend_tracks_x = lf_extend_tracks_x_t()
    lf_extend_tracks_uv = lf_extend_tracks_uv_t()
    lf_quality_filter_length = lf_quality_filter_length_t()
    lf_quality_filter = lf_quality_filter_t()

    prefix_sum_forward_tracks = host_prefix_sum_t(
        "prefix_sum_forward_tracks",
        host_total_sum_holder_t="host_number_of_reconstructed_scifi_tracks_t",
        dev_input_buffer_t=lf_quality_filter.dev_atomics_scifi_t(),
        dev_output_buffer_t="dev_offsets_forward_tracks_t")

    scifi_copy_track_hit_number = scifi_copy_track_hit_number_t()

    prefix_sum_scifi_track_hit_number = host_prefix_sum_t(
        "prefix_sum_scifi_track_hit_number",
        host_total_sum_holder_t=
        "host_accumulated_number_of_hits_in_scifi_tracks_t",
        dev_input_buffer_t=scifi_copy_track_hit_number.
        dev_scifi_track_hit_number_t(),
        dev_output_buffer_t="dev_offsets_scifi_track_hit_number")

    scifi_consolidate_tracks = scifi_consolidate_tracks_t()

    forward_sequence = Sequence(
        scifi_calculate_cluster_count_v4, prefix_sum_scifi_hits,
        scifi_pre_decode_v4, scifi_raw_bank_decoder_v4,
        scifi_direct_decoder_v4, lf_search_initial_windows, lf_triplet_seeding,
        lf_triplet_keep_best, lf_calculate_parametrization, lf_extend_tracks_x,
        lf_extend_tracks_uv, lf_quality_filter_length, lf_quality_filter,
        prefix_sum_forward_tracks, scifi_copy_track_hit_number,
        prefix_sum_scifi_track_hit_number, scifi_consolidate_tracks)

    return forward_sequence
