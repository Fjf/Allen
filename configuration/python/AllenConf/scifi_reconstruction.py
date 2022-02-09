###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.algorithms import (
    data_provider_t, host_prefix_sum_t, scifi_calculate_cluster_count_t,
    scifi_pre_decode_t, scifi_raw_bank_decoder_t, lf_search_initial_windows_t,
    lf_triplet_seeding_t, lf_create_tracks_t, lf_quality_filter_length_t,
    lf_quality_filter_t, scifi_copy_track_hit_number_t,
    scifi_consolidate_tracks_t)
from AllenConf.utils import initialize_number_of_events
from AllenCore.generator import make_algorithm
from PyConf.tonic import configurable


@configurable
def decode_scifi():
    number_of_events = initialize_number_of_events()
    scifi_banks = make_algorithm(
        data_provider_t, name="scifi_banks", bank_type="FTCluster")

    scifi_calculate_cluster_count = make_algorithm(
        scifi_calculate_cluster_count_t,
        name="scifi_calculate_cluster_count",
        dev_scifi_raw_input_t=scifi_banks.dev_raw_banks_t,
        dev_scifi_raw_input_offsets_t=scifi_banks.dev_raw_offsets_t,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_raw_bank_version_t=scifi_banks.host_raw_bank_version_t)

    prefix_sum_scifi_hits = make_algorithm(
        host_prefix_sum_t,
        name="prefix_sum_scifi_hits",
        dev_input_buffer_t=scifi_calculate_cluster_count.dev_scifi_hit_count_t)

    scifi_pre_decode = make_algorithm(
        scifi_pre_decode_t,
        name="scifi_pre_decode",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_accumulated_number_of_scifi_hits_t=prefix_sum_scifi_hits.
        host_total_sum_holder_t,
        dev_scifi_raw_input_t=scifi_banks.dev_raw_banks_t,
        dev_scifi_raw_input_offsets_t=scifi_banks.dev_raw_offsets_t,
        dev_scifi_hit_offsets_t=prefix_sum_scifi_hits.dev_output_buffer_t,
        host_raw_bank_version_t=scifi_banks.host_raw_bank_version_t)

    scifi_raw_bank_decoder = make_algorithm(
        scifi_raw_bank_decoder_t,
        name="scifi_raw_bank_decoder",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        host_accumulated_number_of_scifi_hits_t=prefix_sum_scifi_hits.
        host_total_sum_holder_t,
        dev_scifi_raw_input_t=scifi_banks.dev_raw_banks_t,
        dev_scifi_raw_input_offsets_t=scifi_banks.dev_raw_offsets_t,
        dev_scifi_hit_offsets_t=prefix_sum_scifi_hits.dev_output_buffer_t,
        dev_cluster_references_t=scifi_pre_decode.dev_cluster_references_t,
        host_raw_bank_version_t=scifi_banks.host_raw_bank_version_t)

    return {
        "dev_scifi_hits": scifi_raw_bank_decoder.dev_scifi_hits_t,
        "dev_scifi_hit_offsets": prefix_sum_scifi_hits.dev_output_buffer_t
    }


@configurable
def make_forward_tracks(decoded_scifi, ut_tracks, hit_window_size="32"):
    number_of_events = initialize_number_of_events()

    velo_tracks = ut_tracks["velo_tracks"]
    host_number_of_reconstructed_velo_tracks = velo_tracks[
        "host_number_of_reconstructed_velo_tracks"]
    dev_offsets_all_velo_tracks = velo_tracks["dev_offsets_all_velo_tracks"]
    dev_offsets_velo_track_hit_number = velo_tracks[
        "dev_offsets_velo_track_hit_number"]
    dev_accepted_velo_tracks = velo_tracks["dev_accepted_velo_tracks"]

    host_number_of_reconstructed_ut_tracks = ut_tracks[
        "host_number_of_reconstructed_ut_tracks"]
    dev_offsets_ut_tracks = ut_tracks["dev_offsets_ut_tracks"]
    velo_states = ut_tracks["velo_states"]

    lf_search_initial_windows = make_algorithm(
        lf_search_initial_windows_t,
        name="lf_search_initial_windows",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        host_number_of_reconstructed_ut_tracks_t=
        host_number_of_reconstructed_ut_tracks,
        dev_scifi_hits_t=decoded_scifi["dev_scifi_hits"],
        dev_scifi_hit_offsets_t=decoded_scifi["dev_scifi_hit_offsets"],
        dev_velo_states_view_t=velo_states[
            "dev_velo_kalman_endvelo_states_view"],
        dev_ut_tracks_view_t=ut_tracks["dev_ut_tracks_view"],
        hit_window_size=hit_window_size,
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"])

    lf_triplet_seeding = make_algorithm(
        lf_triplet_seeding_t,
        name="lf_triplet_seeding",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        host_number_of_reconstructed_ut_tracks_t=
        host_number_of_reconstructed_ut_tracks,
        dev_scifi_hits_t=decoded_scifi["dev_scifi_hits"],
        dev_scifi_hit_offsets_t=decoded_scifi["dev_scifi_hit_offsets"],
        dev_offsets_all_velo_tracks_t=dev_offsets_all_velo_tracks,
        dev_velo_states_view_t=velo_states[
            "dev_velo_kalman_endvelo_states_view"],
        dev_ut_tracks_view_t=ut_tracks["dev_ut_tracks_view"],
        dev_scifi_lf_initial_windows_t=lf_search_initial_windows.
        dev_scifi_lf_initial_windows_t,
        dev_ut_states_t=lf_search_initial_windows.dev_ut_states_t,
        dev_scifi_lf_process_track_t=lf_search_initial_windows.
        dev_scifi_lf_process_track_t,
        hit_window_size=hit_window_size)

    lf_create_tracks = make_algorithm(
        lf_create_tracks_t,
        name="lf_create_tracks",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        host_number_of_reconstructed_ut_tracks_t=
        host_number_of_reconstructed_ut_tracks,
        dev_scifi_hits_t=decoded_scifi["dev_scifi_hits"],
        dev_scifi_hit_offsets_t=decoded_scifi["dev_scifi_hit_offsets"],
        dev_velo_tracks_view_t=velo_tracks["dev_velo_tracks_view"],
        dev_velo_states_view_t=velo_states[
            "dev_velo_kalman_endvelo_states_view"],
        dev_ut_tracks_view_t=ut_tracks["dev_ut_tracks_view"],
        dev_scifi_lf_initial_windows_t=lf_search_initial_windows.
        dev_scifi_lf_initial_windows_t,
        dev_ut_states_t=lf_search_initial_windows.dev_ut_states_t,
        dev_scifi_lf_process_track_t=lf_search_initial_windows.
        dev_scifi_lf_process_track_t,
        dev_scifi_lf_found_triplets_t=lf_triplet_seeding.
        dev_scifi_lf_found_triplets_t,
        dev_scifi_lf_number_of_found_triplets_t=lf_triplet_seeding.
        dev_scifi_lf_number_of_found_triplets_t)

    lf_quality_filter_length = make_algorithm(
        lf_quality_filter_length_t,
        name="lf_quality_filter_length",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        host_number_of_reconstructed_ut_tracks_t=
        host_number_of_reconstructed_ut_tracks,
        dev_ut_tracks_view_t=ut_tracks["dev_ut_tracks_view"],
        dev_scifi_lf_tracks_t=lf_create_tracks.dev_scifi_lf_tracks_t,
        dev_scifi_lf_atomics_t=lf_create_tracks.dev_scifi_lf_atomics_t,
        dev_scifi_lf_parametrization_t=lf_create_tracks.
        dev_scifi_lf_parametrization_t)

    lf_quality_filter = make_algorithm(
        lf_quality_filter_t,
        name="lf_quality_filter",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        host_number_of_reconstructed_ut_tracks_t=
        host_number_of_reconstructed_ut_tracks,
        dev_scifi_hits_t=decoded_scifi["dev_scifi_hits"],
        dev_scifi_hit_offsets_t=decoded_scifi["dev_scifi_hit_offsets"],
        dev_ut_tracks_view_t=ut_tracks["dev_ut_tracks_view"],
        dev_scifi_lf_length_filtered_atomics_t=lf_quality_filter_length.
        dev_scifi_lf_length_filtered_atomics_t,
        dev_scifi_lf_length_filtered_tracks_t=lf_quality_filter_length.
        dev_scifi_lf_length_filtered_tracks_t,
        dev_scifi_lf_parametrization_length_filter_t=lf_quality_filter_length.
        dev_scifi_lf_parametrization_length_filter_t,
        dev_ut_states_t=lf_search_initial_windows.dev_ut_states_t)

    prefix_sum_forward_tracks = make_algorithm(
        host_prefix_sum_t,
        name="prefix_sum_forward_tracks",
        dev_input_buffer_t=lf_quality_filter.dev_atomics_scifi_t)

    scifi_copy_track_hit_number = make_algorithm(
        scifi_copy_track_hit_number_t,
        name="scifi_copy_track_hit_number",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_reconstructed_scifi_tracks_t=prefix_sum_forward_tracks.
        host_total_sum_holder_t,
        dev_offsets_ut_tracks_t=dev_offsets_ut_tracks,
        dev_scifi_tracks_t=lf_quality_filter.dev_scifi_tracks_t,
        dev_offsets_forward_tracks_t=prefix_sum_forward_tracks.
        dev_output_buffer_t)

    prefix_sum_scifi_track_hit_number = make_algorithm(
        host_prefix_sum_t,
        name="prefix_sum_scifi_track_hit_number",
        dev_input_buffer_t=scifi_copy_track_hit_number.
        dev_scifi_track_hit_number_t)

    scifi_consolidate_tracks = make_algorithm(
        scifi_consolidate_tracks_t,
        name="scifi_consolidate_tracks",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_number_of_events_t=number_of_events["dev_number_of_events"],
        host_accumulated_number_of_hits_in_scifi_tracks_t=
        prefix_sum_scifi_track_hit_number.host_total_sum_holder_t,
        host_number_of_reconstructed_scifi_tracks_t=prefix_sum_forward_tracks.
        host_total_sum_holder_t,
        dev_scifi_hits_t=decoded_scifi["dev_scifi_hits"],
        dev_scifi_hit_offsets_t=decoded_scifi["dev_scifi_hit_offsets"],
        dev_offsets_forward_tracks_t=prefix_sum_forward_tracks.
        dev_output_buffer_t,
        dev_offsets_scifi_track_hit_number_t=prefix_sum_scifi_track_hit_number.
        dev_output_buffer_t,
        dev_scifi_tracks_t=lf_quality_filter.dev_scifi_tracks_t,
        dev_scifi_lf_parametrization_consolidate_t=lf_quality_filter.
        dev_scifi_lf_parametrization_consolidate_t,
        dev_ut_tracks_view_t=ut_tracks["dev_ut_tracks_view"],
        dev_velo_states_view_t=velo_states[
            "dev_velo_kalman_endvelo_states_view"])

    return {
        "veloUT_tracks":
        ut_tracks,
        "dev_scifi_track_hits":
        scifi_consolidate_tracks.dev_scifi_track_hits_t,
        "dev_scifi_qop":
        scifi_consolidate_tracks.dev_scifi_qop_t,
        "dev_scifi_states":
        scifi_consolidate_tracks.dev_scifi_states_t,
        "dev_scifi_track_ut_indices":
        scifi_consolidate_tracks.dev_scifi_track_ut_indices_t,
        "host_number_of_reconstructed_scifi_tracks":
        prefix_sum_forward_tracks.host_total_sum_holder_t,
        "dev_offsets_forward_tracks":
        prefix_sum_forward_tracks.dev_output_buffer_t,
        "dev_offsets_scifi_track_hit_number":
        prefix_sum_scifi_track_hit_number.dev_output_buffer_t
    }


def forward_tracking():
    from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
    from AllenConf.ut_reconstruction import decode_ut, make_ut_tracks

    decoded_velo = decode_velo()
    velo_tracks = make_velo_tracks(decoded_velo)
    decoded_ut = decode_ut()
    ut_tracks = make_ut_tracks(decoded_ut, velo_tracks)
    decoded_scifi = decode_scifi()
    forward_tracks = make_forward_tracks(decoded_scifi, ut_tracks)
    alg = forward_tracks["dev_scifi_track_hits"].producer
    return alg
