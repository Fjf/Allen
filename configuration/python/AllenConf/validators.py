###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.algorithms import (
    mc_data_provider_t, host_velo_validator_t, host_velo_ut_validator_t,
    host_forward_validator_t, host_muon_validator_t, host_pv_validator_t,
    host_rate_validator_t, host_kalman_validator_t, host_data_provider_t,
    host_sel_report_validator_t)
from AllenConf.utils import initialize_number_of_events
from AllenConf.persistency import make_dec_reporter, make_gather_selections
from AllenCore.generator import make_algorithm


def mc_data_provider():
    host_mc_particle_banks = make_algorithm(
        host_data_provider_t,
        name="host_mc_particle_banks",
        bank_type="tracks")
    host_mc_pv_banks = make_algorithm(
        host_data_provider_t, name="host_mc_pv_banks", bank_type="PVs")
    number_of_events = initialize_number_of_events()
    return make_algorithm(
        mc_data_provider_t,
        name="mc_data_provider",
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_mc_particle_banks_t=host_mc_particle_banks.host_raw_banks_t,
        host_mc_particle_offsets_t=host_mc_particle_banks.host_raw_offsets_t,
        host_mc_particle_sizes_t=host_mc_particle_banks.host_raw_sizes_t,
        host_mc_pv_banks_t=host_mc_pv_banks.host_raw_banks_t,
        host_mc_pv_offsets_t=host_mc_pv_banks.host_raw_offsets_t,
        host_mc_pv_sizes_t=host_mc_pv_banks.host_raw_sizes_t)


def velo_validation(velo_tracks, name="velo_validator"):
    number_of_events = initialize_number_of_events()
    mc_events = mc_data_provider()

    return make_algorithm(
        host_velo_validator_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_offsets_all_velo_tracks_t=velo_tracks[
            "dev_offsets_all_velo_tracks"],
        dev_offsets_velo_track_hit_number_t=velo_tracks[
            "dev_offsets_velo_track_hit_number"],
        dev_velo_track_hits_t=velo_tracks["dev_velo_track_hits"],
        host_mc_events_t=mc_events.host_mc_events_t)


def veloUT_validation(veloUT_tracks, name="veloUT_validator"):
    mc_events = mc_data_provider()
    number_of_events = initialize_number_of_events()

    velo_tracks = veloUT_tracks["velo_tracks"]
    velo_states = veloUT_tracks["velo_states"]

    return make_algorithm(
        host_velo_ut_validator_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_offsets_all_velo_tracks_t=velo_tracks[
            "dev_offsets_all_velo_tracks"],
        dev_offsets_velo_track_hit_number_t=velo_tracks[
            "dev_offsets_velo_track_hit_number"],
        dev_velo_track_hits_t=velo_tracks["dev_velo_track_hits"],
        host_mc_events_t=mc_events.host_mc_events_t,
        dev_velo_kalman_endvelo_states_t=velo_states[
            "dev_velo_kalman_endvelo_states"],
        dev_offsets_ut_tracks_t=veloUT_tracks["dev_offsets_ut_tracks"],
        dev_offsets_ut_track_hit_number_t=veloUT_tracks[
            "dev_offsets_ut_track_hit_number"],
        dev_ut_track_hits_t=veloUT_tracks["dev_ut_track_hits"],
        dev_ut_track_velo_indices_t=veloUT_tracks["dev_ut_track_velo_indices"],
        dev_ut_qop_t=veloUT_tracks["dev_ut_qop"])


def forward_validation(forward_tracks, name="forward_validator"):
    mc_events = mc_data_provider()
    number_of_events = initialize_number_of_events()

    veloUT_tracks = forward_tracks["veloUT_tracks"]
    velo_tracks = veloUT_tracks["velo_tracks"]
    velo_states = veloUT_tracks["velo_states"]

    return make_algorithm(
        host_forward_validator_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_offsets_all_velo_tracks_t=velo_tracks[
            "dev_offsets_all_velo_tracks"],
        dev_offsets_velo_track_hit_number_t=velo_tracks[
            "dev_offsets_velo_track_hit_number"],
        dev_velo_track_hits_t=velo_tracks["dev_velo_track_hits"],
        host_mc_events_t=mc_events.host_mc_events_t,
        dev_velo_kalman_endvelo_states_t=velo_states[
            "dev_velo_kalman_endvelo_states"],
        dev_offsets_ut_tracks_t=veloUT_tracks["dev_offsets_ut_tracks"],
        dev_offsets_ut_track_hit_number_t=veloUT_tracks[
            "dev_offsets_ut_track_hit_number"],
        dev_ut_track_hits_t=veloUT_tracks["dev_ut_track_hits"],
        dev_ut_track_velo_indices_t=veloUT_tracks["dev_ut_track_velo_indices"],
        dev_ut_qop_t=veloUT_tracks["dev_ut_qop"],
        dev_offsets_forward_tracks_t=forward_tracks[
            "dev_offsets_forward_tracks"],
        dev_offsets_scifi_track_hit_number_t=forward_tracks[
            "dev_offsets_scifi_track_hit_number"],
        dev_scifi_track_hits_t=forward_tracks["dev_scifi_track_hits"],
        dev_scifi_track_ut_indices_t=forward_tracks[
            "dev_scifi_track_ut_indices"],
        dev_scifi_qop_t=forward_tracks["dev_scifi_qop"],
        dev_scifi_states_t=forward_tracks["dev_scifi_states"])


def muon_validation(muonID, name="muon_validator"):
    mc_events = mc_data_provider()
    number_of_events = initialize_number_of_events()

    forward_tracks = muonID["forward_tracks"]
    veloUT_tracks = forward_tracks["veloUT_tracks"]
    velo_tracks = veloUT_tracks["velo_tracks"]
    velo_states = veloUT_tracks["velo_states"]

    return make_algorithm(
        host_muon_validator_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_offsets_all_velo_tracks_t=velo_tracks[
            "dev_offsets_all_velo_tracks"],
        dev_offsets_velo_track_hit_number_t=velo_tracks[
            "dev_offsets_velo_track_hit_number"],
        dev_velo_track_hits_t=velo_tracks["dev_velo_track_hits"],
        host_mc_events_t=mc_events.host_mc_events_t,
        dev_velo_kalman_endvelo_states_t=velo_states[
            "dev_velo_kalman_endvelo_states"],
        dev_offsets_ut_tracks_t=veloUT_tracks["dev_offsets_ut_tracks"],
        dev_offsets_ut_track_hit_number_t=veloUT_tracks[
            "dev_offsets_ut_track_hit_number"],
        dev_ut_track_hits_t=veloUT_tracks["dev_ut_track_hits"],
        dev_ut_track_velo_indices_t=veloUT_tracks["dev_ut_track_velo_indices"],
        dev_ut_qop_t=veloUT_tracks["dev_ut_qop"],
        dev_offsets_forward_tracks_t=forward_tracks[
            "dev_offsets_forward_tracks"],
        dev_offsets_scifi_track_hit_number_t=forward_tracks[
            "dev_offsets_scifi_track_hit_number"],
        dev_scifi_track_hits_t=forward_tracks["dev_scifi_track_hits"],
        dev_scifi_track_ut_indices_t=forward_tracks[
            "dev_scifi_track_ut_indices"],
        dev_scifi_qop_t=forward_tracks["dev_scifi_qop"],
        dev_scifi_states_t=forward_tracks["dev_scifi_states"],
        dev_is_muon_t=muonID["dev_is_muon"])


def pv_validation(pvs, name="pv_validator"):
    mc_events = mc_data_provider()

    return make_algorithm(
        host_pv_validator_t,
        name=name,
        host_mc_events_t=mc_events.host_mc_events_t,
        dev_multi_final_vertices_t=pvs["dev_multi_final_vertices"],
        dev_number_of_multi_final_vertices_t=pvs[
            "dev_number_of_multi_final_vertices"])


def rate_validation(lines, name="rate_validator"):
    number_of_events = initialize_number_of_events()
    dec_reporter = make_dec_reporter(lines)
    gather_selections = make_gather_selections(lines)

    return make_algorithm(
        host_rate_validator_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_names_of_lines_t=gather_selections.host_names_of_active_lines_t,
        host_number_of_active_lines_t=gather_selections.
        host_number_of_active_lines_t,
        host_dec_reports_t=dec_reporter.host_dec_reports_t)


def kalman_validation(kalman_velo_only, name="kalman_validator"):
    number_of_events = initialize_number_of_events()
    mc_events = mc_data_provider()

    forward_tracks = kalman_velo_only["forward_tracks"]
    veloUT_tracks = forward_tracks["veloUT_tracks"]
    velo_tracks = veloUT_tracks["velo_tracks"]
    velo_states = veloUT_tracks["velo_states"]
    pvs = kalman_velo_only["pvs"]

    return make_algorithm(
        host_kalman_validator_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        dev_offsets_all_velo_tracks_t=velo_tracks[
            "dev_offsets_all_velo_tracks"],
        dev_offsets_velo_track_hit_number_t=velo_tracks[
            "dev_offsets_velo_track_hit_number"],
        dev_velo_track_hits_t=velo_tracks["dev_velo_track_hits"],
        host_mc_events_t=mc_events.host_mc_events_t,
        dev_velo_kalman_states_t=velo_states["dev_velo_kalman_endvelo_states"],
        dev_offsets_ut_tracks_t=veloUT_tracks["dev_offsets_ut_tracks"],
        dev_offsets_ut_track_hit_number_t=veloUT_tracks[
            "dev_offsets_ut_track_hit_number"],
        dev_ut_track_hits_t=veloUT_tracks["dev_ut_track_hits"],
        dev_ut_track_velo_indices_t=veloUT_tracks["dev_ut_track_velo_indices"],
        dev_ut_qop_t=veloUT_tracks["dev_ut_qop"],
        dev_offsets_forward_tracks_t=forward_tracks[
            "dev_offsets_forward_tracks"],
        dev_offsets_scifi_track_hit_number_t=forward_tracks[
            "dev_offsets_scifi_track_hit_number"],
        dev_scifi_track_hits_t=forward_tracks["dev_scifi_track_hits"],
        dev_scifi_track_ut_indices_t=forward_tracks[
            "dev_scifi_track_ut_indices"],
        dev_scifi_qop_t=forward_tracks["dev_scifi_qop"],
        dev_scifi_states_t=forward_tracks["dev_scifi_states"],
        dev_kf_tracks_t=kalman_velo_only["dev_kf_tracks"],
        dev_multi_final_vertices_t=pvs["dev_multi_final_vertices"],
        dev_number_of_multi_final_vertices_t=pvs[
            "dev_number_of_multi_final_vertices"])


def selreport_validation(make_selreports,
                         gather_selections,
                         name="selreport_validator"):
    number_of_events = initialize_number_of_events()

    return make_algorithm(
        host_sel_report_validator_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_names_of_lines_t=gather_selections.host_names_of_active_lines_t,
        dev_sel_reports_t=make_selreports["dev_sel_reports"],
        dev_sel_report_offsets_t=make_selreports["dev_selrep_offsets"])
