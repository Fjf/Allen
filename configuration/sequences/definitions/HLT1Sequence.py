###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
###############################################################################
from definitions.algorithms import *


def make_selection_gatherer(lines, initialize_lists, layout_provider,
                            populate_odin_banks, **kwargs):
    return gather_selections_t(
        host_number_of_events_t=initialize_lists.host_number_of_events_t(),
        dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
        dev_input_selections_t=tuple(line.dev_decisions_t() for line in lines),
        dev_input_selections_offsets_t=tuple(
            line.dev_decisions_offsets_t() for line in lines),
        host_input_post_scale_factors_t=tuple(
            line.host_post_scaler_t() for line in lines),
        host_input_post_scale_hashes_t=tuple(
            line.host_post_scaler_hash_t() for line in lines),
        dev_odin_raw_input_t=populate_odin_banks.dev_raw_banks_t(),
        dev_odin_raw_input_offsets_t=populate_odin_banks.dev_raw_offsets_t(),
        names_of_active_lines=",".join([line.name() for line in lines]),
        **kwargs)


def HLT1Sequence(layout_provider,
                 initialize_lists,
                 full_event_list,
                 velo_copy_track_hit_number,
                 velo_kalman_filter,
                 prefix_sum_offsets_velo_track_hit_number,
                 pv_beamline_multi_fitter,
                 prefix_sum_forward_tracks,
                 velo_consolidate_tracks,
                 prefix_sum_ut_tracks,
                 prefix_sum_ut_track_hit_number,
                 ut_consolidate_tracks,
                 prefix_sum_scifi_track_hit_number,
                 scifi_consolidate_tracks,
                 is_muon,
                 add_default_lines=True):

    velo_pv_ip = velo_pv_ip_t(
        host_number_of_reconstructed_velo_tracks_t=velo_copy_track_hit_number.
        host_number_of_reconstructed_velo_tracks_t(),
        host_number_of_events_t=initialize_lists.host_number_of_events_t(),
        dev_velo_kalman_beamline_states_t=velo_kalman_filter.
        dev_velo_kalman_beamline_states_t(),
        dev_offsets_all_velo_tracks_t=velo_copy_track_hit_number.
        dev_offsets_all_velo_tracks_t(),
        dev_offsets_velo_track_hit_number_t=
        prefix_sum_offsets_velo_track_hit_number.dev_output_buffer_t(),
        dev_multi_final_vertices_t=pv_beamline_multi_fitter.
        dev_multi_final_vertices_t(),
        dev_number_of_multi_final_vertices_t=pv_beamline_multi_fitter.
        dev_number_of_multi_final_vertices_t(),
        dev_number_of_events_t=initialize_lists.dev_number_of_events_t(),
        dev_event_list_t=initialize_lists.dev_event_list_t())

    kalman_velo_only = kalman_velo_only_t(
        name="kalman_velo_only",
        host_number_of_events_t=initialize_lists.host_number_of_events_t(),
        host_number_of_reconstructed_scifi_tracks_t=prefix_sum_forward_tracks.
        host_total_sum_holder_t(),
        dev_offsets_all_velo_tracks_t=velo_copy_track_hit_number.
        dev_offsets_all_velo_tracks_t(),
        dev_offsets_velo_track_hit_number_t=
        prefix_sum_offsets_velo_track_hit_number.dev_output_buffer_t(),
        dev_velo_track_hits_t=velo_consolidate_tracks.dev_velo_track_hits_t(),
        dev_offsets_ut_tracks_t=prefix_sum_ut_tracks.dev_output_buffer_t(),
        dev_offsets_ut_track_hit_number_t=prefix_sum_ut_track_hit_number.
        dev_output_buffer_t(),
        dev_ut_qop_t=ut_consolidate_tracks.dev_ut_qop_t(),
        dev_ut_track_velo_indices_t=ut_consolidate_tracks.
        dev_ut_track_velo_indices_t(),
        dev_offsets_forward_tracks_t=prefix_sum_forward_tracks.
        dev_output_buffer_t(),
        dev_offsets_scifi_track_hit_number_t=prefix_sum_scifi_track_hit_number.
        dev_output_buffer_t(),
        dev_scifi_qop_t=scifi_consolidate_tracks.dev_scifi_qop_t(),
        dev_scifi_states_t=scifi_consolidate_tracks.dev_scifi_states_t(),
        dev_scifi_track_ut_indices_t=scifi_consolidate_tracks.
        dev_scifi_track_ut_indices_t(),
        dev_velo_pv_ip_t=velo_pv_ip.dev_velo_pv_ip_t(),
        dev_multi_final_vertices_t=pv_beamline_multi_fitter.
        dev_multi_final_vertices_t(),
        dev_number_of_multi_final_vertices_t=pv_beamline_multi_fitter.
        dev_number_of_multi_final_vertices_t(),
        dev_is_muon_t=is_muon.dev_is_muon_t(),
        dev_number_of_events_t=initialize_lists.dev_number_of_events_t(),
        dev_event_list_t=initialize_lists.dev_event_list_t())

    filter_tracks = filter_tracks_t(
        host_number_of_events_t=initialize_lists.host_number_of_events_t(),
        dev_kf_tracks_t=kalman_velo_only.dev_kf_tracks_t(),
        dev_offsets_forward_tracks_t=prefix_sum_forward_tracks.
        dev_output_buffer_t(),
        dev_offsets_scifi_track_hit_number_t=prefix_sum_scifi_track_hit_number.
        dev_output_buffer_t(),
        dev_scifi_qop_t=scifi_consolidate_tracks.dev_scifi_qop_t(),
        dev_scifi_states_t=scifi_consolidate_tracks.dev_scifi_states_t(),
        dev_scifi_track_ut_indices_t=scifi_consolidate_tracks.
        dev_scifi_track_ut_indices_t(),
        dev_multi_final_vertices_t=pv_beamline_multi_fitter.
        dev_multi_final_vertices_t(),
        dev_number_of_multi_final_vertices_t=pv_beamline_multi_fitter.
        dev_number_of_multi_final_vertices_t(),
        dev_kalman_pv_ipchi2_t=kalman_velo_only.dev_kalman_pv_ipchi2_t(),
        dev_number_of_events_t=initialize_lists.dev_number_of_events_t(),
        dev_event_list_t=initialize_lists.dev_event_list_t())

    prefix_sum_secondary_vertices = host_prefix_sum_t(
        name="prefix_sum_secondary_vertices",
        dev_input_buffer_t=filter_tracks.dev_sv_atomics_t())

    fit_secondary_vertices = fit_secondary_vertices_t(
        name="fit_secondary_vertices",
        host_number_of_events_t=initialize_lists.host_number_of_events_t(),
        host_number_of_svs_t=prefix_sum_secondary_vertices.
        host_total_sum_holder_t(),
        dev_kf_tracks_t=kalman_velo_only.dev_kf_tracks_t(),
        dev_offsets_forward_tracks_t=prefix_sum_forward_tracks.
        dev_output_buffer_t(),
        dev_offsets_scifi_track_hit_number_t=prefix_sum_scifi_track_hit_number.
        dev_output_buffer_t(),
        dev_scifi_qop_t=scifi_consolidate_tracks.dev_scifi_qop_t(),
        dev_scifi_states_t=scifi_consolidate_tracks.dev_scifi_states_t(),
        dev_scifi_track_ut_indices_t=scifi_consolidate_tracks.
        dev_scifi_track_ut_indices_t(),
        dev_multi_final_vertices_t=pv_beamline_multi_fitter.
        dev_multi_final_vertices_t(),
        dev_number_of_multi_final_vertices_t=pv_beamline_multi_fitter.
        dev_number_of_multi_final_vertices_t(),
        dev_kalman_pv_ipchi2_t=kalman_velo_only.dev_kalman_pv_ipchi2_t(),
        dev_svs_trk1_idx_t=filter_tracks.dev_svs_trk1_idx_t(),
        dev_svs_trk2_idx_t=filter_tracks.dev_svs_trk2_idx_t(),
        dev_sv_offsets_t=prefix_sum_secondary_vertices.dev_output_buffer_t(),
        dev_number_of_events_t=initialize_lists.dev_number_of_events_t(),
        dev_event_list_t=initialize_lists.dev_event_list_t())

    two_track_preprocess = two_track_preprocess_t(
        name="two_track_preprocess",
        host_number_of_svs_t=prefix_sum_secondary_vertices.
        host_total_sum_holder_t(),
        dev_consolidated_svs_t=fit_secondary_vertices.dev_consolidated_svs_t(),
        dev_sv_offsets_t=prefix_sum_secondary_vertices.dev_output_buffer_t(),
        dev_offsets_all_velo_tracks_t=velo_copy_track_hit_number.
        dev_offsets_all_velo_tracks_t(),
        dev_offsets_velo_track_hit_number_t=
        prefix_sum_offsets_velo_track_hit_number.dev_output_buffer_t(),
        dev_event_list_t=initialize_lists.dev_event_list_t())

    two_track_evaluator = two_track_evaluator_t(
        name="two_track_evaluator",
        host_number_of_svs_t=prefix_sum_secondary_vertices.
        host_total_sum_holder_t(),
        dev_two_track_catboost_preprocess_output_t=two_track_preprocess.
        dev_two_track_preprocess_output_t())

    odin_banks = data_provider_t(name="odin_banks", bank_type="ODIN")

    hlt1_sequence = Sequence(velo_pv_ip, kalman_velo_only, filter_tracks,
                             prefix_sum_secondary_vertices,
                             fit_secondary_vertices, two_track_preprocess,
                             two_track_evaluator, odin_banks)

    if add_default_lines:
        track_mva_line = track_mva_line_t(
            name="Hlt1TrackMVA",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_reconstructed_scifi_tracks_t=
            prefix_sum_forward_tracks.host_total_sum_holder_t(),
            dev_tracks_t=kalman_velo_only.dev_kf_tracks_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_track_offsets_t=prefix_sum_forward_tracks.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="track_mva_line_pre",
            post_scaler_hash_string="track_mva_line_post")

        two_track_mva_line = two_track_mva_line_t(
            name="Hlt1TwoTrackMVA",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_svs_t=prefix_sum_secondary_vertices.
            host_total_sum_holder_t(),
            dev_svs_t=fit_secondary_vertices.dev_consolidated_svs_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_sv_offsets_t=prefix_sum_secondary_vertices.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="two_track_mva_line_pre",
            post_scaler_hash_string="two_track_mva_line_post")

        two_track_catboost_line = two_track_catboost_line_t(
            name='Hlt1TwoTrackCatBoost',
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_svs_t=prefix_sum_secondary_vertices.
            host_total_sum_holder_t(),
            dev_svs_t=fit_secondary_vertices.dev_consolidated_svs_t(),
            dev_two_track_evaluation_t=two_track_evaluator.
            dev_two_track_catboost_evaluation_t(),
            dev_sv_offsets_t=prefix_sum_secondary_vertices.
            dev_output_buffer_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="two_track_catboost_line_pre",
            post_scaler_hash_string="two_track_catboost_line_post")

        no_beam_line = beam_crossing_line_t(
            name="Hlt1NoBeam",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            dev_event_list_t=full_event_list.dev_event_list_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            beam_crossing_type="0",
            pre_scaler_hash_string="no_beam_line_pre",
            post_scaler_hash_string="no_beam_line_post")

        beam_one_line = beam_crossing_line_t(
            name="Hlt1BeamOne",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            dev_event_list_t=full_event_list.dev_event_list_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            beam_crossing_type="1",
            pre_scaler_hash_string="beam_one_line_pre",
            post_scaler_hash_string="beam_one_line_post")

        beam_two_line = beam_crossing_line_t(
            name="Hlt1BeamTwo",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            dev_event_list_t=full_event_list.dev_event_list_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            beam_crossing_type="2",
            pre_scaler_hash_string="beam_two_line_pre",
            post_scaler_hash_string="beam_two_line_post")

        both_beams_line = beam_crossing_line_t(
            name="Hlt1BothBeams",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            dev_event_list_t=full_event_list.dev_event_list_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            beam_crossing_type="3",
            pre_scaler_hash_string="both_beams_line_pre",
            post_scaler_hash_string="both_beams_line_post")

        velo_micro_bias_line = velo_micro_bias_line_t(
            name="Hlt1VeloMicroBias",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            dev_number_of_events_t=initialize_lists.dev_number_of_events_t(),
            dev_event_list_t=full_event_list.dev_event_list_t(),
            dev_offsets_velo_tracks_t=velo_copy_track_hit_number.
            dev_offsets_all_velo_tracks_t(),
            dev_offsets_velo_track_hit_number_t=
            prefix_sum_offsets_velo_track_hit_number.dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="velo_micro_bias_line_pre",
            post_scaler_hash_string="velo_micro_bias_line_post")

        odin_lumi_line = odin_event_type_line_t(
            name="Hlt1ODINLumi",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            dev_event_list_t=full_event_list.dev_event_list_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            pre_scaler_hash_string="odin_lumi_line_pre",
            post_scaler_hash_string="odin_lumi_line_post",
            odin_event_type=int("0x0008", 0))

        odin_no_bias = odin_event_type_line_t(
            name="Hlt1ODINNoBias",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            dev_event_list_t=full_event_list.dev_event_list_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            pre_scaler_hash_string="odin_no_bias_pre",
            post_scaler_hash_string="odin_no_bias_post",
            odin_event_type=int("0x0004", 0))

        single_high_pt_muon_line = single_high_pt_muon_line_t(
            name="Hlt1SingleHighPtMuon",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_reconstructed_scifi_tracks_t=
            prefix_sum_forward_tracks.host_total_sum_holder_t(),
            dev_tracks_t=kalman_velo_only.dev_kf_tracks_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_track_offsets_t=prefix_sum_forward_tracks.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="single_high_pt_muon_line_pre",
            post_scaler_hash_string="single_high_pt_muon_line_post")

        low_pt_muon_line = low_pt_muon_line_t(
            name="Hlt1LowPtMuon",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_reconstructed_scifi_tracks_t=
            prefix_sum_forward_tracks.host_total_sum_holder_t(),
            dev_tracks_t=kalman_velo_only.dev_kf_tracks_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_track_offsets_t=prefix_sum_forward_tracks.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="low_pt_muon_line_pre",
            post_scaler_hash_string="low_pt_muon_line_post")

        d2kk_line = d2kk_line_t(
            name="Hlt1D2KK",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_svs_t=prefix_sum_secondary_vertices.
            host_total_sum_holder_t(),
            dev_svs_t=fit_secondary_vertices.dev_consolidated_svs_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_sv_offsets_t=prefix_sum_secondary_vertices.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="d2kk_line_pre",
            post_scaler_hash_string="d2kk_line_post")

        d2kpi_line = d2kpi_line_t(
            name="Hlt1D2KPi",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_svs_t=prefix_sum_secondary_vertices.
            host_total_sum_holder_t(),
            dev_svs_t=fit_secondary_vertices.dev_consolidated_svs_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_sv_offsets_t=prefix_sum_secondary_vertices.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="d2kpi_line_pre",
            post_scaler_hash_string="d2kpi_line_post")

        d2pipi_line = d2pipi_line_t(
            name="Hlt1D2PiPi",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_svs_t=prefix_sum_secondary_vertices.
            host_total_sum_holder_t(),
            dev_svs_t=fit_secondary_vertices.dev_consolidated_svs_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_sv_offsets_t=prefix_sum_secondary_vertices.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="d2pipi_line_pre",
            post_scaler_hash_string="d2pipi_line_post")

        di_muon_high_mass_line = di_muon_mass_line_t(
            name="Hlt1DiMuonHighMass",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_svs_t=prefix_sum_secondary_vertices.
            host_total_sum_holder_t(),
            dev_svs_t=fit_secondary_vertices.dev_consolidated_svs_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_sv_offsets_t=prefix_sum_secondary_vertices.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="di_muon_high_mass_line_pre",
            post_scaler_hash_string="di_muon_high_mass_line_post")

        di_muon_low_mass_line = di_muon_mass_line_t(
            name="Hlt1DiMuonLowMass",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_svs_t=prefix_sum_secondary_vertices.
            host_total_sum_holder_t(),
            dev_svs_t=fit_secondary_vertices.dev_consolidated_svs_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_sv_offsets_t=prefix_sum_secondary_vertices.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="di_muon_low_mass_line_pre",
            post_scaler_hash_string="di_muon_low_mass_line_post",
            minHighMassTrackPt="500",
            minHighMassTrackP="3000",
            minMass="0",
            maxDoca="0.2",
            maxVertexChi2="25",
            minIPChi2="4")

        di_muon_soft_line = di_muon_soft_line_t(
            name="Hlt1DiMuonSoft",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_svs_t=prefix_sum_secondary_vertices.
            host_total_sum_holder_t(),
            dev_svs_t=fit_secondary_vertices.dev_consolidated_svs_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_sv_offsets_t=prefix_sum_secondary_vertices.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="di_muon_soft_line_pre",
            post_scaler_hash_string="di_muon_soft_line_post")

        low_pt_di_muon_line = low_pt_di_muon_line_t(
            name="Hlt1LowPtDiMuon",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_svs_t=prefix_sum_secondary_vertices.
            host_total_sum_holder_t(),
            dev_svs_t=fit_secondary_vertices.dev_consolidated_svs_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_sv_offsets_t=prefix_sum_secondary_vertices.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="low_pt_di_muon_line_pre",
            post_scaler_hash_string="low_pt_di_muon_line_post")

        track_muon_mva_line = track_muon_mva_line_t(
            name="Hlt1TrackMuonMVA",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_reconstructed_scifi_tracks_t=
            prefix_sum_forward_tracks.host_total_sum_holder_t(),
            dev_tracks_t=kalman_velo_only.dev_kf_tracks_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_track_offsets_t=prefix_sum_forward_tracks.
            dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="track_muon_mva_line_pre",
            post_scaler_hash_string="track_muon_mva_line_post")

        gec_passthrough_line = passthrough_line_t(
            name="Hlt1GECPassthrough",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            dev_number_of_events_t=initialize_lists.dev_number_of_events_t(),
            dev_event_list_t=initialize_lists.dev_event_list_t(),
            dev_offsets_velo_tracks_t=velo_copy_track_hit_number.
            dev_offsets_all_velo_tracks_t(),
            dev_offsets_velo_track_hit_number_t=
            prefix_sum_offsets_velo_track_hit_number.dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="gec_passthrough_line_pre",
            post_scaler_hash_string="gec_passthrough_line_post")

        passthrough_line = passthrough_line_t(
            name="Hlt1Passthrough",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            dev_number_of_events_t=initialize_lists.dev_number_of_events_t(),
            dev_event_list_t=full_event_list.dev_event_list_t(),
            dev_offsets_velo_tracks_t=velo_copy_track_hit_number.
            dev_offsets_all_velo_tracks_t(),
            dev_offsets_velo_track_hit_number_t=
            prefix_sum_offsets_velo_track_hit_number.dev_output_buffer_t(),
            dev_odin_raw_input_t=odin_banks.dev_raw_banks_t(),
            dev_odin_raw_input_offsets_t=odin_banks.dev_raw_offsets_t(),
            dev_mep_layout_t=layout_provider.dev_mep_layout_t(),
            pre_scaler_hash_string="passthrough_line_pre",
            post_scaler_hash_string="passthrough_line_post")

        lines = (track_mva_line, two_track_mva_line, two_track_catboost_line,
                 no_beam_line, beam_one_line, beam_two_line, both_beams_line,
                 velo_micro_bias_line, odin_lumi_line, odin_no_bias,
                 single_high_pt_muon_line, low_pt_muon_line, d2kk_line,
                 d2kpi_line, d2pipi_line, di_muon_high_mass_line,
                 di_muon_low_mass_line, di_muon_soft_line, low_pt_di_muon_line,
                 track_muon_mva_line, gec_passthrough_line, passthrough_line)
        gatherer = make_selection_gatherer(
            lines,
            initialize_lists,
            layout_provider,
            odin_banks,
            name="gather_selections")

        dec_reporter = dec_reporter_t(
            name="dec_reporter",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_active_lines_t=gatherer.
            host_number_of_active_lines_t(),
            dev_number_of_active_lines_t=gatherer.
            dev_number_of_active_lines_t(),
            dev_selections_t=gatherer.dev_selections_t(),
            dev_selections_offsets_t=gatherer.dev_selections_offsets_t())

        global_decision = global_decision_t(
            name="global_decision",
            host_number_of_events_t=initialize_lists.host_number_of_events_t(),
            host_number_of_active_lines_t=gatherer.
            host_number_of_active_lines_t(),
            dev_number_of_events_t=initialize_lists.dev_number_of_events_t(),
            dev_number_of_active_lines_t=gatherer.
            dev_number_of_active_lines_t(),
            dev_dec_reports_t=dec_reporter.dev_dec_reports_t())

        return extend_sequence(
            extend_sequence(hlt1_sequence, *lines), gatherer, dec_reporter,
            global_decision)

    return hlt1_sequence
