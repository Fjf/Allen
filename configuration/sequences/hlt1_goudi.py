from definitions.VeloSequence import VeloSequence
from definitions.PVSequence import PVSequence
from definitions.UTSequence import UTSequence
from definitions.ForwardSequence import ForwardSequence
from definitions.MuonSequence import MuonSequence
from definitions.HLT1Sequence import HLT1Sequence, make_selection_gatherer
from definitions import algorithms

velo_sequence = VeloSequence(doGEC=True)

pv_sequence = PVSequence(
    initialize_lists=velo_sequence["initialize_lists"],
    velo_copy_track_hit_number=velo_sequence["velo_copy_track_hit_number"],
    velo_consolidate_tracks=velo_sequence["velo_consolidate_tracks"],
    prefix_sum_offsets_velo_track_hit_number=velo_sequence[
        "prefix_sum_offsets_velo_track_hit_number"])

ut_sequence = UTSequence(
    initialize_lists=velo_sequence["initialize_lists"],
    velo_copy_track_hit_number=velo_sequence["velo_copy_track_hit_number"],
    velo_consolidate_tracks=velo_sequence["velo_consolidate_tracks"],
    prefix_sum_offsets_velo_track_hit_number=velo_sequence[
        "prefix_sum_offsets_velo_track_hit_number"])

forward_sequence = ForwardSequence(
    initialize_lists=velo_sequence["initialize_lists"],
    velo_copy_track_hit_number=velo_sequence["velo_copy_track_hit_number"],
    velo_consolidate_tracks=velo_sequence["velo_consolidate_tracks"],
    prefix_sum_offsets_velo_track_hit_number=velo_sequence[
        "prefix_sum_offsets_velo_track_hit_number"],
    prefix_sum_ut_tracks=ut_sequence["prefix_sum_ut_tracks"],
    prefix_sum_ut_track_hit_number=ut_sequence[
        "prefix_sum_ut_track_hit_number"],
    ut_consolidate_tracks=ut_sequence["ut_consolidate_tracks"])

muon_sequence = MuonSequence(
    initialize_lists=velo_sequence["initialize_lists"],
    prefix_sum_forward_tracks=forward_sequence["prefix_sum_forward_tracks"],
    prefix_sum_scifi_track_hit_number=forward_sequence[
        "prefix_sum_scifi_track_hit_number"],
    scifi_consolidate_tracks_t=forward_sequence["scifi_consolidate_tracks_t"])

hlt1_sequence = HLT1Sequence(
    initialize_lists=velo_sequence["initialize_lists"],
    velo_copy_track_hit_number=velo_sequence["velo_copy_track_hit_number"],
    velo_kalman_filter=pv_sequence["velo_kalman_filter"],
    prefix_sum_offsets_velo_track_hit_number=velo_sequence[
        "prefix_sum_offsets_velo_track_hit_number"],
    pv_beamline_multi_fitter=pv_sequence["pv_beamline_multi_fitter"],
    prefix_sum_forward_tracks=forward_sequence["prefix_sum_forward_tracks"],
    velo_consolidate_tracks=velo_sequence["velo_consolidate_tracks"],
    prefix_sum_ut_tracks=ut_sequence["prefix_sum_ut_tracks"],
    prefix_sum_ut_track_hit_number=ut_sequence[
        "prefix_sum_ut_track_hit_number"],
    ut_consolidate_tracks=ut_sequence["ut_consolidate_tracks"],
    prefix_sum_scifi_track_hit_number=forward_sequence[
        "prefix_sum_scifi_track_hit_number"],
    scifi_consolidate_tracks=forward_sequence["scifi_consolidate_tracks_t"],
    is_muon=muon_sequence["is_muon_t"],
    add_default_lines=False)

track_mva_line = algorithms.track_mva_line_t(
    name = "track_mva_line",
    host_number_of_events_t=velo_sequence["initialize_lists"].host_number_of_events_t(),
    host_number_of_reconstructed_scifi_tracks_t=forward_sequence["prefix_sum_forward_tracks"].host_total_sum_holder_t(),
    dev_tracks_t=hlt1_sequence["kalman_velo_only"].dev_kf_tracks_t(),
    dev_event_list_t=velo_sequence["initialize_lists"].dev_event_list_t(),
    dev_track_offsets_t=forward_sequence["prefix_sum_forward_tracks"].dev_output_buffer_t())

two_track_mva_line = algorithms.two_track_mva_line_t(
    name = "two_track_mva_line",
    host_number_of_events_t=velo_sequence["initialize_lists"].host_number_of_events_t(),
    host_number_of_svs_t=hlt1_sequence["prefix_sum_secondary_vertices"].host_total_sum_holder_t(),
    dev_svs_t=hlt1_sequence["fit_secondary_vertices"].dev_consolidated_svs_t(),
    dev_event_list_t=velo_sequence["initialize_lists"].dev_event_list_t(),
    dev_sv_offsets_t=hlt1_sequence["prefix_sum_secondary_vertices"].dev_output_buffer_t())

no_beam_line = algorithms.beam_crossing_line_t(
    name="no_beam_line",
    host_number_of_events_t=velo_sequence["initialize_lists"].host_number_of_events_t(),
    dev_event_list_t=velo_sequence["full_event_list"].dev_event_list_t(),
    dev_odin_raw_input_t=hlt1_sequence["odin_banks"].dev_raw_banks_t(),
    dev_odin_raw_input_offsets_t=hlt1_sequence["odin_banks"].dev_raw_offsets_t(),
    beam_crossing_type="0")

beam_one_line = algorithms.beam_crossing_line_t(
    name="beam_one_line",
    host_number_of_events_t=velo_sequence["initialize_lists"].host_number_of_events_t(),
    dev_event_list_t=velo_sequence["full_event_list"].dev_event_list_t(),
    dev_odin_raw_input_t=hlt1_sequence["odin_banks"].dev_raw_banks_t(),
    dev_odin_raw_input_offsets_t=hlt1_sequence["odin_banks"].dev_raw_offsets_t(),
    beam_crossing_type="1")

beam_two_line = algorithms.beam_crossing_line_t(
    name="beam_two_line",
    host_number_of_events_t=velo_sequence["initialize_lists"].host_number_of_events_t(),
    dev_event_list_t=velo_sequence["full_event_list"].dev_event_list_t(),
    dev_odin_raw_input_t=hlt1_sequence["odin_banks"].dev_raw_banks_t(),
    dev_odin_raw_input_offsets_t=hlt1_sequence["odin_banks"].dev_raw_offsets_t(),
    beam_crossing_type="2")

lines = (track_mva_line, two_track_mva_line, no_beam_line)
gatherer = make_selection_gatherer(lines, velo_sequence["initialize_lists"], hlt1_sequence["odin_banks"], name="gather_selections")

algorithms.extend_sequence(algorithms.compose_sequences(velo_sequence, pv_sequence, ut_sequence, forward_sequence,
                  muon_sequence, hlt1_sequence), *lines, gatherer).generate()
