from algorithms import *
from MuonSequence import Muon_sequence

def HLT1_sequence(validate=False):
  kalman_velo_only = kalman_velo_only_t()
  kalman_pv_ipchi2 = kalman_pv_ipchi2_t()
  
  filter_tracks = filter_tracks_t()
  fit_secondary_vertices = fit_secondary_vertices_t()
  prefix_sum_secondary_vertices = host_prefix_sum_t("prefix_sum_secondary_vertices",
    host_total_sum_holder_t="host_number_of_svs_t",
    dev_input_buffer_t=filter_tracks.dev_sv_atomics_t(),
    dev_output_buffer_t="dev_sv_offsets_t")

  muon_filter = muon_filter_t()
  match_upstream_muon = match_upstream_muon_t()
  
  prefix_sum_mf_tracks = host_prefix_sum_t("prefix_sum_mf_tracks",
    host_total_sum_holder_t="host_number_of_mf_tracks_t",
    dev_input_buffer_t=muon_filter.dev_mf_track_atomics_t(),
    dev_output_buffer_t="dev_mf_track_offsets_t")

  package_mf_tracks = package_mf_tracks_t()
  filter_mf_tracks = filter_mf_tracks_t()
  
  prefix_sum_mf_vertices = host_prefix_sum_t("prefix_sum_mf_vertices",
    host_total_sum_holder_t="host_number_of_mf_svs_t",
    dev_input_buffer_t=filter_mf_tracks.dev_mf_sv_atomics_t(),
    dev_output_buffer_t="dev_mf_sv_offsets_t")

  fit_mf_vertices = fit_mf_vertices_t()
  
  run_hlt1 = run_hlt1_t()
  run_postscale = run_postscale_t()
  prepare_decisions = prepare_decisions_t()
  prepare_raw_banks = prepare_raw_banks_t()

  prefix_sum_sel_reps = host_prefix_sum_t("prefix_sum_sel_reps",
    host_total_sum_holder_t="host_number_of_sel_rep_words_t",
    dev_input_buffer_t=prepare_raw_banks.dev_sel_rep_sizes_t(),
    dev_output_buffer_t="dev_sel_rep_offsets_t")

  package_sel_reports = package_sel_reports_t()

  PassThrough_line = PassThrough_t()
  OneTrackMVA_line = OneTrackMVA_t()
  SingleMuon_line = SingleMuon_t()
  TwoTrackMVA_line = TwoTrackMVA_t()
  HighMassDiMuon_line = HighMassDiMuon_t()
  DisplacedDiMuon_line = DisplacedDiMuon_t()
  DiMuonSoft_line = DiMuonSoft_t()
  DiMuonTrackEff_line = DiMuonTrackEff_t() 
  D2KPi_line = D2KPi_t()
  D2PiPi_line = D2PiPi_t()
  D2KK_line = D2KK_t()

  muon_sequence = Muon_sequence()
  hlt1_sequence = extend_sequence(muon_sequence,
    kalman_velo_only,
    kalman_pv_ipchi2,
    filter_tracks,
    prefix_sum_secondary_vertices,
    muon_filter,
    match_upstream_muon,
    prefix_sum_mf_tracks,
    package_mf_tracks,
    filter_mf_tracks,
    prefix_sum_mf_vertices,
    fit_secondary_vertices,
    fit_mf_vertices,
    run_hlt1,
    run_postscale,
    prepare_decisions,
    prepare_raw_banks,
    prefix_sum_sel_reps,
    package_sel_reports,
    PassThrough_line,
    OneTrackMVA_line,
    SingleMuon_line,
    TwoTrackMVA_line,
    HighMassDiMuon_line,
    DisplacedDiMuon_line,
    DiMuonSoft_line,
    DiMuonTrackEff_line,
    D2KPi_line,
    D2PiPi_line,
    D2KK_line)

  if validate:
    hlt1_sequence.validate()

  return hlt1_sequence
