from algorithms import *
from MuonSequence import Muon_sequence

def HLT1_sequence(validate=False):
  kalman_velo_only = kalman_velo_only_t()
  kalman_pv_ipchi2 = kalman_pv_ipchi2_t()
  fit_secondary_vertices = fit_secondary_vertices_t()
  
  prefix_sum_secondary_vertices = host_prefix_sum_t("prefix_sum_secondary_vertices",
    host_total_sum_holder_t="host_number_of_svs_t",
    dev_input_buffer_t=fit_secondary_vertices.dev_sv_atomics_t(),
    dev_output_buffer_t="dev_sv_offsets_t")

  consolidate_svs = consolidate_svs_t()
  run_hlt1 = run_hlt1_t()
  run_postscale = run_postscale_t()
  prepare_decisions = prepare_decisions_t()
  prepare_raw_banks = prepare_raw_banks_t()

  prefix_sum_sel_reps = host_prefix_sum_t("prefix_sum_sel_reps",
    host_total_sum_holder_t="host_number_of_sel_rep_words_t",
    dev_input_buffer_t=prepare_raw_banks.dev_sel_rep_sizes_t(),
    dev_output_buffer_t="dev_sel_rep_offsets_t")

  package_sel_reports = package_sel_reports_t()

  muon_sequence = Muon_sequence()
  hlt1_sequence = extend_sequence(muon_sequence,
    kalman_velo_only,
    kalman_pv_ipchi2,
    fit_secondary_vertices,
    prefix_sum_secondary_vertices,
    consolidate_svs,
    run_hlt1,
    run_postscale,
    prepare_decisions,
    prepare_raw_banks,
    prefix_sum_sel_reps,
    package_sel_reports)

  if validate:
    hlt1_sequence.validate()

  return hlt1_sequence
