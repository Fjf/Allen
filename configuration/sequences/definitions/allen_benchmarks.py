# Throughput test 12/08/2020
benchmark_weights = dict(
    [
        # Manual entries
        ("velo_banks", 9000.0),
        ("ut_banks", 9100.0),
        ("scifi_banks", 9200.0),
        ("muon_banks", 9300.0),
        ("populate_odin_banks", 9400.0),
        ("initialize_number_of_events", 100.0),
        ("lf_create_tracks", 1827.857),
        ("host_ut_banks", 1000.0),
        ("host_scifi_banks", 1000.0),
        ("global_event_cut", 10.0),
        ("gather_selections", 100.0),
        ("initialize_event_lists", 1.0),
        # Automatic entries
        ("lf_triplet_seeding", 5766.6),
        ("velo_search_by_triplet", 3086.5),
        ("pv_beamline_peak", 1473.3),
        ("muon_add_coords_crossing_maps", 1421.8),
        ("velo_estimate_input_size", 1366.1),
        ("velo_calculate_phi_and_sort", 1197.8),
        ("ut_search_windows", 1068.6),
        ("is_muon", 894.88),
        ("compass_ut", 617.35),
        ("muon_populate_hits", 572.91),
        ("velo_masked_clustering", 543.13),
        ("scifi_direct_decoder", 451.40),
        ("pv_beamline_multi_fitter", 441.71),
        ("lf_quality_filter", 407.44),
        ("fit_secondary_vertices", 375.16),
        ("prepare_decisions", 345.35),
        ("pv_beamline_extrapolate", 344.68),
        ("lf_search_initial_windows", 267.03),
        ("velo_consolidate_tracks", 266.99),
        ("ut_pre_decode", 202.76),
        ("scifi_consolidate_tracks", 202.24),
        ("ut_find_permutation", 185.93),
        ("muon_populate_tile_and_tdc", 181.30),
        ("ut_decode_raw_banks_in_order", 167.47),
        ("velo_calculate_number_of_candidates", 130.07),
        ("lf_quality_filter_length", 130.02),
        ("scifi_pre_decode", 126.53),
        ("muon_calculate_srq_size", 123.97),
        ("ut_calculate_number_of_hits", 117.59),
        ("scifi_raw_bank_decoder", 104.95),
        ("scifi_calculate_cluster_count", 98.427),
        ("filter_tracks", 95.771),
        ("ut_consolidate_tracks", 89.158),
        ("pv_beamline_histo", 88.138),
        ("velo_kalman_filter", 71.307),
        ("kalman_velo_only", 69.355),
        ("ut_select_velo_tracks_with_windows", 68.022),
        ("prepare_raw_banks", 54.877),
        ("ut_copy_track_hit_number", 54.779),
        ("scifi_copy_track_hit_number", 45.863),
        ("velo_pv_ip", 37.416),
        ("velo_copy_track_hit_number", 35.089),
        ("ut_select_velo_tracks", 34.468),
        ("kalman_pv_ipchi2", 32.849),
        ("velo_three_hit_tracks_filter", 31.615),
        ("pv_beamline_calculate_denom", 30.905),
        ("package_sel_reports", 27.382),
        ("pv_beamline_cleanup", 17.844),
    ]
)

# Hardcoded on 29/07/2020
benchmark_efficiencies = dict(
    [
        ("host_global_event_cut_t", 0.9)
    ]
)
