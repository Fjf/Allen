#pragma once

#include "Argument.cuh"
#include "VeloEventModel.cuh"
#include "VeloUTDefinitions.cuh"
#include "PrForward.cuh"
#include "MuonDefinitions.cuh"
#include "patPV_Definitions.cuh"

/**
 * @brief Definition of arguments. All arguments should be defined here,
 *        with their associated type.
 */
ARGUMENT(dev_raw_input, char)
ARGUMENT(dev_raw_input_offsets, uint)
ARGUMENT(dev_estimated_input_size, uint)
ARGUMENT(dev_module_cluster_num, uint)
ARGUMENT(dev_module_candidate_num, uint)
ARGUMENT(dev_cluster_offset, uint)
ARGUMENT(dev_cluster_candidates, uint)
ARGUMENT(dev_velo_cluster_container, uint)
ARGUMENT(dev_tracks, Velo::TrackHits)
ARGUMENT(dev_tracks_to_follow, uint)
ARGUMENT(dev_hit_used, bool)
ARGUMENT(dev_atomics_storage, int)
ARGUMENT(dev_tracklets, Velo::TrackletHits)
ARGUMENT(dev_weak_tracks, Velo::TrackletHits)
ARGUMENT(dev_h0_candidates, short)
ARGUMENT(dev_h2_candidates, short)
ARGUMENT(dev_rel_indices, unsigned short)
ARGUMENT(dev_hit_permutation, uint)
ARGUMENT(dev_velo_track_hit_number, uint)
ARGUMENT(dev_prefix_sum_auxiliary_array_2, uint)
ARGUMENT(dev_velo_track_hits, uint)
ARGUMENT(dev_velo_states, uint)
ARGUMENT(dev_kalmanvelo_states, uint)
ARGUMENT(dev_seeds, PatPV::XYZPoint)
ARGUMENT(dev_number_seeds, uint)
ARGUMENT(dev_vertex, PatPV::Vertex)
ARGUMENT(dev_number_vertex, int)
ARGUMENT(dev_ut_raw_input, uint)
ARGUMENT(dev_ut_raw_input_offsets, uint)
ARGUMENT(dev_ut_hit_offsets, uint)
ARGUMENT(dev_ut_hit_count, uint)
ARGUMENT(dev_prefix_sum_auxiliary_array_3, uint)
ARGUMENT(dev_ut_hits, uint)
ARGUMENT(dev_ut_hit_permutations, uint)
ARGUMENT(dev_veloUT_tracks, VeloUTTracking::TrackUT)
ARGUMENT(dev_atomics_veloUT, int)
ARGUMENT(dev_scifi_raw_input_offsets, uint)
ARGUMENT(dev_scifi_hit_count, uint)
ARGUMENT(dev_prefix_sum_auxiliary_array_4, uint)
ARGUMENT(dev_scifi_hit_permutations, uint)
ARGUMENT(dev_scifi_hits, uint)
ARGUMENT(dev_scifi_raw_input, char)
ARGUMENT(dev_scifi_tracks, SciFi::Track)
ARGUMENT(dev_n_scifi_tracks, uint)
ARGUMENT(dev_muon_track, Muon::State)
ARGUMENT(dev_muon_hits, Muon::HitsSoA)
ARGUMENT(dev_muon_catboost_features, float)

