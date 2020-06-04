/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <tuple>
#include "ConfiguredInputAggregates.h"
#include "../../stream/gear/include/ArgumentManager.cuh"
#include "../../host/data_provider/include/HostDataProvider.h"
#include "../../host/data_provider/include/HostDataProvider.h"
#include "../../host/global_event_cut/include/HostGlobalEventCut.h"
#include "../../host/init_event_list/include/HostInitEventList.h"
#include "../../host/data_provider/include/DataProvider.h"
#include "../../device/velo/mask_clustering/include/VeloCalculateNumberOfCandidates.cuh"
#include "../../host/prefix_sum/include/HostPrefixSum.h"
#include "../../device/velo/mask_clustering/include/EstimateInputSize.cuh"
#include "../../host/prefix_sum/include/HostPrefixSum.h"
#include "../../device/velo/mask_clustering/include/MaskedVeloClustering.cuh"
#include "../../device/velo/calculate_phi_and_sort/include/CalculatePhiAndSort.cuh"
#include "../../device/velo/search_by_triplet/include/SearchByTriplet.cuh"
#include "../../host/prefix_sum/include/HostPrefixSum.h"
#include "../../device/velo/search_by_triplet/include/ThreeHitTracksFilter.cuh"
#include "../../host/prefix_sum/include/HostPrefixSum.h"
#include "../../device/velo/consolidate_tracks/include/VeloCopyTrackHitNumber.cuh"
#include "../../host/prefix_sum/include/HostPrefixSum.h"
#include "../../device/velo/consolidate_tracks/include/VeloConsolidateTracks.cuh"
#include "../../device/velo/simplified_kalman_filter/include/VeloKalmanFilter.cuh"
#include "../../device/PV/beamlinePV/include/pv_beamline_extrapolate.cuh"
#include "../../device/PV/beamlinePV/include/pv_beamline_histo.cuh"
#include "../../device/PV/beamlinePV/include/pv_beamline_peak.cuh"
#include "../../device/PV/beamlinePV/include/pv_beamline_calculate_denom.cuh"
#include "../../device/PV/beamlinePV/include/pv_beamline_multi_fitter.cuh"
#include "../../device/PV/beamlinePV/include/pv_beamline_cleanup.cuh"

struct host_ut_banks__host_raw_banks_t : host_data_provider::Parameters::host_raw_banks_t,
                                         host_global_event_cut::Parameters::host_ut_raw_banks_t,
                                         host_init_event_list::Parameters::host_ut_raw_banks_t {
  using type = host_data_provider::Parameters::host_raw_banks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "host_ut_banks__host_raw_banks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct host_ut_banks__host_raw_offsets_t : host_data_provider::Parameters::host_raw_offsets_t,
                                           host_global_event_cut::Parameters::host_ut_raw_offsets_t,
                                           host_init_event_list::Parameters::host_ut_raw_offsets_t {
  using type = host_data_provider::Parameters::host_raw_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "host_ut_banks__host_raw_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct host_scifi_banks__host_raw_banks_t : host_data_provider::Parameters::host_raw_banks_t,
                                            host_global_event_cut::Parameters::host_scifi_raw_banks_t,
                                            host_init_event_list::Parameters::host_scifi_raw_banks_t {
  using type = host_data_provider::Parameters::host_raw_banks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "host_scifi_banks__host_raw_banks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct host_scifi_banks__host_raw_offsets_t : host_data_provider::Parameters::host_raw_offsets_t,
                                              host_global_event_cut::Parameters::host_scifi_raw_offsets_t,
                                              host_init_event_list::Parameters::host_scifi_raw_offsets_t {
  using type = host_data_provider::Parameters::host_raw_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "host_scifi_banks__host_raw_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct initialize_lists__host_event_list_t : host_global_event_cut::Parameters::host_event_list_t {
  using type = host_global_event_cut::Parameters::host_event_list_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "initialize_lists__host_event_list_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct initialize_lists__host_number_of_events_t
  : host_global_event_cut::Parameters::host_number_of_events_t,
    velo_calculate_number_of_candidates::Parameters::host_number_of_events_t,
    velo_estimate_input_size::Parameters::host_number_of_events_t,
    velo_masked_clustering::Parameters::host_number_of_events_t,
    velo_calculate_phi_and_sort::Parameters::host_number_of_events_t,
    velo_search_by_triplet::Parameters::host_number_of_events_t,
    velo_three_hit_tracks_filter::Parameters::host_number_of_events_t,
    velo_copy_track_hit_number::Parameters::host_number_of_events_t,
    velo_consolidate_tracks::Parameters::host_number_of_events_t,
    velo_kalman_filter::Parameters::host_number_of_events_t,
    pv_beamline_extrapolate::Parameters::host_number_of_events_t,
    pv_beamline_histo::Parameters::host_number_of_events_t,
    pv_beamline_peak::Parameters::host_number_of_events_t,
    pv_beamline_calculate_denom::Parameters::host_number_of_events_t,
    pv_beamline_multi_fitter::Parameters::host_number_of_events_t,
    pv_beamline_cleanup::Parameters::host_number_of_events_t {
  using type = host_global_event_cut::Parameters::host_number_of_events_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "initialize_lists__host_number_of_events_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct initialize_lists__host_number_of_selected_events_t
  : host_global_event_cut::Parameters::host_number_of_selected_events_t {
  using type = host_global_event_cut::Parameters::host_number_of_selected_events_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "initialize_lists__host_number_of_selected_events_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct initialize_lists__dev_number_of_events_t : host_global_event_cut::Parameters::dev_number_of_events_t,
                                                  velo_masked_clustering::Parameters::dev_number_of_events_t,
                                                  velo_calculate_phi_and_sort::Parameters::dev_number_of_events_t,
                                                  velo_search_by_triplet::Parameters::dev_number_of_events_t,
                                                  velo_three_hit_tracks_filter::Parameters::dev_number_of_events_t,
                                                  velo_consolidate_tracks::Parameters::dev_number_of_events_t,
                                                  velo_kalman_filter::Parameters::dev_number_of_events_t,
                                                  pv_beamline_extrapolate::Parameters::dev_number_of_events_t,
                                                  pv_beamline_histo::Parameters::dev_number_of_events_t,
                                                  pv_beamline_peak::Parameters::dev_number_of_events_t,
                                                  pv_beamline_calculate_denom::Parameters::dev_number_of_events_t,
                                                  pv_beamline_multi_fitter::Parameters::dev_number_of_events_t {
  using type = host_global_event_cut::Parameters::dev_number_of_events_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "initialize_lists__dev_number_of_events_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct initialize_lists__dev_event_list_t : host_global_event_cut::Parameters::dev_event_list_t,
                                            velo_calculate_number_of_candidates::Parameters::dev_event_list_t,
                                            velo_estimate_input_size::Parameters::dev_event_list_t,
                                            velo_masked_clustering::Parameters::dev_event_list_t,
                                            velo_calculate_phi_and_sort::Parameters::dev_event_list_t,
                                            velo_search_by_triplet::Parameters::dev_event_list_t,
                                            velo_three_hit_tracks_filter::Parameters::dev_event_list_t,
                                            velo_consolidate_tracks::Parameters::dev_event_list_t,
                                            velo_kalman_filter::Parameters::dev_event_list_t,
                                            pv_beamline_extrapolate::Parameters::dev_event_list_t,
                                            pv_beamline_histo::Parameters::dev_event_list_t,
                                            pv_beamline_peak::Parameters::dev_event_list_t,
                                            pv_beamline_calculate_denom::Parameters::dev_event_list_t,
                                            pv_beamline_multi_fitter::Parameters::dev_event_list_t,
                                            pv_beamline_cleanup::Parameters::dev_event_list_t {
  using type = host_global_event_cut::Parameters::dev_event_list_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "initialize_lists__dev_event_list_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct full_event_list__host_number_of_events_t : host_init_event_list::Parameters::host_number_of_events_t {
  using type = host_init_event_list::Parameters::host_number_of_events_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "full_event_list__host_number_of_events_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct full_event_list__host_event_list_t : host_init_event_list::Parameters::host_event_list_t {
  using type = host_init_event_list::Parameters::host_event_list_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "full_event_list__host_event_list_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct full_event_list__dev_number_of_events_t : host_init_event_list::Parameters::dev_number_of_events_t {
  using type = host_init_event_list::Parameters::dev_number_of_events_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "full_event_list__dev_number_of_events_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct full_event_list__dev_event_list_t : host_init_event_list::Parameters::dev_event_list_t {
  using type = host_init_event_list::Parameters::dev_event_list_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "full_event_list__dev_event_list_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_banks__dev_raw_banks_t : data_provider::Parameters::dev_raw_banks_t,
                                     velo_calculate_number_of_candidates::Parameters::dev_velo_raw_input_t,
                                     velo_estimate_input_size::Parameters::dev_velo_raw_input_t,
                                     velo_masked_clustering::Parameters::dev_velo_raw_input_t {
  using type = data_provider::Parameters::dev_raw_banks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_banks__dev_raw_banks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_banks__dev_raw_offsets_t : data_provider::Parameters::dev_raw_offsets_t,
                                       velo_calculate_number_of_candidates::Parameters::dev_velo_raw_input_offsets_t,
                                       velo_estimate_input_size::Parameters::dev_velo_raw_input_offsets_t,
                                       velo_masked_clustering::Parameters::dev_velo_raw_input_offsets_t {
  using type = data_provider::Parameters::dev_raw_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_banks__dev_raw_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_calculate_number_of_candidates__dev_number_of_candidates_t
  : velo_calculate_number_of_candidates::Parameters::dev_number_of_candidates_t,
    host_prefix_sum::Parameters::dev_input_buffer_t {
  using type = velo_calculate_number_of_candidates::Parameters::dev_number_of_candidates_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_calculate_number_of_candidates__dev_number_of_candidates_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct prefix_sum_offsets_velo_candidates__host_total_sum_holder_t
  : host_prefix_sum::Parameters::host_total_sum_holder_t,
    velo_estimate_input_size::Parameters::host_number_of_cluster_candidates_t {
  using type = host_prefix_sum::Parameters::host_total_sum_holder_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "prefix_sum_offsets_velo_candidates__host_total_sum_holder_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct prefix_sum_offsets_velo_candidates__dev_output_buffer_t
  : host_prefix_sum::Parameters::dev_output_buffer_t,
    velo_estimate_input_size::Parameters::dev_candidates_offsets_t,
    velo_masked_clustering::Parameters::dev_candidates_offsets_t {
  using type = host_prefix_sum::Parameters::dev_output_buffer_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "prefix_sum_offsets_velo_candidates__dev_output_buffer_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_estimate_input_size__dev_estimated_input_size_t
  : velo_estimate_input_size::Parameters::dev_estimated_input_size_t,
    host_prefix_sum::Parameters::dev_input_buffer_t {
  using type = velo_estimate_input_size::Parameters::dev_estimated_input_size_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_estimate_input_size__dev_estimated_input_size_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_estimate_input_size__dev_module_candidate_num_t
  : velo_estimate_input_size::Parameters::dev_module_candidate_num_t,
    velo_masked_clustering::Parameters::dev_module_candidate_num_t {
  using type = velo_estimate_input_size::Parameters::dev_module_candidate_num_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_estimate_input_size__dev_module_candidate_num_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_estimate_input_size__dev_cluster_candidates_t
  : velo_estimate_input_size::Parameters::dev_cluster_candidates_t,
    velo_masked_clustering::Parameters::dev_cluster_candidates_t {
  using type = velo_estimate_input_size::Parameters::dev_cluster_candidates_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_estimate_input_size__dev_cluster_candidates_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct prefix_sum_offsets_estimated_input_size__host_total_sum_holder_t
  : host_prefix_sum::Parameters::host_total_sum_holder_t,
    velo_masked_clustering::Parameters::host_total_number_of_velo_clusters_t,
    velo_calculate_phi_and_sort::Parameters::host_total_number_of_velo_clusters_t,
    velo_search_by_triplet::Parameters::host_total_number_of_velo_clusters_t {
  using type = host_prefix_sum::Parameters::host_total_sum_holder_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "prefix_sum_offsets_estimated_input_size__host_total_sum_holder_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct prefix_sum_offsets_estimated_input_size__dev_output_buffer_t
  : host_prefix_sum::Parameters::dev_output_buffer_t,
    velo_masked_clustering::Parameters::dev_offsets_estimated_input_size_t,
    velo_calculate_phi_and_sort::Parameters::dev_offsets_estimated_input_size_t,
    velo_search_by_triplet::Parameters::dev_offsets_estimated_input_size_t,
    velo_three_hit_tracks_filter::Parameters::dev_offsets_estimated_input_size_t,
    velo_consolidate_tracks::Parameters::dev_offsets_estimated_input_size_t {
  using type = host_prefix_sum::Parameters::dev_output_buffer_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "prefix_sum_offsets_estimated_input_size__dev_output_buffer_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_masked_clustering__dev_module_cluster_num_t
  : velo_masked_clustering::Parameters::dev_module_cluster_num_t,
    velo_calculate_phi_and_sort::Parameters::dev_module_cluster_num_t,
    velo_search_by_triplet::Parameters::dev_module_cluster_num_t {
  using type = velo_masked_clustering::Parameters::dev_module_cluster_num_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_masked_clustering__dev_module_cluster_num_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_masked_clustering__dev_velo_cluster_container_t
  : velo_masked_clustering::Parameters::dev_velo_cluster_container_t,
    velo_calculate_phi_and_sort::Parameters::dev_velo_cluster_container_t {
  using type = velo_masked_clustering::Parameters::dev_velo_cluster_container_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_masked_clustering__dev_velo_cluster_container_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_calculate_phi_and_sort__dev_sorted_velo_cluster_container_t
  : velo_calculate_phi_and_sort::Parameters::dev_sorted_velo_cluster_container_t,
    velo_search_by_triplet::Parameters::dev_sorted_velo_cluster_container_t,
    velo_three_hit_tracks_filter::Parameters::dev_sorted_velo_cluster_container_t,
    velo_consolidate_tracks::Parameters::dev_sorted_velo_cluster_container_t {
  using type = velo_calculate_phi_and_sort::Parameters::dev_sorted_velo_cluster_container_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_calculate_phi_and_sort__dev_sorted_velo_cluster_container_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_calculate_phi_and_sort__dev_hit_permutation_t
  : velo_calculate_phi_and_sort::Parameters::dev_hit_permutation_t {
  using type = velo_calculate_phi_and_sort::Parameters::dev_hit_permutation_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_calculate_phi_and_sort__dev_hit_permutation_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_calculate_phi_and_sort__dev_hit_phi_t : velo_calculate_phi_and_sort::Parameters::dev_hit_phi_t,
                                                    velo_search_by_triplet::Parameters::dev_hit_phi_t {
  using type = velo_calculate_phi_and_sort::Parameters::dev_hit_phi_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_calculate_phi_and_sort__dev_hit_phi_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_search_by_triplet__dev_tracks_t : velo_search_by_triplet::Parameters::dev_tracks_t,
                                              velo_copy_track_hit_number::Parameters::dev_tracks_t,
                                              velo_consolidate_tracks::Parameters::dev_tracks_t {
  using type = velo_search_by_triplet::Parameters::dev_tracks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_search_by_triplet__dev_tracks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_search_by_triplet__dev_tracklets_t : velo_search_by_triplet::Parameters::dev_tracklets_t {
  using type = velo_search_by_triplet::Parameters::dev_tracklets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_search_by_triplet__dev_tracklets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_search_by_triplet__dev_tracks_to_follow_t : velo_search_by_triplet::Parameters::dev_tracks_to_follow_t {
  using type = velo_search_by_triplet::Parameters::dev_tracks_to_follow_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_search_by_triplet__dev_tracks_to_follow_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_search_by_triplet__dev_three_hit_tracks_t
  : velo_search_by_triplet::Parameters::dev_three_hit_tracks_t,
    velo_three_hit_tracks_filter::Parameters::dev_three_hit_tracks_input_t {
  using type = velo_search_by_triplet::Parameters::dev_three_hit_tracks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_search_by_triplet__dev_three_hit_tracks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_search_by_triplet__dev_hit_used_t : velo_search_by_triplet::Parameters::dev_hit_used_t,
                                                velo_three_hit_tracks_filter::Parameters::dev_hit_used_t {
  using type = velo_search_by_triplet::Parameters::dev_hit_used_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_search_by_triplet__dev_hit_used_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_search_by_triplet__dev_atomics_velo_t : velo_search_by_triplet::Parameters::dev_atomics_velo_t,
                                                    velo_three_hit_tracks_filter::Parameters::dev_atomics_velo_t {
  using type = velo_search_by_triplet::Parameters::dev_atomics_velo_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_search_by_triplet__dev_atomics_velo_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_search_by_triplet__dev_rel_indices_t : velo_search_by_triplet::Parameters::dev_rel_indices_t {
  using type = velo_search_by_triplet::Parameters::dev_rel_indices_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_search_by_triplet__dev_rel_indices_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_search_by_triplet__dev_number_of_velo_tracks_t
  : velo_search_by_triplet::Parameters::dev_number_of_velo_tracks_t,
    host_prefix_sum::Parameters::dev_input_buffer_t {
  using type = velo_search_by_triplet::Parameters::dev_number_of_velo_tracks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_search_by_triplet__dev_number_of_velo_tracks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct prefix_sum_offsets_velo_tracks__host_total_sum_holder_t
  : host_prefix_sum::Parameters::host_total_sum_holder_t,
    velo_copy_track_hit_number::Parameters::host_number_of_velo_tracks_at_least_four_hits_t {
  using type = host_prefix_sum::Parameters::host_total_sum_holder_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "prefix_sum_offsets_velo_tracks__host_total_sum_holder_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct prefix_sum_offsets_velo_tracks__dev_output_buffer_t
  : host_prefix_sum::Parameters::dev_output_buffer_t,
    velo_copy_track_hit_number::Parameters::dev_offsets_velo_tracks_t {
  using type = host_prefix_sum::Parameters::dev_output_buffer_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "prefix_sum_offsets_velo_tracks__dev_output_buffer_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_three_hit_tracks_filter__dev_three_hit_tracks_output_t
  : velo_three_hit_tracks_filter::Parameters::dev_three_hit_tracks_output_t,
    velo_consolidate_tracks::Parameters::dev_three_hit_tracks_output_t {
  using type = velo_three_hit_tracks_filter::Parameters::dev_three_hit_tracks_output_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_three_hit_tracks_filter__dev_three_hit_tracks_output_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_three_hit_tracks_filter__dev_number_of_three_hit_tracks_output_t
  : velo_three_hit_tracks_filter::Parameters::dev_number_of_three_hit_tracks_output_t,
    host_prefix_sum::Parameters::dev_input_buffer_t {
  using type = velo_three_hit_tracks_filter::Parameters::dev_number_of_three_hit_tracks_output_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_three_hit_tracks_filter__dev_number_of_three_hit_tracks_output_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct prefix_sum_offsets_number_of_three_hit_tracks_filtered__host_total_sum_holder_t
  : host_prefix_sum::Parameters::host_total_sum_holder_t,
    velo_copy_track_hit_number::Parameters::host_number_of_three_hit_tracks_filtered_t,
    velo_consolidate_tracks::Parameters::host_number_of_three_hit_tracks_filtered_t {
  using type = host_prefix_sum::Parameters::host_total_sum_holder_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override
  {
    return "prefix_sum_offsets_number_of_three_hit_tracks_filtered__host_total_sum_holder_t";
  }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct prefix_sum_offsets_number_of_three_hit_tracks_filtered__dev_output_buffer_t
  : host_prefix_sum::Parameters::dev_output_buffer_t,
    velo_copy_track_hit_number::Parameters::dev_offsets_number_of_three_hit_tracks_filtered_t,
    velo_consolidate_tracks::Parameters::dev_offsets_number_of_three_hit_tracks_filtered_t {
  using type = host_prefix_sum::Parameters::dev_output_buffer_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override
  {
    return "prefix_sum_offsets_number_of_three_hit_tracks_filtered__dev_output_buffer_t";
  }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_copy_track_hit_number__host_number_of_reconstructed_velo_tracks_t
  : velo_copy_track_hit_number::Parameters::host_number_of_reconstructed_velo_tracks_t,
    velo_consolidate_tracks::Parameters::host_number_of_reconstructed_velo_tracks_t,
    velo_kalman_filter::Parameters::host_number_of_reconstructed_velo_tracks_t,
    pv_beamline_extrapolate::Parameters::host_number_of_reconstructed_velo_tracks_t,
    pv_beamline_calculate_denom::Parameters::host_number_of_reconstructed_velo_tracks_t,
    pv_beamline_multi_fitter::Parameters::host_number_of_reconstructed_velo_tracks_t {
  using type = velo_copy_track_hit_number::Parameters::host_number_of_reconstructed_velo_tracks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_copy_track_hit_number__host_number_of_reconstructed_velo_tracks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_copy_track_hit_number__dev_velo_track_hit_number_t
  : velo_copy_track_hit_number::Parameters::dev_velo_track_hit_number_t,
    host_prefix_sum::Parameters::dev_input_buffer_t {
  using type = velo_copy_track_hit_number::Parameters::dev_velo_track_hit_number_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_copy_track_hit_number__dev_velo_track_hit_number_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_copy_track_hit_number__dev_offsets_all_velo_tracks_t
  : velo_copy_track_hit_number::Parameters::dev_offsets_all_velo_tracks_t,
    velo_consolidate_tracks::Parameters::dev_offsets_all_velo_tracks_t,
    velo_kalman_filter::Parameters::dev_offsets_all_velo_tracks_t,
    pv_beamline_extrapolate::Parameters::dev_offsets_all_velo_tracks_t,
    pv_beamline_histo::Parameters::dev_offsets_all_velo_tracks_t,
    pv_beamline_calculate_denom::Parameters::dev_offsets_all_velo_tracks_t,
    pv_beamline_multi_fitter::Parameters::dev_offsets_all_velo_tracks_t {
  using type = velo_copy_track_hit_number::Parameters::dev_offsets_all_velo_tracks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_copy_track_hit_number__dev_offsets_all_velo_tracks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct prefix_sum_offsets_velo_track_hit_number__host_total_sum_holder_t
  : host_prefix_sum::Parameters::host_total_sum_holder_t,
    velo_consolidate_tracks::Parameters::host_accumulated_number_of_hits_in_velo_tracks_t {
  using type = host_prefix_sum::Parameters::host_total_sum_holder_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "prefix_sum_offsets_velo_track_hit_number__host_total_sum_holder_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct prefix_sum_offsets_velo_track_hit_number__dev_output_buffer_t
  : host_prefix_sum::Parameters::dev_output_buffer_t,
    velo_consolidate_tracks::Parameters::dev_offsets_velo_track_hit_number_t,
    velo_kalman_filter::Parameters::dev_offsets_velo_track_hit_number_t,
    pv_beamline_extrapolate::Parameters::dev_offsets_velo_track_hit_number_t,
    pv_beamline_histo::Parameters::dev_offsets_velo_track_hit_number_t,
    pv_beamline_calculate_denom::Parameters::dev_offsets_velo_track_hit_number_t,
    pv_beamline_multi_fitter::Parameters::dev_offsets_velo_track_hit_number_t {
  using type = host_prefix_sum::Parameters::dev_output_buffer_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "prefix_sum_offsets_velo_track_hit_number__dev_output_buffer_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_consolidate_tracks__dev_accepted_velo_tracks_t
  : velo_consolidate_tracks::Parameters::dev_accepted_velo_tracks_t {
  using type = velo_consolidate_tracks::Parameters::dev_accepted_velo_tracks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_consolidate_tracks__dev_accepted_velo_tracks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_consolidate_tracks__dev_velo_states_t : velo_consolidate_tracks::Parameters::dev_velo_states_t,
                                                    velo_kalman_filter::Parameters::dev_velo_states_t {
  using type = velo_consolidate_tracks::Parameters::dev_velo_states_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_consolidate_tracks__dev_velo_states_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_consolidate_tracks__dev_velo_track_hits_t : velo_consolidate_tracks::Parameters::dev_velo_track_hits_t,
                                                        velo_kalman_filter::Parameters::dev_velo_track_hits_t {
  using type = velo_consolidate_tracks::Parameters::dev_velo_track_hits_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_consolidate_tracks__dev_velo_track_hits_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_kalman_filter__dev_velo_kalman_beamline_states_t
  : velo_kalman_filter::Parameters::dev_velo_kalman_beamline_states_t,
    pv_beamline_extrapolate::Parameters::dev_velo_kalman_beamline_states_t {
  using type = velo_kalman_filter::Parameters::dev_velo_kalman_beamline_states_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_kalman_filter__dev_velo_kalman_beamline_states_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct pv_beamline_extrapolate__dev_pvtracks_t : pv_beamline_extrapolate::Parameters::dev_pvtracks_t,
                                                 pv_beamline_histo::Parameters::dev_pvtracks_t,
                                                 pv_beamline_calculate_denom::Parameters::dev_pvtracks_t,
                                                 pv_beamline_multi_fitter::Parameters::dev_pvtracks_t {
  using type = pv_beamline_extrapolate::Parameters::dev_pvtracks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "pv_beamline_extrapolate__dev_pvtracks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct pv_beamline_extrapolate__dev_pvtrack_z_t : pv_beamline_extrapolate::Parameters::dev_pvtrack_z_t,
                                                  pv_beamline_multi_fitter::Parameters::dev_pvtrack_z_t {
  using type = pv_beamline_extrapolate::Parameters::dev_pvtrack_z_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "pv_beamline_extrapolate__dev_pvtrack_z_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct pv_beamline_histo__dev_zhisto_t : pv_beamline_histo::Parameters::dev_zhisto_t,
                                         pv_beamline_peak::Parameters::dev_zhisto_t {
  using type = pv_beamline_histo::Parameters::dev_zhisto_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "pv_beamline_histo__dev_zhisto_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct pv_beamline_peak__dev_zpeaks_t : pv_beamline_peak::Parameters::dev_zpeaks_t,
                                        pv_beamline_calculate_denom::Parameters::dev_zpeaks_t,
                                        pv_beamline_multi_fitter::Parameters::dev_zpeaks_t {
  using type = pv_beamline_peak::Parameters::dev_zpeaks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "pv_beamline_peak__dev_zpeaks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct pv_beamline_peak__dev_number_of_zpeaks_t : pv_beamline_peak::Parameters::dev_number_of_zpeaks_t,
                                                  pv_beamline_calculate_denom::Parameters::dev_number_of_zpeaks_t,
                                                  pv_beamline_multi_fitter::Parameters::dev_number_of_zpeaks_t {
  using type = pv_beamline_peak::Parameters::dev_number_of_zpeaks_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "pv_beamline_peak__dev_number_of_zpeaks_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct pv_beamline_calculate_denom__dev_pvtracks_denom_t
  : pv_beamline_calculate_denom::Parameters::dev_pvtracks_denom_t,
    pv_beamline_multi_fitter::Parameters::dev_pvtracks_denom_t {
  using type = pv_beamline_calculate_denom::Parameters::dev_pvtracks_denom_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "pv_beamline_calculate_denom__dev_pvtracks_denom_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct pv_beamline_multi_fitter__dev_multi_fit_vertices_t
  : pv_beamline_multi_fitter::Parameters::dev_multi_fit_vertices_t,
    pv_beamline_cleanup::Parameters::dev_multi_fit_vertices_t {
  using type = pv_beamline_multi_fitter::Parameters::dev_multi_fit_vertices_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "pv_beamline_multi_fitter__dev_multi_fit_vertices_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct pv_beamline_multi_fitter__dev_number_of_multi_fit_vertices_t
  : pv_beamline_multi_fitter::Parameters::dev_number_of_multi_fit_vertices_t,
    pv_beamline_cleanup::Parameters::dev_number_of_multi_fit_vertices_t {
  using type = pv_beamline_multi_fitter::Parameters::dev_number_of_multi_fit_vertices_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "pv_beamline_multi_fitter__dev_number_of_multi_fit_vertices_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct pv_beamline_cleanup__dev_multi_final_vertices_t : pv_beamline_cleanup::Parameters::dev_multi_final_vertices_t {
  using type = pv_beamline_cleanup::Parameters::dev_multi_final_vertices_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "pv_beamline_cleanup__dev_multi_final_vertices_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct pv_beamline_cleanup__dev_number_of_multi_final_vertices_t
  : pv_beamline_cleanup::Parameters::dev_number_of_multi_final_vertices_t {
  using type = pv_beamline_cleanup::Parameters::dev_number_of_multi_final_vertices_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "pv_beamline_cleanup__dev_number_of_multi_final_vertices_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }
private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};

using configured_arguments_t = std::tuple<
  host_ut_banks__host_raw_banks_t,
  host_ut_banks__host_raw_offsets_t,
  host_scifi_banks__host_raw_banks_t,
  host_scifi_banks__host_raw_offsets_t,
  initialize_lists__host_event_list_t,
  initialize_lists__host_number_of_events_t,
  initialize_lists__host_number_of_selected_events_t,
  initialize_lists__dev_number_of_events_t,
  initialize_lists__dev_event_list_t,
  full_event_list__host_number_of_events_t,
  full_event_list__host_event_list_t,
  full_event_list__dev_number_of_events_t,
  full_event_list__dev_event_list_t,
  velo_banks__dev_raw_banks_t,
  velo_banks__dev_raw_offsets_t,
  velo_calculate_number_of_candidates__dev_number_of_candidates_t,
  prefix_sum_offsets_velo_candidates__host_total_sum_holder_t,
  prefix_sum_offsets_velo_candidates__dev_output_buffer_t,
  velo_estimate_input_size__dev_estimated_input_size_t,
  velo_estimate_input_size__dev_module_candidate_num_t,
  velo_estimate_input_size__dev_cluster_candidates_t,
  prefix_sum_offsets_estimated_input_size__host_total_sum_holder_t,
  prefix_sum_offsets_estimated_input_size__dev_output_buffer_t,
  velo_masked_clustering__dev_module_cluster_num_t,
  velo_masked_clustering__dev_velo_cluster_container_t,
  velo_calculate_phi_and_sort__dev_sorted_velo_cluster_container_t,
  velo_calculate_phi_and_sort__dev_hit_permutation_t,
  velo_calculate_phi_and_sort__dev_hit_phi_t,
  velo_search_by_triplet__dev_tracks_t,
  velo_search_by_triplet__dev_tracklets_t,
  velo_search_by_triplet__dev_tracks_to_follow_t,
  velo_search_by_triplet__dev_three_hit_tracks_t,
  velo_search_by_triplet__dev_hit_used_t,
  velo_search_by_triplet__dev_atomics_velo_t,
  velo_search_by_triplet__dev_rel_indices_t,
  velo_search_by_triplet__dev_number_of_velo_tracks_t,
  prefix_sum_offsets_velo_tracks__host_total_sum_holder_t,
  prefix_sum_offsets_velo_tracks__dev_output_buffer_t,
  velo_three_hit_tracks_filter__dev_three_hit_tracks_output_t,
  velo_three_hit_tracks_filter__dev_number_of_three_hit_tracks_output_t,
  prefix_sum_offsets_number_of_three_hit_tracks_filtered__host_total_sum_holder_t,
  prefix_sum_offsets_number_of_three_hit_tracks_filtered__dev_output_buffer_t,
  velo_copy_track_hit_number__host_number_of_reconstructed_velo_tracks_t,
  velo_copy_track_hit_number__dev_velo_track_hit_number_t,
  velo_copy_track_hit_number__dev_offsets_all_velo_tracks_t,
  prefix_sum_offsets_velo_track_hit_number__host_total_sum_holder_t,
  prefix_sum_offsets_velo_track_hit_number__dev_output_buffer_t,
  velo_consolidate_tracks__dev_accepted_velo_tracks_t,
  velo_consolidate_tracks__dev_velo_states_t,
  velo_consolidate_tracks__dev_velo_track_hits_t,
  velo_kalman_filter__dev_velo_kalman_beamline_states_t,
  pv_beamline_extrapolate__dev_pvtracks_t,
  pv_beamline_extrapolate__dev_pvtrack_z_t,
  pv_beamline_histo__dev_zhisto_t,
  pv_beamline_peak__dev_zpeaks_t,
  pv_beamline_peak__dev_number_of_zpeaks_t,
  pv_beamline_calculate_denom__dev_pvtracks_denom_t,
  pv_beamline_multi_fitter__dev_multi_fit_vertices_t,
  pv_beamline_multi_fitter__dev_number_of_multi_fit_vertices_t,
  pv_beamline_cleanup__dev_multi_final_vertices_t,
  pv_beamline_cleanup__dev_number_of_multi_final_vertices_t>;

using configured_sequence_t = std::tuple<
  host_data_provider::host_data_provider_t,
  host_data_provider::host_data_provider_t,
  host_global_event_cut::host_global_event_cut_t,
  host_init_event_list::host_init_event_list_t,
  data_provider::data_provider_t,
  velo_calculate_number_of_candidates::velo_calculate_number_of_candidates_t,
  host_prefix_sum::host_prefix_sum_t,
  velo_estimate_input_size::velo_estimate_input_size_t,
  host_prefix_sum::host_prefix_sum_t,
  velo_masked_clustering::velo_masked_clustering_t,
  velo_calculate_phi_and_sort::velo_calculate_phi_and_sort_t,
  velo_search_by_triplet::velo_search_by_triplet_t,
  host_prefix_sum::host_prefix_sum_t,
  velo_three_hit_tracks_filter::velo_three_hit_tracks_filter_t,
  host_prefix_sum::host_prefix_sum_t,
  velo_copy_track_hit_number::velo_copy_track_hit_number_t,
  host_prefix_sum::host_prefix_sum_t,
  velo_consolidate_tracks::velo_consolidate_tracks_t,
  velo_kalman_filter::velo_kalman_filter_t,
  pv_beamline_extrapolate::pv_beamline_extrapolate_t,
  pv_beamline_histo::pv_beamline_histo_t,
  pv_beamline_peak::pv_beamline_peak_t,
  pv_beamline_calculate_denom::pv_beamline_calculate_denom_t,
  pv_beamline_multi_fitter::pv_beamline_multi_fitter_t,
  pv_beamline_cleanup::pv_beamline_cleanup_t>;

using configured_sequence_arguments_t = std::tuple<
  std::tuple<host_ut_banks__host_raw_banks_t, host_ut_banks__host_raw_offsets_t>,
  std::tuple<host_scifi_banks__host_raw_banks_t, host_scifi_banks__host_raw_offsets_t>,
  std::tuple<
    host_ut_banks__host_raw_banks_t,
    host_ut_banks__host_raw_offsets_t,
    host_scifi_banks__host_raw_banks_t,
    host_scifi_banks__host_raw_offsets_t,
    initialize_lists__host_event_list_t,
    initialize_lists__host_number_of_events_t,
    initialize_lists__host_number_of_selected_events_t,
    initialize_lists__dev_number_of_events_t,
    initialize_lists__dev_event_list_t>,
  std::tuple<
    host_ut_banks__host_raw_banks_t,
    host_ut_banks__host_raw_offsets_t,
    host_scifi_banks__host_raw_banks_t,
    host_scifi_banks__host_raw_offsets_t,
    full_event_list__host_number_of_events_t,
    full_event_list__host_event_list_t,
    full_event_list__dev_number_of_events_t,
    full_event_list__dev_event_list_t>,
  std::tuple<velo_banks__dev_raw_banks_t, velo_banks__dev_raw_offsets_t>,
  std::tuple<
    initialize_lists__host_number_of_events_t,
    initialize_lists__dev_event_list_t,
    velo_banks__dev_raw_banks_t,
    velo_banks__dev_raw_offsets_t,
    velo_calculate_number_of_candidates__dev_number_of_candidates_t>,
  std::tuple<
    prefix_sum_offsets_velo_candidates__host_total_sum_holder_t,
    velo_calculate_number_of_candidates__dev_number_of_candidates_t,
    prefix_sum_offsets_velo_candidates__dev_output_buffer_t>,
  std::tuple<
    initialize_lists__host_number_of_events_t,
    prefix_sum_offsets_velo_candidates__host_total_sum_holder_t,
    initialize_lists__dev_event_list_t,
    prefix_sum_offsets_velo_candidates__dev_output_buffer_t,
    velo_banks__dev_raw_banks_t,
    velo_banks__dev_raw_offsets_t,
    velo_estimate_input_size__dev_estimated_input_size_t,
    velo_estimate_input_size__dev_module_candidate_num_t,
    velo_estimate_input_size__dev_cluster_candidates_t>,
  std::tuple<
    prefix_sum_offsets_estimated_input_size__host_total_sum_holder_t,
    velo_estimate_input_size__dev_estimated_input_size_t,
    prefix_sum_offsets_estimated_input_size__dev_output_buffer_t>,
  std::tuple<
    prefix_sum_offsets_estimated_input_size__host_total_sum_holder_t,
    initialize_lists__host_number_of_events_t,
    velo_banks__dev_raw_banks_t,
    velo_banks__dev_raw_offsets_t,
    prefix_sum_offsets_estimated_input_size__dev_output_buffer_t,
    velo_estimate_input_size__dev_module_candidate_num_t,
    velo_estimate_input_size__dev_cluster_candidates_t,
    initialize_lists__dev_event_list_t,
    prefix_sum_offsets_velo_candidates__dev_output_buffer_t,
    initialize_lists__dev_number_of_events_t,
    velo_masked_clustering__dev_module_cluster_num_t,
    velo_masked_clustering__dev_velo_cluster_container_t>,
  std::tuple<
    initialize_lists__host_number_of_events_t,
    prefix_sum_offsets_estimated_input_size__host_total_sum_holder_t,
    initialize_lists__dev_event_list_t,
    prefix_sum_offsets_estimated_input_size__dev_output_buffer_t,
    velo_masked_clustering__dev_module_cluster_num_t,
    velo_masked_clustering__dev_velo_cluster_container_t,
    initialize_lists__dev_number_of_events_t,
    velo_calculate_phi_and_sort__dev_sorted_velo_cluster_container_t,
    velo_calculate_phi_and_sort__dev_hit_permutation_t,
    velo_calculate_phi_and_sort__dev_hit_phi_t>,
  std::tuple<
    initialize_lists__host_number_of_events_t,
    prefix_sum_offsets_estimated_input_size__host_total_sum_holder_t,
    initialize_lists__dev_event_list_t,
    initialize_lists__dev_number_of_events_t,
    velo_calculate_phi_and_sort__dev_sorted_velo_cluster_container_t,
    prefix_sum_offsets_estimated_input_size__dev_output_buffer_t,
    velo_masked_clustering__dev_module_cluster_num_t,
    velo_calculate_phi_and_sort__dev_hit_phi_t,
    velo_search_by_triplet__dev_tracks_t,
    velo_search_by_triplet__dev_tracklets_t,
    velo_search_by_triplet__dev_tracks_to_follow_t,
    velo_search_by_triplet__dev_three_hit_tracks_t,
    velo_search_by_triplet__dev_hit_used_t,
    velo_search_by_triplet__dev_atomics_velo_t,
    velo_search_by_triplet__dev_rel_indices_t,
    velo_search_by_triplet__dev_number_of_velo_tracks_t>,
  std::tuple<
    prefix_sum_offsets_velo_tracks__host_total_sum_holder_t,
    velo_search_by_triplet__dev_number_of_velo_tracks_t,
    prefix_sum_offsets_velo_tracks__dev_output_buffer_t>,
  std::tuple<
    initialize_lists__host_number_of_events_t,
    initialize_lists__dev_event_list_t,
    velo_calculate_phi_and_sort__dev_sorted_velo_cluster_container_t,
    prefix_sum_offsets_estimated_input_size__dev_output_buffer_t,
    velo_search_by_triplet__dev_three_hit_tracks_t,
    velo_search_by_triplet__dev_atomics_velo_t,
    velo_search_by_triplet__dev_hit_used_t,
    initialize_lists__dev_number_of_events_t,
    velo_three_hit_tracks_filter__dev_three_hit_tracks_output_t,
    velo_three_hit_tracks_filter__dev_number_of_three_hit_tracks_output_t>,
  std::tuple<
    prefix_sum_offsets_number_of_three_hit_tracks_filtered__host_total_sum_holder_t,
    velo_three_hit_tracks_filter__dev_number_of_three_hit_tracks_output_t,
    prefix_sum_offsets_number_of_three_hit_tracks_filtered__dev_output_buffer_t>,
  std::tuple<
    initialize_lists__host_number_of_events_t,
    prefix_sum_offsets_velo_tracks__host_total_sum_holder_t,
    prefix_sum_offsets_number_of_three_hit_tracks_filtered__host_total_sum_holder_t,
    velo_copy_track_hit_number__host_number_of_reconstructed_velo_tracks_t,
    velo_search_by_triplet__dev_tracks_t,
    prefix_sum_offsets_velo_tracks__dev_output_buffer_t,
    prefix_sum_offsets_number_of_three_hit_tracks_filtered__dev_output_buffer_t,
    velo_copy_track_hit_number__dev_velo_track_hit_number_t,
    velo_copy_track_hit_number__dev_offsets_all_velo_tracks_t>,
  std::tuple<
    prefix_sum_offsets_velo_track_hit_number__host_total_sum_holder_t,
    velo_copy_track_hit_number__dev_velo_track_hit_number_t,
    prefix_sum_offsets_velo_track_hit_number__dev_output_buffer_t>,
  std::tuple<
    prefix_sum_offsets_velo_track_hit_number__host_total_sum_holder_t,
    velo_copy_track_hit_number__host_number_of_reconstructed_velo_tracks_t,
    prefix_sum_offsets_number_of_three_hit_tracks_filtered__host_total_sum_holder_t,
    initialize_lists__host_number_of_events_t,
    initialize_lists__dev_event_list_t,
    velo_copy_track_hit_number__dev_offsets_all_velo_tracks_t,
    velo_search_by_triplet__dev_tracks_t,
    prefix_sum_offsets_velo_track_hit_number__dev_output_buffer_t,
    velo_calculate_phi_and_sort__dev_sorted_velo_cluster_container_t,
    prefix_sum_offsets_estimated_input_size__dev_output_buffer_t,
    velo_three_hit_tracks_filter__dev_three_hit_tracks_output_t,
    prefix_sum_offsets_number_of_three_hit_tracks_filtered__dev_output_buffer_t,
    initialize_lists__dev_number_of_events_t,
    velo_consolidate_tracks__dev_accepted_velo_tracks_t,
    velo_consolidate_tracks__dev_velo_states_t,
    velo_consolidate_tracks__dev_velo_track_hits_t>,
  std::tuple<
    velo_copy_track_hit_number__host_number_of_reconstructed_velo_tracks_t,
    initialize_lists__host_number_of_events_t,
    initialize_lists__dev_event_list_t,
    initialize_lists__dev_number_of_events_t,
    velo_copy_track_hit_number__dev_offsets_all_velo_tracks_t,
    prefix_sum_offsets_velo_track_hit_number__dev_output_buffer_t,
    velo_consolidate_tracks__dev_velo_track_hits_t,
    velo_consolidate_tracks__dev_velo_states_t,
    velo_kalman_filter__dev_velo_kalman_beamline_states_t>,
  std::tuple<
    velo_copy_track_hit_number__host_number_of_reconstructed_velo_tracks_t,
    initialize_lists__host_number_of_events_t,
    initialize_lists__dev_event_list_t,
    initialize_lists__dev_number_of_events_t,
    velo_kalman_filter__dev_velo_kalman_beamline_states_t,
    velo_copy_track_hit_number__dev_offsets_all_velo_tracks_t,
    prefix_sum_offsets_velo_track_hit_number__dev_output_buffer_t,
    pv_beamline_extrapolate__dev_pvtracks_t,
    pv_beamline_extrapolate__dev_pvtrack_z_t>,
  std::tuple<
    initialize_lists__host_number_of_events_t,
    initialize_lists__dev_event_list_t,
    initialize_lists__dev_number_of_events_t,
    velo_copy_track_hit_number__dev_offsets_all_velo_tracks_t,
    prefix_sum_offsets_velo_track_hit_number__dev_output_buffer_t,
    pv_beamline_extrapolate__dev_pvtracks_t,
    pv_beamline_histo__dev_zhisto_t>,
  std::tuple<
    initialize_lists__host_number_of_events_t,
    initialize_lists__dev_event_list_t,
    initialize_lists__dev_number_of_events_t,
    pv_beamline_histo__dev_zhisto_t,
    pv_beamline_peak__dev_zpeaks_t,
    pv_beamline_peak__dev_number_of_zpeaks_t>,
  std::tuple<
    initialize_lists__host_number_of_events_t,
    velo_copy_track_hit_number__host_number_of_reconstructed_velo_tracks_t,
    initialize_lists__dev_event_list_t,
    initialize_lists__dev_number_of_events_t,
    velo_copy_track_hit_number__dev_offsets_all_velo_tracks_t,
    prefix_sum_offsets_velo_track_hit_number__dev_output_buffer_t,
    pv_beamline_extrapolate__dev_pvtracks_t,
    pv_beamline_peak__dev_zpeaks_t,
    pv_beamline_peak__dev_number_of_zpeaks_t,
    pv_beamline_calculate_denom__dev_pvtracks_denom_t>,
  std::tuple<
    initialize_lists__host_number_of_events_t,
    velo_copy_track_hit_number__host_number_of_reconstructed_velo_tracks_t,
    initialize_lists__dev_event_list_t,
    initialize_lists__dev_number_of_events_t,
    velo_copy_track_hit_number__dev_offsets_all_velo_tracks_t,
    prefix_sum_offsets_velo_track_hit_number__dev_output_buffer_t,
    pv_beamline_extrapolate__dev_pvtracks_t,
    pv_beamline_calculate_denom__dev_pvtracks_denom_t,
    pv_beamline_peak__dev_zpeaks_t,
    pv_beamline_peak__dev_number_of_zpeaks_t,
    pv_beamline_extrapolate__dev_pvtrack_z_t,
    pv_beamline_multi_fitter__dev_multi_fit_vertices_t,
    pv_beamline_multi_fitter__dev_number_of_multi_fit_vertices_t>,
  std::tuple<
    initialize_lists__host_number_of_events_t,
    initialize_lists__dev_event_list_t,
    pv_beamline_multi_fitter__dev_multi_fit_vertices_t,
    pv_beamline_multi_fitter__dev_number_of_multi_fit_vertices_t,
    pv_beamline_cleanup__dev_multi_final_vertices_t,
    pv_beamline_cleanup__dev_number_of_multi_final_vertices_t>>;

void inline populate_sequence_algorithm_names(configured_sequence_t& sequence)
{
  std::get<0>(sequence).set_name("host_ut_banks");
  std::get<1>(sequence).set_name("host_scifi_banks");
  std::get<2>(sequence).set_name("initialize_lists");
  std::get<3>(sequence).set_name("full_event_list");
  std::get<4>(sequence).set_name("velo_banks");
  std::get<5>(sequence).set_name("velo_calculate_number_of_candidates");
  std::get<6>(sequence).set_name("prefix_sum_offsets_velo_candidates");
  std::get<7>(sequence).set_name("velo_estimate_input_size");
  std::get<8>(sequence).set_name("prefix_sum_offsets_estimated_input_size");
  std::get<9>(sequence).set_name("velo_masked_clustering");
  std::get<10>(sequence).set_name("velo_calculate_phi_and_sort");
  std::get<11>(sequence).set_name("velo_search_by_triplet");
  std::get<12>(sequence).set_name("prefix_sum_offsets_velo_tracks");
  std::get<13>(sequence).set_name("velo_three_hit_tracks_filter");
  std::get<14>(sequence).set_name("prefix_sum_offsets_number_of_three_hit_tracks_filtered");
  std::get<15>(sequence).set_name("velo_copy_track_hit_number");
  std::get<16>(sequence).set_name("prefix_sum_offsets_velo_track_hit_number");
  std::get<17>(sequence).set_name("velo_consolidate_tracks");
  std::get<18>(sequence).set_name("velo_kalman_filter");
  std::get<19>(sequence).set_name("pv_beamline_extrapolate");
  std::get<20>(sequence).set_name("pv_beamline_histo");
  std::get<21>(sequence).set_name("pv_beamline_peak");
  std::get<22>(sequence).set_name("pv_beamline_calculate_denom");
  std::get<23>(sequence).set_name("pv_beamline_multi_fitter");
  std::get<24>(sequence).set_name("pv_beamline_cleanup");
}
