/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "Constants.cuh"
#include "UTDefinitions.cuh"
#include "VeloDefinitions.cuh"
#include "ClusteringDefinitions.cuh"
#include "KalmanParametrizations.cuh"
#include "LookingForwardConstants.cuh"
#include "MuonDefinitions.cuh"
#include "MuonGeometry.cuh"
#include "MuonTables.cuh"

void Constants::reserve_constants()
{
  Allen::malloc((void**) &dev_inv_clus_res, host_inv_clus_res.size() * sizeof(float));
  Allen::malloc((void**) &dev_kalman_params, sizeof(ParKalmanFilter::KalmanParametrizations));
  Allen::malloc((void**) &dev_looking_forward_constants, sizeof(LookingForward::Constants));
  Allen::malloc((void**) &dev_muon_foi, sizeof(Muon::Constants::FieldOfInterest));
  Allen::malloc((void**) &dev_muon_momentum_cuts, 3 * sizeof(float));
  Allen::malloc((void**) &dev_muonmatch_search_muon_chambers, sizeof(MatchUpstreamMuon::MuonChambers));
  Allen::malloc((void**) &dev_muonmatch_search_windows, sizeof(MatchUpstreamMuon::SearchWindows));

  host_ut_region_offsets.resize(UT::Constants::n_layers * UT::Constants::n_regions_in_layer + 1);
  host_ut_dxDy.resize(UT::Constants::n_layers);
  host_unique_x_sector_layer_offsets.resize(UT::Constants::n_layers + 1);
}

void Constants::initialize_constants(
  const std::vector<float>& muon_field_of_interest_params,
  const std::string& folder_params_kalman)
{
  // SciFi constants
  host_inv_clus_res = {1 / 0.05, 1 / 0.08, 1 / 0.11, 1 / 0.14, 1 / 0.17, 1 / 0.20, 1 / 0.23, 1 / 0.26, 1 / 0.29};
  Allen::memcpy(
    dev_inv_clus_res, &host_inv_clus_res, host_inv_clus_res.size() * sizeof(float), Allen::memcpyHostToDevice);

  host_looking_forward_constants = new LookingForward::Constants {};

  // Kalman filter constants.
  ParKalmanFilter::KalmanParametrizations host_kalman_params;
  host_kalman_params.SetParameters(folder_params_kalman, ParKalmanFilter::Polarity::Down);
  Allen::memcpy(
    dev_kalman_params, &host_kalman_params, sizeof(ParKalmanFilter::KalmanParametrizations), Allen::memcpyHostToDevice);

  Allen::memcpy(
    dev_looking_forward_constants,
    host_looking_forward_constants,
    sizeof(LookingForward::Constants),
    Allen::memcpyHostToDevice);

  // Muon constants
  Muon::Constants::FieldOfInterest host_muon_foi;
  std::copy_n(
    muon_field_of_interest_params.data(),
    Muon::Constants::n_regions * Muon::Constants::n_stations * Muon::Constants::FoiParams::n_parameters *
      Muon::Constants::FoiParams::n_coordinates,
    host_muon_foi.params_begin());

  Allen::memcpy(dev_muon_momentum_cuts, &Muon::Constants::momentum_cuts, 3 * sizeof(float), Allen::memcpyHostToDevice);
  Allen::memcpy(dev_muon_foi, &host_muon_foi, sizeof(Muon::Constants::FieldOfInterest), Allen::memcpyHostToDevice);

  // Velo-UT-muon
  MatchUpstreamMuon::MuonChambers host_muonmatch_search_muon_chambers;
  MatchUpstreamMuon::SearchWindows host_muonmatch_search_windows;
  Allen::memcpy(
    dev_muonmatch_search_muon_chambers,
    &host_muonmatch_search_muon_chambers,
    sizeof(MatchUpstreamMuon::MuonChambers),
    Allen::memcpyHostToDevice);

  Allen::memcpy(
    dev_muonmatch_search_windows,
    &host_muonmatch_search_windows,
    sizeof(MatchUpstreamMuon::SearchWindows),
    Allen::memcpyHostToDevice);
}

void Constants::initialize_muon_catboost_model_constants(
  const int n_trees,
  const std::vector<int>& tree_depths,
  const std::vector<int>& tree_offsets,
  const std::vector<float>& leaf_values,
  const std::vector<int>& leaf_offsets,
  const std::vector<float>& split_borders,
  const std::vector<int>& split_features)
{
  muon_catboost_n_trees = n_trees;
  Allen::malloc((void**) &dev_muon_catboost_split_features, split_features.size() * sizeof(int));
  Allen::malloc((void**) &dev_muon_catboost_split_borders, split_borders.size() * sizeof(float));
  Allen::malloc((void**) &dev_muon_catboost_leaf_values, leaf_values.size() * sizeof(float));
  Allen::malloc((void**) &dev_muon_catboost_tree_depths, tree_depths.size() * sizeof(int));
  Allen::malloc((void**) &dev_muon_catboost_tree_offsets, tree_offsets.size() * sizeof(int));
  Allen::malloc((void**) &dev_muon_catboost_leaf_offsets, leaf_offsets.size() * sizeof(int));

  Allen::memcpy(
    dev_muon_catboost_split_features,
    split_features.data(),
    split_features.size() * sizeof(int),
    Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_muon_catboost_split_borders,
    split_borders.data(),
    split_borders.size() * sizeof(float),
    Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_muon_catboost_leaf_values, leaf_values.data(), leaf_values.size() * sizeof(float), Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_muon_catboost_tree_depths, tree_depths.data(), tree_depths.size() * sizeof(int), Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_muon_catboost_tree_offsets, tree_offsets.data(), tree_offsets.size() * sizeof(int), Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_muon_catboost_leaf_offsets, leaf_offsets.data(), leaf_offsets.size() * sizeof(int), Allen::memcpyHostToDevice);
}

void Constants::initialize_two_track_catboost_model_constants(
  const int n_trees,
  const std::vector<int>& tree_depths,
  const std::vector<int>& tree_offsets,
  const std::vector<float>& leaf_values,
  const std::vector<int>& leaf_offsets,
  const std::vector<float>& split_borders,
  const std::vector<int>& split_features)
{
  two_track_catboost_n_trees = n_trees;
  Allen::malloc((void**) &dev_two_track_catboost_split_features, split_features.size() * sizeof(int));
  Allen::malloc((void**) &dev_two_track_catboost_split_borders, split_borders.size() * sizeof(float));
  Allen::malloc((void**) &dev_two_track_catboost_leaf_values, leaf_values.size() * sizeof(float));
  Allen::malloc((void**) &dev_two_track_catboost_tree_depths, tree_depths.size() * sizeof(int));
  Allen::malloc((void**) &dev_two_track_catboost_tree_offsets, tree_offsets.size() * sizeof(int));
  Allen::malloc((void**) &dev_two_track_catboost_leaf_offsets, leaf_offsets.size() * sizeof(int));

  Allen::memcpy(
    dev_two_track_catboost_split_features,
    split_features.data(),
    split_features.size() * sizeof(int),
    Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_two_track_catboost_split_borders,
    split_borders.data(),
    split_borders.size() * sizeof(float),
    Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_two_track_catboost_leaf_values,
    leaf_values.data(),
    leaf_values.size() * sizeof(float),
    Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_two_track_catboost_tree_depths,
    tree_depths.data(),
    tree_depths.size() * sizeof(int),
    Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_two_track_catboost_tree_offsets,
    tree_offsets.data(),
    tree_offsets.size() * sizeof(int),
    Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_two_track_catboost_leaf_offsets,
    leaf_offsets.data(),
    leaf_offsets.size() * sizeof(int),
    Allen::memcpyHostToDevice);
}

void Constants::initialize_two_track_mva_model_constants(
  const std::vector<float>& weights,
  const std::vector<float>& biases,
  const std::vector<int>& layer_sizes,
  const int n_layers,
  const std::vector<float>& monotone_constraints,
  float nominal_cut,
  float lambda)
{
  dev_two_track_mva_nominal_cut = nominal_cut;
  dev_two_track_mva_lambda = lambda;
  dev_two_track_mva_n_layers = n_layers;

  Allen::malloc((void**) &dev_two_track_mva_weights, weights.size() * sizeof(float));
  Allen::malloc((void**) &dev_two_track_mva_biases, biases.size() * sizeof(float));
  Allen::malloc((void**) &dev_two_track_mva_layer_sizes, layer_sizes.size() * sizeof(int));
  Allen::malloc((void**) &dev_two_track_mva_monotone_constraints, monotone_constraints.size() * sizeof(float));

  Allen::memcpy(dev_two_track_mva_weights, weights.data(), weights.size() * sizeof(float), Allen::memcpyHostToDevice);
  Allen::memcpy(dev_two_track_mva_biases, biases.data(), biases.size() * sizeof(float), Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_two_track_mva_layer_sizes, layer_sizes.data(), layer_sizes.size() * sizeof(int), Allen::memcpyHostToDevice);
  Allen::memcpy(
    dev_two_track_mva_monotone_constraints,
    monotone_constraints.data(),
    monotone_constraints.size() * sizeof(float),
    Allen::memcpyHostToDevice);
}
