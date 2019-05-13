#include "Constants.cuh"

void Constants::reserve_constants()
{
  cudaCheck(cudaMalloc((void**) &dev_velo_module_zs, Velo::Constants::n_modules * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_velo_candidate_ks, 9 * sizeof(uint8_t)));
  cudaCheck(cudaMalloc((void**) &dev_velo_sp_patterns, 256 * sizeof(uint8_t)));
  cudaCheck(cudaMalloc((void**) &dev_velo_sp_fx, 512 * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_velo_sp_fy, 512 * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_scifi_tmva1, sizeof(SciFi::Tracking::TMVA)));
  cudaCheck(cudaMalloc((void**) &dev_scifi_tmva2, sizeof(SciFi::Tracking::TMVA)));
  cudaCheck(cudaMalloc((void**) &dev_scifi_constArrays, sizeof(SciFi::Tracking::Arrays)));
  cudaCheck(cudaMalloc((void**) &dev_inv_clus_res, host_inv_clus_res.size() * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_kalman_params, sizeof(ParKalmanFilter::KalmanParametrizations)));
  cudaCheck(cudaMalloc((void**) &dev_looking_forward_constants, sizeof(LookingForward::Constants)));
  cudaCheck(cudaMalloc((void**) &dev_muon_foi, sizeof(Muon::Constants::FieldOfInterest)));
  cudaCheck(cudaMalloc((void**) &dev_muon_momentum_cuts, 3 * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_magnet_polarity, sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_beamline, 2 * sizeof(float)));
}

void Constants::initialize_constants(
  const std::vector<float>& muon_field_of_interest_params,
  const std::string& folder_params_kalman
) {
  // Magnet polarity
  const float host_magnet_polarity = -1.f;
  cudaCheck(cudaMemcpy(
    dev_magnet_polarity, &host_magnet_polarity, sizeof(float), cudaMemcpyHostToDevice));

  // PV constants
  const float host_beamline[2] = {0.0f, 0.0f};
  cudaCheck(cudaMemcpy(
    dev_beamline, &host_beamline, 2 * sizeof(float), cudaMemcpyHostToDevice));

  // Velo module constants
  const std::array<float, Velo::Constants::n_modules> velo_module_zs = {
    -287.5, -275,  -262.5, -250,  -237.5, -225,  -212.5, -200,  -137.5, -125,  -62.5, -50,   -37.5,
    -25,    -12.5, 0,      12.5,  25,     37.5,  50,     62.5,  75,     87.5,  100,   112.5, 125,
    137.5,  150,   162.5,  175,   187.5,  200,   212.5,  225,   237.5,  250,   262.5, 275,   312.5,
    325,    387.5, 400,    487.5, 500,    587.5, 600,    637.5, 650,    687.5, 700,   737.5, 750};
  cudaCheck(cudaMemcpy(
    dev_velo_module_zs, velo_module_zs.data(), velo_module_zs.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Velo clustering candidate ks
  host_candidate_ks = {0, 0, 1, 4, 4, 5, 5, 5, 5};
  cudaCheck(cudaMemcpy(
    dev_velo_candidate_ks,
    host_candidate_ks.data(),
    host_candidate_ks.size() * sizeof(uint8_t),
    cudaMemcpyHostToDevice));

  // Velo clustering patterns
  // Fetch patterns and populate in GPU
  std::vector<uint8_t> sp_patterns(256, 0);
  std::vector<uint8_t> sp_sizes(256, 0);
  std::vector<float> sp_fx(512, 0);
  std::vector<float> sp_fy(512, 0);
  cache_sp_patterns(sp_patterns, sp_sizes, sp_fx, sp_fy);

  cudaCheck(cudaMemcpy(dev_velo_sp_patterns, sp_patterns.data(), sp_patterns.size(), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_velo_sp_fx, sp_fx.data(), sp_fx.size() * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_velo_sp_fy, sp_fy.data(), sp_fy.size() * sizeof(float), cudaMemcpyHostToDevice));

  // SciFi constants
  SciFi::Tracking::TMVA host_tmva1;
  SciFi::Tracking::TMVA host_tmva2;
  SciFi::Tracking::TMVA1_Init(host_tmva1);
  SciFi::Tracking::TMVA2_Init(host_tmva2);
  SciFi::Tracking::Arrays host_constArrays;

  cudaCheck(cudaMemcpy(dev_scifi_tmva1, &host_tmva1, sizeof(SciFi::Tracking::TMVA), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_scifi_tmva2, &host_tmva2, sizeof(SciFi::Tracking::TMVA), cudaMemcpyHostToDevice));
  cudaCheck(
    cudaMemcpy(dev_scifi_constArrays, &host_constArrays, sizeof(SciFi::Tracking::Arrays), cudaMemcpyHostToDevice));
  host_inv_clus_res = {1 / 0.05, 1 / 0.08, 1 / 0.11, 1 / 0.14, 1 / 0.17, 1 / 0.20, 1 / 0.23, 1 / 0.26, 1 / 0.29};
  cudaCheck(cudaMemcpy(dev_inv_clus_res, &host_inv_clus_res, host_inv_clus_res.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Kalman filter constants.
  ParKalmanFilter::KalmanParametrizations host_kalman_params;
  host_kalman_params.SetParameters(folder_params_kalman, ParKalmanFilter::Polarity::Down);
  cudaCheck(cudaMemcpy(
    dev_kalman_params, &host_kalman_params, sizeof(ParKalmanFilter::KalmanParametrizations), cudaMemcpyHostToDevice));

  cudaCheck(cudaMemcpy(
    dev_looking_forward_constants, &host_looking_forward_constants, sizeof(LookingForward::Constants), cudaMemcpyHostToDevice))

  // Muon constants
  Muon::Constants::FieldOfInterest host_muon_foi;
  const float* foi_iterator = muon_field_of_interest_params.data();
  for (int i_station = 0; i_station < Muon::Constants::n_stations; i_station++) {
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_a_x[i_station]);
    foi_iterator += Muon::Constants::n_regions;// * sizeof(float);
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_a_y[i_station]);
    foi_iterator += Muon::Constants::n_regions;// * sizeof(float);
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_b_x[i_station]);
    foi_iterator += Muon::Constants::n_regions;// * sizeof(float);
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_b_y[i_station]);
    foi_iterator += Muon::Constants::n_regions;// * sizeof(float);
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_c_x[i_station]);
    foi_iterator += Muon::Constants::n_regions;// * sizeof(float);
    std::copy_n(foi_iterator, Muon::Constants::n_regions, host_muon_foi.param_c_y[i_station]);
    foi_iterator += Muon::Constants::n_regions;
  }
  cudaCheck(cudaMemcpy(dev_muon_momentum_cuts, &Muon::Constants::momentum_cuts, 3 * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dev_muon_foi, &host_muon_foi, sizeof(Muon::Constants::FieldOfInterest), cudaMemcpyHostToDevice));
}

void Constants::initialize_muon_catboost_model_constants(
  const int n_trees,
  const std::vector<int>& tree_depths,
  const std::vector<int>& tree_offsets,
  const std::vector<float>& leaf_values,
  const std::vector<int>& leaf_offsets,
  const std::vector<float>& split_borders,
  const std::vector<int>& split_features
) {
  muon_catboost_n_trees = n_trees;
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_split_features, split_features.size() * sizeof(int)));
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_split_borders, split_borders.size() * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_leaf_values, leaf_values.size() * sizeof(float)));
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_tree_depths, tree_depths.size() * sizeof(int)));
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_tree_offsets, tree_offsets.size() * sizeof(int)));
  cudaCheck(cudaMalloc((void**) &dev_muon_catboost_leaf_offsets, leaf_offsets.size() * sizeof(int)));

  cudaCheck(cudaMemcpy(
    dev_muon_catboost_split_features,
    split_features.data(),
    split_features.size() * sizeof(int),
    cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_muon_catboost_split_borders,
    split_borders.data(),
    split_borders.size() * sizeof(float),
    cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_muon_catboost_leaf_values, leaf_values.data(), leaf_values.size() * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_muon_catboost_tree_depths, tree_depths.data(), tree_depths.size() * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_muon_catboost_tree_offsets, tree_offsets.data(), tree_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(
    dev_muon_catboost_leaf_offsets, leaf_offsets.data(), leaf_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
}
