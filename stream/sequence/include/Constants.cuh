#pragma once

#include <array>
#include <cstdint>
#include <algorithm>
#include <numeric>

#include "CudaCommon.h"
#include "VeloDefinitions.cuh"
#include "ClusteringDefinitions.cuh"
#include "ClusteringCommon.h"
#include "VeloUTDefinitions.cuh"
#include "PrForwardConstants.cuh"
#include "TMVA_Forward_1.cuh"
#include "TMVA_Forward_2.cuh"
#include "UTDefinitions.cuh"
#include "Logger.h"
#include "PrVeloUTMagnetToolDefinitions.h"
#include "json.h"

/**
 * @brief Struct intended as a singleton with constants defined on GPU.
 * @details __constant__ memory on the GPU has very few use cases.
 *          Instead, global memory is preferred. Hence, this singleton
 *          should allocate the requested buffers on GPU and serve the
 *          pointers wherever needed.
 *
 *          The pointers are hard-coded. Feel free to write more as needed.
 */
struct Constants {
  std::array<float, VeloUTTracking::n_layers> host_ut_dxDy;
  std::array<uint, VeloUTTracking::n_layers + 1> host_unique_x_sector_layer_offsets;
  std::vector<uint> host_unique_x_sector_offsets;
  std::vector<float> host_unique_sector_xs;
  std::array<uint, VeloUTTracking::n_layers * VeloUTTracking::n_regions_in_layer + 1> host_ut_region_offsets;
  std::array<uint8_t, VeloClustering::lookup_table_size> host_candidate_ks;
  std::array<float, 9> host_inv_clus_res;

  float* dev_velo_module_zs;
  uint8_t* dev_velo_candidate_ks;
  uint8_t* dev_velo_sp_patterns;
  float* dev_velo_sp_fx;
  float* dev_velo_sp_fy;
  float* dev_ut_dxDy;
  SciFi::Tracking::TMVA* dev_scifi_tmva1;
  SciFi::Tracking::TMVA* dev_scifi_tmva2;
  SciFi::Tracking::Arrays* dev_scifi_constArrays;
  uint* dev_unique_x_sector_layer_offsets;
  uint* dev_unique_x_sector_offsets;
  uint* dev_ut_region_offsets;
  float* dev_unique_sector_xs;
  float* dev_inv_clus_res;

  // Geometry constants
  char* dev_velo_geometry;
  char* dev_ut_boards;
  char* dev_ut_geometry;
  char* dev_scifi_geometry;
  const char* host_scifi_geometry; //for debugging
  PrUTMagnetTool* dev_ut_magnet_tool;
  
  // Muon classification model constatns
  int host_muon_catboost_tree_num;
  int host_muon_catboost_float_feature_num;
  int host_muon_catboost_bin_feature_num;
  int* dev_muon_catboost_tree_sizes;
  int* dev_muon_catboost_border_nums;
  int** dev_muon_catboost_tree_splits;
  int* dev_muon_catboost_feature_map;
  int* dev_muon_catboost_border_map;
  float** dev_muon_catboost_borders;
  double** dev_muon_catboost_leaf_values;


  /**
   * @brief Reserves and initializes constants.
   */
  void reserve_and_initialize() {
    reserve_constants();
    initialize_constants();
  }

  /**
   * @brief Reserves the constants of the GPU.
   */
  void reserve_constants();

  /**
   * @brief Initializes constants on the GPU.
   */
  void initialize_constants();

  /**
   * @brief Initializes UT decoding constants.
   */
  void initialize_ut_decoding_constants(const std::vector<char>& ut_geometry);

  /**
   * @brief Initializes geometry constants and magnet field.
   */
  void initialize_geometry_constants(
    const std::vector<char>& velopix_geometry,
    const std::vector<char>& ut_boards,
    const std::vector<char>& ut_geometry,
    const std::vector<char>& ut_magnet_tool,
    const std::vector<char>& scifi_geometry);

  void initialize_muon_catboost_model_constants(const nlohmann::json& model);

};
