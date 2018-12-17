#pragma once

#include "Handler.cuh"
#include "MuonDefinitions.cuh"

__global__ void muon_catboost_evaluator(
  const float* dev_muon_catboost_features,
  float* dev_muon_catboost_output,
  const float* dev_muon_catboost_leaf_values,
  const int* dev_muon_catboost_leaf_offsets,
  const float* dev_muon_catbost_split_borders,
  const int* dev_muon_catboost_split_features,
  const int* dev_muon_catboost_tree_sizes,
  const int* dev_muon_catboost_tree_offsets,
  const int n_trees
);

__device__ void warp_reduce(
  volatile float* sdata, 
  int tid
);

ALGORITHM(muon_catboost_evaluator, muon_catboost_evaluator_t)
