#pragma once

#include "../../../cuda/velo/simplified_kalman_filter/include/VeloKalmanFilter.cuh"
#include "Handler.cuh"

struct SimplifiedKalmanFilter : public Handler {
  // Call parameters
  uint32_t* dev_velo_cluster_container;
  uint* dev_module_cluster_start;
  int* dev_atomics_storage;
  VeloTracking::TrackHits* dev_tracks;
  VeloState* dev_velo_states;

  SimplifiedKalmanFilter() = default;

  void setParameters(
    uint32_t* param_dev_velo_cluster_container,
    uint* param_dev_module_cluster_start,
    int* param_dev_atomics_storage,
    VeloTracking::TrackHits* param_dev_tracks,
    VeloState* param_dev_velo_states
  ) {
    dev_velo_cluster_container = param_dev_velo_cluster_container;
    dev_module_cluster_start = param_dev_module_cluster_start;
    dev_atomics_storage = param_dev_atomics_storage;
    dev_tracks = param_dev_tracks;
    dev_velo_states = param_dev_velo_states;
  }

  void operator()();
};
