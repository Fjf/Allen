#pragma once

#include <cstdint>
#include <cassert>
#include "ClusteringDefinitions.cuh"
#include "Handler.cuh"
#include "ArgumentsCommon.cuh"
#include "ArgumentsVelo.cuh"

__global__ void estimate_input_size(
  char* dev_raw_input,
  uint* dev_raw_input_offsets,
  uint* dev_estimated_input_size,
  uint* dev_module_candidate_num,
  uint32_t* dev_cluster_candidates,
  const uint* dev_event_list,
  uint8_t* dev_velo_candidate_ks);

ALGORITHM(
  estimate_input_size,
  velo_estimate_input_size_allen_t,
  ARGUMENTS(
    dev_velo_raw_input,
    dev_velo_raw_input_offsets,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates,
    dev_event_list))

__global__ void estimate_input_size_mep(
  char* dev_raw_input,
  uint* dev_raw_input_offsets,
  uint* dev_estimated_input_size,
  uint* dev_module_candidate_num,
  uint32_t* dev_cluster_candidates,
  const uint* dev_event_list,
  uint8_t* dev_velo_candidate_ks);

ALGORITHM(
  estimate_input_size_mep,
  velo_estimate_input_size_mep_t,
  ARGUMENTS(
    dev_velo_raw_input,
    dev_velo_raw_input_offsets,
    dev_estimated_input_size,
    dev_module_cluster_num,
    dev_module_candidate_num,
    dev_cluster_candidates,
    dev_event_list))

XOR_ALGORITHM(velo_estimate_input_size_mep_t,
              velo_estimate_input_size_allen_t,
              velo_estimate_input_size_t)
