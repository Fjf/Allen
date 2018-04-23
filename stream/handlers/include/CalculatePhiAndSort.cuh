#pragma once

#include "../../../cuda/velo/calculate_phi_and_sort/include/CalculatePhiAndSort.cuh"
#include "../../../main/include/CudaCommon.h"
#include <vector>
#include <iostream>

struct CalculatePhiAndSort {
  // Call options
  dim3 num_blocks, num_threads;

  // Cuda stream
  cudaStream_t* stream;

  // Call parameters
  uint* dev_module_cluster_start;
  uint* dev_module_cluster_num;
  uint32_t* dev_velo_cluster_container;
  uint* dev_hit_permutation;

  CalculatePhiAndSort() = default;

  void set(
    const dim3& param_num_blocks,
    const dim3& param_num_threads,
    cudaStream_t& param_stream
  ) {
    num_blocks = param_num_blocks;
    num_threads = param_num_threads;
    stream = &param_stream;
  }

  void setParameters(
    uint* param_dev_module_cluster_start,
    uint* param_dev_module_cluster_num,
    uint32_t* param_dev_velo_cluster_container,
    uint* param_dev_hit_permutation
  ) {
    dev_module_cluster_start = param_dev_module_cluster_start;
    dev_module_cluster_num = param_dev_module_cluster_num;
    dev_velo_cluster_container = param_dev_velo_cluster_container;
    dev_hit_permutation = param_dev_hit_permutation;
  }

  void operator()();

  void print_output(
    const uint number_of_events,
    const int print_max_per_module = -1
  );
};