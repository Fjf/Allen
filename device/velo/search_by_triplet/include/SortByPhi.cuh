/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <cstdint>
#include <cassert>
#include "BackendCommon.h"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "DeviceAlgorithm.cuh"
#include "VeloTools.cuh"

namespace velo_sort_by_phi {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_total_number_of_velo_clusters_t, unsigned) host_total_number_of_velo_clusters;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_offsets_estimated_input_size_t, unsigned) dev_offsets_estimated_input_size;
    DEVICE_INPUT(dev_module_cluster_num_t, unsigned) dev_module_cluster_num;
    DEVICE_INPUT(dev_velo_cluster_container_t, char) dev_velo_cluster_container;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    DEVICE_OUTPUT(dev_sorted_velo_cluster_container_t, char) dev_sorted_velo_cluster_container;
    DEVICE_OUTPUT(dev_hit_permutation_t, unsigned) dev_hit_permutation;
    DEVICE_INPUT(dev_velo_clusters_t, Velo::Clusters) dev_velo_clusters;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
  };

  __device__ void calculate_permutation(
    const unsigned* module_hitStarts,
    const unsigned* module_hitNums,
    const Velo::Clusters& velo_cluster_container,
    unsigned* hit_permutations);

  __device__ void sort_by_phi(
    const unsigned event_hit_start,
    const unsigned event_number_of_hits,
    const Velo::Clusters& velo_cluster_container,
    Velo::Clusters& velo_sorted_cluster_container,
    unsigned* hit_permutations);

  __global__ void velo_sort_by_phi(Parameters);

  struct velo_sort_by_phi_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      HostBuffers&,
      const Allen::Context& context) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{2, 64, 1}}};
  };
} // namespace velo_sort_by_phi