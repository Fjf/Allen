#pragma once

#include <cstdint>
#include <cassert>
#include "CudaCommon.h"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "DeviceAlgorithm.cuh"

namespace velo_calculate_phi_and_sort {
  struct Arguments {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    HOST_INPUT(host_total_number_of_velo_clusters_t, uint);
    DEVICE_INPUT(dev_offsets_estimated_input_size_t, uint) dev_offsets_estimated_input_size;
    DEVICE_INPUT(dev_module_cluster_num_t, uint) dev_module_cluster_num;
    DEVICE_INPUT(dev_velo_cluster_container_t, uint32_t) dev_velo_cluster_container;
    DEVICE_OUTPUT(dev_sorted_velo_cluster_container_t, uint32_t) dev_sorted_velo_cluster_container;
    DEVICE_OUTPUT(dev_hit_permutation_t, uint) dev_hit_permutation;
    DEVICE_OUTPUT(dev_hit_phi_t, float) dev_hit_phi;
  };

  __device__ void calculate_phi(
    const uint* module_hitStarts,
    const uint* module_hitNums,
    const Velo::Clusters<const uint32_t>& velo_cluster_container,
    float* hit_Phis,
    uint* hit_permutations,
    float* shared_hit_phis);

  __device__ void sort_by_phi(
    const uint event_hit_start,
    const uint event_number_of_hits,
    const Velo::Clusters<const uint32_t>& velo_cluster_container,
    Velo::Clusters<uint32_t>& velo_sorted_cluster_container,
    uint* hit_permutations);

  __global__ void velo_calculate_phi_and_sort(Arguments);

  template<typename T>
  struct velo_calculate_phi_and_sort_t : public DeviceAlgorithm, Arguments {
    constexpr static auto name {"velo_calculate_phi_and_sort_t"};
    decltype(global_function(velo_calculate_phi_and_sort)) function {velo_calculate_phi_and_sort};

    void set_arguments_size(
      ArgumentRefManager<T> manager,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const HostBuffers& host_buffers) const
    {
      set_size<dev_sorted_velo_cluster_container_t>(
        manager, size<dev_velo_cluster_container_t>(manager) / sizeof(uint32_t));
      set_size<dev_hit_permutation_t>(manager, value<host_total_number_of_velo_clusters_t>(manager));
      set_size<dev_hit_phi_t>(manager, value<host_total_number_of_velo_clusters_t>(manager));
    }

    void operator()(
      const ArgumentRefManager<T>& manager,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      cudaStream_t& cuda_stream,
      cudaEvent_t& cuda_generic_event) const
    {
      cudaCheck(
        cudaMemsetAsync(offset<dev_hit_permutation_t>(manager), 0, size<dev_hit_permutation_t>(manager), cuda_stream));

      function(dim3(value<host_number_of_selected_events_t>(manager)), block_dimension(), cuda_stream)(
        Arguments {offset<dev_offsets_estimated_input_size_t>(manager),
                   offset<dev_module_cluster_num_t>(manager),
                   offset<dev_velo_cluster_container_t>(manager),
                   offset<dev_sorted_velo_cluster_container_t>(manager),
                   offset<dev_hit_permutation_t>(manager),
                   offset<dev_hit_phi_t>(manager)});

      // Prints the x values
      // std::vector<uint> a (size<dev_velo_cluster_container_t>(manager) / sizeof(uint));
      // cudaCheck(cudaMemcpy(
      //   a.data(),
      //   offset<dev_velo_cluster_container_t>(manager),
      //   size<dev_velo_cluster_container_t>(manager),
      //   cudaMemcpyDeviceToHost));
      // const auto velo_cluster_container = Velo::Clusters<const uint>{a.data(),
      // value<host_total_number_of_velo_clusters_t>(manager)}; for (uint i = 0; i <
      // value<host_total_number_of_velo_clusters_t>(manager); ++i) {
      //   std::cout << velo_cluster_container.x(i) << ", ";
      // }
      // std::cout << "\n";
    }
  };
} // namespace velo_calculate_phi_and_sort