/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "VeloDefinitions.cuh"
#include "CalculatePhiAndSort.cuh"
#include "VeloTools.cuh"
#include "Vector.h"
#include <numeric>
#include <algorithm>

using namespace Allen::device;

void velo_calculate_phi_and_sort::velo_calculate_phi_and_sort_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_sorted_velo_cluster_container_t>(arguments, size<dev_velo_cluster_container_t>(arguments));
  set_size<dev_hit_permutation_t>(arguments, first<host_total_number_of_velo_clusters_t>(arguments));
  set_size<dev_hit_phi_t>(arguments, first<host_total_number_of_velo_clusters_t>(arguments));
  set_size<dev_hit_phi_temp_t>(arguments, first<host_total_number_of_velo_clusters_t>(arguments));
}

void velo_calculate_phi_and_sort::velo_calculate_phi_and_sort_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  HostBuffers&,
  const Allen::Context& context) const
{
  initialize<dev_hit_permutation_t>(arguments, 0, context);

  global_function(velo_calculate_phi_and_sort)(
    dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(arguments);

  if (property<verbosity_t>() >= logger::debug) {
    info_cout << "VELO clusters after velo_calculate_phi_and_sort:\n";
    print_velo_clusters<
      dev_sorted_velo_cluster_container_t,
      dev_offsets_estimated_input_size_t,
      dev_module_cluster_num_t,
      host_total_number_of_velo_clusters_t,
      host_number_of_events_t>(arguments);
  }
}

/**
 * @brief Track forwarding algorithm based on triplet finding
 */
__global__ void velo_calculate_phi_and_sort::velo_calculate_phi_and_sort(
  velo_calculate_phi_and_sort::Parameters parameters)
{
  /* Data initialization */
  // Each event is treated with two blocks, one for each side.
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const unsigned number_of_events = parameters.dev_number_of_events[0];

  // Pointers to data within the event
  const unsigned total_estimated_number_of_clusters =
    parameters.dev_offsets_estimated_input_size[Velo::Constants::n_module_pairs * number_of_events];
  const unsigned* module_pair_hit_start =
    parameters.dev_offsets_estimated_input_size + event_number * Velo::Constants::n_module_pairs;
  const unsigned* module_pair_hit_num =
    parameters.dev_module_cluster_num + event_number * Velo::Constants::n_module_pairs;

  const auto velo_cluster_container = parameters.dev_velo_clusters[event_number];
  auto velo_sorted_cluster_container =
    Velo::Clusters {parameters.dev_sorted_velo_cluster_container, total_estimated_number_of_clusters};

  const unsigned event_hit_start = module_pair_hit_start[0];
  const unsigned event_number_of_hits = module_pair_hit_start[Velo::Constants::n_module_pairs] - event_hit_start;

  // Calculate phi and populate hit_permutations
  calculate_phi(
    parameters.dev_hit_phi_temp,
    module_pair_hit_start,
    module_pair_hit_num,
    velo_cluster_container,
    parameters.dev_hit_phi,
    parameters.dev_hit_permutation);

  // Due to phi RAW
  __syncthreads();

  // Sort by phi
  sort_by_phi(
    event_hit_start,
    event_number_of_hits,
    velo_cluster_container,
    velo_sorted_cluster_container,
    parameters.dev_hit_permutation);
}

/**
 * @brief Calculates a phi side
 */
__device__ void velo_calculate_phi_and_sort::calculate_phi(
  int16_t* dev_hit_phi_temp,
  const unsigned* module_pair_hit_start,
  const unsigned* module_pair_hit_num,
  const Velo::Clusters& velo_cluster_container,
  int16_t* hit_phis,
  unsigned* hit_permutations)
{
  for (unsigned module_pair = 0; module_pair < Velo::Constants::n_module_pairs; ++module_pair) {
    const auto hit_start = module_pair_hit_start[module_pair];
    const auto hit_num = module_pair_hit_num[module_pair];

    // Calculate phis
    for (unsigned hit_rel_id = threadIdx.x; hit_rel_id < hit_num; hit_rel_id += blockDim.x) {
      const auto hit_index = hit_start + hit_rel_id;
      const auto hit_phi_int = hit_phi_16(velo_cluster_container.x(hit_index), velo_cluster_container.y(hit_index));
      dev_hit_phi_temp[hit_index] = hit_phi_int;
    }

    // dev_hit_phi_temp
    __syncthreads();

    // Find the permutations given the phis in dev_hit_phi_temp
    for (unsigned hit_rel_id = threadIdx.x; hit_rel_id < hit_num; hit_rel_id += blockDim.x) {
      const auto hit_index = hit_start + hit_rel_id;
      const auto phi = dev_hit_phi_temp[hit_index];

      // Find out local position
      unsigned position = 0;
      for (unsigned j = 0; j < hit_num; ++j) {
        const auto other_phi = dev_hit_phi_temp[hit_start + j];
        // Stable sorting
        position += phi > other_phi || (phi == other_phi && hit_rel_id > j);
      }

      // Store it in hit permutations and in hit_phis, already ordered
      const auto global_position = hit_start + position;
      hit_permutations[global_position] = hit_index;
      hit_phis[global_position] = phi;
    }

    // dev_hit_phi_temp
    __syncthreads();
  }
}

/**
 * @brief Sorts all VELO decoded data by phi onto another container.
 */
__device__ void velo_calculate_phi_and_sort::sort_by_phi(
  const unsigned event_hit_start,
  const unsigned event_number_of_hits,
  const Velo::Clusters& velo_cluster_container,
  Velo::Clusters& velo_sorted_cluster_container,
  unsigned* hit_permutations)
{
  for (unsigned i = threadIdx.x; i < event_number_of_hits; i += blockDim.x) {
    const auto hit_index_global = hit_permutations[event_hit_start + i];
    velo_sorted_cluster_container.set_x(event_hit_start + i, velo_cluster_container.x(hit_index_global));
    velo_sorted_cluster_container.set_y(event_hit_start + i, velo_cluster_container.y(hit_index_global));
    velo_sorted_cluster_container.set_z(event_hit_start + i, velo_cluster_container.z(hit_index_global));
  }

  for (unsigned i = threadIdx.x; i < event_number_of_hits; i += blockDim.x) {
    const auto hit_index_global = hit_permutations[event_hit_start + i];
    velo_sorted_cluster_container.set_id(event_hit_start + i, velo_cluster_container.id(hit_index_global));
  }
}
