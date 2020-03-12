#include "FillCandidates.cuh"
#include "ClusteringDefinitions.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "VeloTools.cuh"
#include "BinarySearch.cuh"
#include <cassert>
#include <cstdio>
#include <tuple>

/**
 * @brief Fills candidates according to the phi window,
 *        with no restriction on the number of candidates.
 *        Returns a tuple<int, int>, with the structure:
 *
 *        * first candidate
 *        * size of window
 */
__device__ std::tuple<int, int> candidate_binary_search(
  const float* hit_Phis,
  const int module_hit_start,
  const int module_number_of_hits,
  const float h1_phi,
  const float phi_window)
{
  // Do a binary search for the first candidate
  const auto first_candidate =
    binary_search_leftmost(hit_Phis + module_hit_start, module_number_of_hits, h1_phi - phi_window);

  if (
    first_candidate == module_number_of_hits ||
    fabsf(static_cast<float>(hit_Phis[module_hit_start + first_candidate]) - h1_phi) > phi_window) {
    return {-1, 0};
  }
  else {
    // Find number of candidates with a second binary search
    const auto number_of_candidates = binary_search_leftmost(
      hit_Phis + module_hit_start + first_candidate,
      module_number_of_hits - first_candidate,
      h1_phi + phi_window);
    return {first_candidate, number_of_candidates};
  }
}

/**
 * @brief Finds candidates with a maximum number of candidates to the
 *        left and right wrt the h1 phi.
 */
__device__ std::tuple<int, int> candidate_capped_search(
  const float* hit_Phis,
  const int module_hit_start,
  const int module_number_of_hits,
  const float h1_phi,
  const float phi_window,
  const int maximum_candidates_side)
{
  int first_candidate = -1;
  int number_of_candidates = 0;

  if (module_number_of_hits > 0) {
    // Do a binary search for h0 candidates
    const auto candidate_position =
      binary_search_leftmost(hit_Phis + module_hit_start, module_number_of_hits, h1_phi);

    if (
      candidate_position<module_number_of_hits&& static_cast<float>(hit_Phis[module_hit_start + candidate_position])>(
        h1_phi - phi_window) &&
      static_cast<float>(hit_Phis[module_hit_start + candidate_position]) < (h1_phi + phi_window)) {

      first_candidate = candidate_position;
      number_of_candidates = 1;

      // Find a maximum of candidates to both sides
      for (int i = 0; i < maximum_candidates_side; ++i) {
        const auto current_left_candidate = candidate_position - i - 1;
        if (
          current_left_candidate >= 0 &&
          static_cast<float>(hit_Phis[module_hit_start + current_left_candidate]) > (h1_phi - phi_window)) {
          first_candidate = current_left_candidate;
          number_of_candidates++;
        }

        const auto current_right_candidate = candidate_position + i + 1;
        if (
          current_right_candidate < module_number_of_hits &&
          static_cast<float>(hit_Phis[module_hit_start + current_right_candidate]) < (h1_phi + phi_window)) {
          number_of_candidates++;
        }
      }
    }
  }

  return {first_candidate, number_of_candidates};
}

/**
 * @brief Implementation of FillCandidates for a single module.
 */
__device__ void fill_candidates_impl(
  short* h0_candidates,
  short* h2_candidates,
  const uint* module_hitStarts,
  const uint* module_hitNums,
  Velo::ConstClusters& velo_cluster_container,
  const float* hit_Phis,
  const uint hit_offset,
  const float phi_extrapolation_base,
  const float phi_extrapolation_coef)
{
  // Notation is m0, m1, m2 in reverse order for each module
  // A hit in those is h0, h1, h2 respectively

  // Assign a h1 to each threadIdx.x
  const auto module_index = blockIdx.y + 2; // 48 blocks y
  const auto m1_hitNums = module_hitNums[module_index];
  for (uint h1_rel_index = threadIdx.x; h1_rel_index < m1_hitNums; h1_rel_index += blockDim.x) {
    // Find for module module_index, hit h1_rel_index the candidates
    const auto m0_hitStarts = module_hitStarts[module_index + 2];
    const auto m2_hitStarts = module_hitStarts[module_index - 2];
    const auto m0_hitNums = module_hitNums[module_index + 2];
    const auto m2_hitNums = module_hitNums[module_index - 2];

    const auto h1_index = module_hitStarts[module_index] + h1_rel_index;

    // Calculate phi limits
    const float h1_phi = hit_Phis[h1_index];
    const Velo::HitBase h1 {
      velo_cluster_container.x(h1_index), velo_cluster_container.y(h1_index), velo_cluster_container.z(h1_index)};
    const auto phi_window = phi_extrapolation_base + fabsf(h1.z) * phi_extrapolation_coef;

    const auto found_h0_candidates = candidate_binary_search(hit_Phis, m0_hitStarts, m0_hitNums, h1_phi, phi_window);
    const auto found_h2_candidates = candidate_binary_search(hit_Phis, m2_hitStarts, m2_hitNums, h1_phi, phi_window);

    // // Check if there is at least a compatible triplet
    // constexpr float max_scatter = 0.1f;
    // bool found = false;

    // for (uint i = 0; !found && i < std::get<1>(found_h0_candidates); ++i) {
    //   const auto h0_index = m0_hitStarts + std::get<0>(found_h0_candidates) + i;
    //   const Velo::HitBase h0 {velo_cluster_container.x(h0_index),
    //                           velo_cluster_container.y(h0_index),
    //                           velo_cluster_container.z(h0_index)};

    //   const auto partial_tz = 1.f / (h1.z - h0.z);

    //   for (uint j = 0; !found && j < std::get<1>(found_h2_candidates); ++j) {
    //     const auto h2_index = m2_hitStarts + std::get<0>(found_h2_candidates) + j;
    //     const Velo::HitBase h2 {velo_cluster_container.x(h2_index),
    //                             velo_cluster_container.y(h2_index),
    //                             velo_cluster_container.z(h2_index)};

    //     // Calculate prediction
    //     const auto z2_tz = (h2.z - h0.z) * partial_tz;
    //     const auto x = h0.x + (h1.x - h0.x) * z2_tz;
    //     const auto y = h0.y + (h1.y - h0.y) * z2_tz;
    //     const auto dx = x - h2.x;
    //     const auto dy = y - h2.y;

    //     // Calculate fit
    //     const auto scatter = (dx * dx) + (dy * dy);

    //     if (scatter < max_scatter) {
    //       found = true;
    //     }
    //   }
    // }

    // if (found) {
    // h0_candidates[2 * h1_index] = std::get<0>(found_h0_candidates) + m0_hitStarts - hit_offset;
    // h0_candidates[2 * h1_index + 1] = std::get<1>(found_h0_candidates);
    // h2_candidates[2 * h1_index] = std::get<0>(found_h2_candidates) + m2_hitStarts - hit_offset;
    // h2_candidates[2 * h1_index + 1] = std::get<1>(found_h2_candidates);
    // }

    // constexpr auto lumi_region = 130.f;
    // const auto r1 = sqrtf(h1.x * h1.x + h1.y * h1.y);
    // uint first_h0_candidate = 0, first_h2_candidate = 0;

    // for (uint i = 0; i < std::get<1>(found_h0_candidates); ++i) {
    //   const auto h0_index = m0_hitStarts + std::get<0>(found_h0_candidates) + i;
    //   const Velo::HitBase h0 {velo_cluster_container.x(h0_index),
    //                           velo_cluster_container.y(h0_index),
    //                           velo_cluster_container.z(h0_index)};

    //   const auto r0 = sqrtf(h0.x * h0.x + h0.y * h0.y);
    //   const auto c0 = 0.8f * r0 * (h1.z - lumi_region) / (h0.z - lumi_region);
    //   const auto c1 = 1.2f * r0 * (h1.z + lumi_region) / (h0.z + lumi_region);

    //   const bool keep = (fabsf(h1.z) > lumi_region && r1 > c0 && r1 < c1) ||
    //     (fabsf(h1.z) < lumi_region && (r1 > c0 || r1 < c1));

    //   if (keep) {
    //     break;
    //   } else {
    //     first_h0_candidate++;
    //   }
    // }

    // for (uint i = 0; i < std::get<1>(found_h2_candidates); ++i) {
    //   const auto h2_index = m2_hitStarts + std::get<0>(found_h2_candidates) + i;
    //   const Velo::HitBase h2 {velo_cluster_container.x(h2_index),
    //                           velo_cluster_container.y(h2_index),
    //                           velo_cluster_container.z(h2_index)};

    //   const auto r2 = sqrtf(h2.x * h2.x + h2.y * h2.y);
    //   const auto c1 = 0.8f * r1 * (h2.z - lumi_region) / (h1.z - lumi_region);
    //   const auto c2 = 1.2f * r1 * (h2.z + lumi_region) / (h1.z + lumi_region);

    //   const bool keep = (fabsf(h2.z) > lumi_region && r2 > c1 && r2 < c2) ||
    //     (fabsf(h2.z) < lumi_region && (r2 > c1 || r2 < c2));

    //   if (keep) {
    //     break;
    //   } else {
    //     first_h2_candidate++;
    //   }
    // }

    // h0_candidates[2 * h1_index] = std::get<0>(found_h0_candidates) + first_h0_candidate + m0_hitStarts - hit_offset;
    // h0_candidates[2 * h1_index + 1] = std::get<1>(found_h0_candidates) - first_h0_candidate;

    // h2_candidates[2 * h1_index] = std::get<0>(found_h2_candidates) + first_h2_candidate + m2_hitStarts - hit_offset;
    // h2_candidates[2 * h1_index + 1] = std::get<1>(found_h2_candidates) - first_h2_candidate;

    if (std::get<1>(found_h0_candidates) && std::get<1>(found_h2_candidates)) {
      h0_candidates[2 * h1_index] = std::get<0>(found_h0_candidates) + m0_hitStarts - hit_offset;
      h0_candidates[2 * h1_index + 1] = std::get<1>(found_h0_candidates);
      h2_candidates[2 * h1_index] = std::get<0>(found_h2_candidates) + m2_hitStarts - hit_offset;
      h2_candidates[2 * h1_index + 1] = std::get<1>(found_h2_candidates);
    }
  }
}

/**
 * @brief Fills the first candidate and size for each hit in a middle module.
 * @details Considering hits in consecutive modules m0, m1 and m2, for every
 *          hit h1 in module m1, the following structure is created:
 *
 *          * h0 first candidate
 *          * h0 number of candidates
 *          * h2 first candidate
 *          * h2 number of candidates
 *
 *          These candidates will be then iterated in the seeding step of Sbt.
 */
__global__ void velo_fill_candidates::velo_fill_candidates(velo_fill_candidates::Parameters parameters)
{
  const uint event_number = blockIdx.x;
  const uint number_of_events = gridDim.x;

  // Pointers to data within the event
  const uint total_estimated_number_of_clusters =
    parameters.dev_offsets_estimated_input_size[Velo::Constants::n_modules * number_of_events];
  const uint* module_hitStarts =
    parameters.dev_offsets_estimated_input_size + event_number * Velo::Constants::n_modules;
  const uint* module_hitNums = parameters.dev_module_cluster_num + event_number * Velo::Constants::n_modules;
  const auto hit_offset = module_hitStarts[0];

  const auto velo_cluster_container =
    Velo::ConstClusters {parameters.dev_sorted_velo_cluster_container, total_estimated_number_of_clusters};

  fill_candidates_impl(
    parameters.dev_h0_candidates,
    parameters.dev_h2_candidates,
    module_hitStarts,
    module_hitNums,
    velo_cluster_container,
    parameters.dev_hit_phi,
    hit_offset,
    parameters.phi_extrapolation_base,
    parameters.phi_extrapolation_coef);
}
