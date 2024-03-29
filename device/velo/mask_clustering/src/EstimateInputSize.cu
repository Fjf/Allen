/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <MEPTools.h>
#include <EstimateInputSize.cuh>

INSTANTIATE_ALGORITHM(velo_estimate_input_size::velo_estimate_input_size_t)

template<int decoding_version>
__device__ void estimate_raw_bank_size(
  unsigned* estimated_input_size,
  uint32_t* cluster_candidates,
  unsigned* event_candidate_num,
  Velo::VeloRawBank<decoding_version> const& raw_bank)
{
  // both sensors in the same module so sensor_index0/8 covers both module offsets
  unsigned* estimated_module_pair_size;
  uint32_t n_sp;
  if constexpr (decoding_version == 2 || decoding_version == 3) {
    estimated_module_pair_size = estimated_input_size + (raw_bank.sensor_pair() / 8);
    n_sp = raw_bank.count;
  }
  else {
    estimated_module_pair_size = estimated_input_size + (raw_bank.sensor_index0() / 8);
    n_sp = raw_bank.size / 4;
  }
  unsigned found_cluster_candidates = 0;
  for (unsigned sp_index = threadIdx.x; sp_index < n_sp; sp_index += blockDim.x) { // Decode sp
    const uint32_t sp_word = raw_bank.word[sp_index];
    const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
    const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
    const uint8_t sp = sp_word & 0xFFU;

    uint32_t sensor_number;
    if constexpr (decoding_version == 2 || decoding_version == 3) {
      sensor_number = raw_bank.sensor_pair();
    }
    else {
      sensor_number = (raw_bank.sensor_index0() | ((sp_word >> 23) & 0x1));
    }

    if (no_sp_neighbours) {
      // The SP does not have any neighbours
      // The problem is as simple as a lookup pattern
      // It can be implemented in two operations

      // Pattern 0:
      // (x  x)
      //  o  o
      // (x  x
      //  x  x)
      //
      // Note: Pixel order in sp
      // 0x08 | 0x80
      // 0x04 | 0x40
      // 0x02 | 0x20
      // 0x01 | 0x10
      const bool pattern_0 = (sp & 0x88) && !(sp & 0x44) && (sp & 0x33);

      // Pattern 1:
      // (x  x
      //  x  x)
      //  o  o
      // (x  x)
      const bool pattern_1 = (sp & 0xCC) && !(sp & 0x22) && (sp & 0x11);
      const unsigned number_of_clusters = (pattern_0 | pattern_1) ? 2 : 1;

      // Add the found clusters
      [[maybe_unused]] const unsigned current_estimated_module_pair_size =
        atomicAdd(estimated_module_pair_size, number_of_clusters);
    }
    else {
      // Find candidates that follow this condition:
      // For pixel o, all pixels x should *not* be populated
      // x x
      // o x
      //   x

      // Load required neighbouring pixels in order to check the condition
      // x x x
      // o o x
      // o o x
      // o o x
      // o o x
      //   x x
      //
      // Use an int for storing and calculating
      // Bit order
      //
      // 4 10 16
      // 3  9 15
      // 2  8 14
      // 1  7 13
      // 0  6 12
      //    5 11
      //
      // Bit masks
      //
      // 0x10 0x0400 0x010000
      // 0x08 0x0200   0x8000
      // 0x04 0x0100   0x4000
      // 0x02   0x80   0x2000
      // 0x01   0x40   0x1000
      //        0x20   0x0800
      uint32_t pixels = (sp & 0x0F) | ((sp & 0xF0) << 2);

      // Current row and col
      const uint32_t sp_row = sp_addr & 0x3FU;
      const uint32_t sp_col = sp_addr >> 6;

      for (unsigned k = 0; k < n_sp; ++k) {
        const uint32_t other_sp_word = raw_bank.word[k];
        const uint32_t other_no_sp_neighbours = sp_word & 0x80000000U;

        if (!other_no_sp_neighbours) {
          const uint32_t other_sp_addr = (other_sp_word & 0x007FFF00U) >> 8;
          const uint32_t other_sp_row = other_sp_addr & 0x3FU;
          const uint32_t other_sp_col = (other_sp_addr >> 6);
          const uint8_t other_sp = other_sp_word & 0xFFU;

          // Populate pixels
          // Note: Pixel order in sp
          // 0x08 | 0x80
          // 0x04 | 0x40
          // 0x02 | 0x20
          // 0x01 | 0x10
          const bool is_top = other_sp_row == (sp_row + 1) && other_sp_col == sp_col;
          const bool is_top_right = other_sp_row == (sp_row + 1) && other_sp_col == (sp_col + 1);
          const bool is_right = other_sp_row == sp_row && other_sp_col == (sp_col + 1);
          const bool is_right_bottom = other_sp_row == (sp_row - 1) && other_sp_col == (sp_col + 1);
          const bool is_bottom = other_sp_row == (sp_row - 1) && other_sp_col == sp_col;

          if (is_top || is_top_right || is_right || is_right_bottom || is_bottom) {
            pixels |= is_top * (((other_sp & 0x01) | ((other_sp & 0x10) << 2)) << 4);
            pixels |= is_top_right * ((other_sp & 0x01) << 16);
            pixels |= is_right * ((other_sp & 0x0F) << 12);
            pixels |= is_right_bottom * ((other_sp & 0x08) << 8);
            pixels |= is_bottom * ((other_sp & 0x80) >> 2);
          }
        }
      }

      // 16 1024 65536
      //  8  512 32768
      //  4  256 16384
      //  2  128  8192
      //  1   64  4096
      //      32  2048
      //
      // 5 11 17
      // 4 10 16
      // 3  9 15
      // 2  8 14
      // 1  7 13
      //    6 12
      //
      // Look up pattern
      // x x
      // o x
      //   x
      //
      const uint32_t sp_inside_pixel = pixels & 0x3CF;
      const uint32_t mask =
        (sp_inside_pixel << 1) | (sp_inside_pixel << 5) | (sp_inside_pixel << 6) | (sp_inside_pixel << 7);

      const uint32_t working_cluster = mask & (~pixels);
      const uint32_t candidates_temp =
        (working_cluster >> 1) & (working_cluster >> 5) & (working_cluster >> 6) & (working_cluster >> 7);

      const uint32_t candidates = candidates_temp & pixels;
      const uint32_t candidates_consolidated = (candidates & 0x0F) | ((candidates >> 2) & 0xF0);

      const auto first_candidate = candidates_consolidated & 0x33;
      const auto second_candidate = candidates_consolidated & 0xCC;

      // Add candidates 0, 1, 4, 5
      // Only one of those candidates can be flagged at a time
      if (first_candidate) {
        // Verify candidates are correctly created
        assert(__popc(first_candidate) <= 1);

        // Decode the candidate number (ie. find out the active bit)
        const auto candidate_pixel = __clz(first_candidate) - 24;

        auto current_cluster_candidate = atomicAdd(event_candidate_num, 1);
        uint32_t candidate = (sp_index << 11) | (sensor_number << 3) | candidate_pixel;
        cluster_candidates[current_cluster_candidate] = candidate;
        ++found_cluster_candidates;
      }

      // Add candidates 2, 3, 6, 7
      // Only one of those candidates can be flagged at a time
      if (second_candidate) {
        assert(__popc(second_candidate) <= 1);

        // Decode the candidate number (ie. find out the active bit)
        const auto candidate_pixel = __clz(second_candidate) - 24;

        auto current_cluster_candidate = atomicAdd(event_candidate_num, 1);
        uint32_t candidate = (sp_index << 11) | (sensor_number << 3) | candidate_pixel;
        cluster_candidates[current_cluster_candidate] = candidate;
        ++found_cluster_candidates;
      }
    }
  }

  // Add the found cluster candidates
  if (found_cluster_candidates > 0) {
    [[maybe_unused]] const unsigned current_estimated_module_pair_size =
      atomicAdd(estimated_module_pair_size, found_cluster_candidates);
  }
}

template<int decoding_version, bool mep_layout>
__global__ void velo_estimate_input_size_kernel(
  velo_estimate_input_size::Parameters parameters,
  unsigned const event_start)
{
  const auto event_number = parameters.dev_event_list[blockIdx.x];
  unsigned* estimated_input_size = parameters.dev_estimated_input_size + event_number * Velo::Constants::n_module_pairs;
  unsigned* event_candidate_num = parameters.dev_module_candidate_num + event_number;
  uint32_t* cluster_candidates = parameters.dev_cluster_candidates + parameters.dev_candidates_offsets[event_number];

  // The event number is with respect to the start of the batch, but
  // the raw data is not organised like that, so the event start is
  // needed.
  const auto velo_raw_event = Velo::RawEvent<decoding_version, mep_layout> {parameters.dev_velo_raw_input,
                                                                            parameters.dev_velo_raw_input_offsets,
                                                                            parameters.dev_velo_raw_input_sizes,
                                                                            parameters.dev_velo_raw_input_types,
                                                                            event_number + event_start};
  for (unsigned raw_bank_number = threadIdx.y; raw_bank_number < velo_raw_event.number_of_raw_banks();
       raw_bank_number += blockDim.y) {
    const auto raw_bank = velo_raw_event.raw_bank(raw_bank_number);

    if (raw_bank.type != LHCb::RawBank::VP && raw_bank.type != LHCb::RawBank::Velo) continue;
    estimate_raw_bank_size<decoding_version>(estimated_input_size, cluster_candidates, event_candidate_num, raw_bank);
  }
}

void velo_estimate_input_size::velo_estimate_input_size_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_estimated_input_size_t>(
    arguments, first<host_number_of_events_t>(arguments) * Velo::Constants::n_module_pairs);
  set_size<dev_module_candidate_num_t>(arguments, first<host_number_of_events_t>(arguments));
  set_size<dev_cluster_candidates_t>(arguments, first<host_number_of_cluster_candidates_t>(arguments));
}

void velo_estimate_input_size::velo_estimate_input_size_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_estimated_input_size_t>(arguments, 0, context);
  Allen::memset_async<dev_module_candidate_num_t>(arguments, 0, context);

  auto const bank_version = first<host_raw_bank_version_t>(arguments);

  if (bank_version < 0) return; // no VP banks present in data

  // Ensure the bank version is supported
  if (bank_version != 2 && bank_version != 3 && bank_version != 4) {
    throw StrException("Velo SP bank version not supported (" + std::to_string(bank_version) + ")");
  }

  auto kernel_fn = (bank_version == 2) ?
                     (runtime_options.mep_layout ? global_function(velo_estimate_input_size_kernel<2, true>) :
                                                   global_function(velo_estimate_input_size_kernel<2, false>)) :
                     (bank_version == 3) ?
                     (runtime_options.mep_layout ? global_function(velo_estimate_input_size_kernel<3, true>) :
                                                   global_function(velo_estimate_input_size_kernel<3, false>)) :
                     (runtime_options.mep_layout ? global_function(velo_estimate_input_size_kernel<4, true>) :
                                                   global_function(velo_estimate_input_size_kernel<4, false>));

  kernel_fn(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(
    arguments, std::get<0>(runtime_options.event_interval));
}
