#include <MEPTools.h>
#include <EstimateInputSize.cuh>

void velo_estimate_input_size::velo_estimate_input_size_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  if (logger::verbosity() >= logger::debug) {
    debug_cout << "# of events = " << first<host_number_of_selected_events_t>(arguments) << std::endl;
  }

  set_size<dev_estimated_input_size_t>(
    arguments, first<host_number_of_selected_events_t>(arguments) * Velo::Constants::n_module_pairs);
  set_size<dev_module_candidate_num_t>(arguments, first<host_number_of_selected_events_t>(arguments));
  set_size<dev_cluster_candidates_t>(arguments, first<host_number_of_cluster_candidates_t>(arguments));
}

void velo_estimate_input_size::velo_estimate_input_size_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  HostBuffers&,
  cudaStream_t& cuda_stream,
  cudaEvent_t&) const
{
  initialize<dev_estimated_input_size_t>(arguments, 0, cuda_stream);
  initialize<dev_module_candidate_num_t>(arguments, 0, cuda_stream);

  if (runtime_options.mep_layout) {
    global_function(velo_estimate_input_size_mep)(
      dim3(first<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(arguments);
  }
  else {
    global_function(velo_estimate_input_size)(
      dim3(first<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(arguments);
  }
}

__device__ void estimate_raw_bank_size(
  unsigned* estimated_input_size,
  uint32_t* cluster_candidates,
  unsigned* event_candidate_num,
  unsigned raw_bank_number,
  VeloRawBank const& raw_bank)
{
  unsigned* estimated_module_pair_size = estimated_input_size + (raw_bank.sensor_index / 8);
  unsigned found_cluster_candidates = 0;
  for (unsigned sp_index = threadIdx.x; sp_index < raw_bank.sp_count; sp_index += blockDim.x) { // Decode sp
    const uint32_t sp_word = raw_bank.sp_word[sp_index];
    const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
    const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
    const uint8_t sp = sp_word & 0xFFU;

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
      assert(current_estimated_module_pair_size < Velo::Constants::max_numhits_in_module_pair);
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

      for (unsigned k = 0; k < raw_bank.sp_count; ++k) {
        const uint32_t other_sp_word = raw_bank.sp_word[k];
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
        const uint32_t candidate = (sp_index << 11) | (raw_bank_number << 3) | candidate_pixel;
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
        const uint32_t candidate = (sp_index << 11) | (raw_bank_number << 3) | candidate_pixel;
        cluster_candidates[current_cluster_candidate] = candidate;
        ++found_cluster_candidates;
      }
    }
  }

  // Add the found cluster candidates
  if (found_cluster_candidates > 0) {
    [[maybe_unused]] const unsigned current_estimated_module_pair_size =
      atomicAdd(estimated_module_pair_size, found_cluster_candidates);
    assert(current_estimated_module_pair_size + found_cluster_candidates < Velo::Constants::max_numhits_in_module_pair);
  }
}

__global__ void velo_estimate_input_size::velo_estimate_input_size(velo_estimate_input_size::Parameters parameters)
{
  const auto event_number = blockIdx.x;
  const auto selected_event_number = parameters.dev_event_list[event_number];

  const char* raw_input = parameters.dev_velo_raw_input + parameters.dev_velo_raw_input_offsets[selected_event_number];
  unsigned* estimated_input_size = parameters.dev_estimated_input_size + event_number * Velo::Constants::n_module_pairs;
  unsigned* event_candidate_num = parameters.dev_module_candidate_num + event_number;
  uint32_t* cluster_candidates = parameters.dev_cluster_candidates + parameters.dev_candidates_offsets[event_number];

  // Read raw event
  const auto raw_event = VeloRawEvent(raw_input);

  for (unsigned raw_bank_number = threadIdx.y; raw_bank_number < raw_event.number_of_raw_banks;
       raw_bank_number += blockDim.y) {
    // Read raw bank
    const auto raw_bank = VeloRawBank(raw_event.payload + raw_event.raw_bank_offset[raw_bank_number]);
    estimate_raw_bank_size(estimated_input_size, cluster_candidates, event_candidate_num, raw_bank_number, raw_bank);
  }
}

__global__ void velo_estimate_input_size::velo_estimate_input_size_mep(velo_estimate_input_size::Parameters parameters)
{
  const unsigned event_number = blockIdx.x;
  const unsigned selected_event_number = parameters.dev_event_list[event_number];

  unsigned* estimated_input_size = parameters.dev_estimated_input_size + event_number * Velo::Constants::n_module_pairs;
  unsigned* event_candidate_num = parameters.dev_module_candidate_num + event_number;
  uint32_t* cluster_candidates = parameters.dev_cluster_candidates + parameters.dev_candidates_offsets[event_number];

  // Read raw event
  auto const number_of_raw_banks = parameters.dev_velo_raw_input_offsets[0];

  for (unsigned raw_bank_number = threadIdx.y; raw_bank_number < number_of_raw_banks; raw_bank_number += blockDim.y) {

    // Create raw bank from MEP layout
    const auto raw_bank = MEP::raw_bank<VeloRawBank>(
      parameters.dev_velo_raw_input, parameters.dev_velo_raw_input_offsets, selected_event_number, raw_bank_number);

    estimate_raw_bank_size(estimated_input_size, cluster_candidates, event_candidate_num, raw_bank_number, raw_bank);
  }
}