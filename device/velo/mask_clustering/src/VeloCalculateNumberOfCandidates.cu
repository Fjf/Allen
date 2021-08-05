/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <MEPTools.h>
#include <VeloCalculateNumberOfCandidates.cuh>

template<bool mep_layout>
__global__ void velo_calculate_number_of_candidates_kernel(
  velo_calculate_number_of_candidates::Parameters parameters,
  const unsigned number_of_events)
{
  for (auto event_index = blockIdx.x * blockDim.x + threadIdx.x; event_index < number_of_events;
       event_index += blockDim.x * gridDim.x) {
    const auto event_number = parameters.dev_event_list[event_index];

    // Read raw event
    unsigned number_of_raw_banks;
    if constexpr (mep_layout) {
      number_of_raw_banks = parameters.dev_velo_raw_input_offsets[0];
    }
    else {
      const char* raw_input = parameters.dev_velo_raw_input + parameters.dev_velo_raw_input_offsets[event_number];
      const auto raw_event = VeloRawEvent(raw_input);
      number_of_raw_banks = raw_event.number_of_raw_banks;
    }

    unsigned number_of_candidates = 0;
    for (unsigned raw_bank_number = 0; raw_bank_number < number_of_raw_banks; ++raw_bank_number) {
      // Read raw bank
      VeloRawBank raw_bank;
      if constexpr (mep_layout) {
        raw_bank = MEP::raw_bank<VeloRawBank>(
          parameters.dev_velo_raw_input, parameters.dev_velo_raw_input_offsets, event_number, raw_bank_number);
      }
      else {
        const char* raw_input = parameters.dev_velo_raw_input + parameters.dev_velo_raw_input_offsets[event_number];
        const auto raw_event = VeloRawEvent(raw_input);
        raw_bank = VeloRawBank(raw_event.payload + raw_event.raw_bank_offset[raw_bank_number]);
      }

      number_of_candidates += raw_bank.sp_count;
    }

    // The maximum number of candidates is two times the number of SPs
    parameters.dev_number_of_candidates[event_number] = 2 * number_of_candidates;
  }
}

void velo_calculate_number_of_candidates::velo_calculate_number_of_candidates_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_number_of_candidates_t>(arguments, first<host_number_of_events_t>(arguments));
}

void velo_calculate_number_of_candidates::velo_calculate_number_of_candidates_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  HostBuffers&,
  const Allen::Context& context) const
{
  initialize<dev_number_of_candidates_t>(arguments, 0, context);

  // Enough blocks to cover all events
  const auto grid_size =
    dim3((size<dev_event_list_t>(arguments) + property<block_dim_x_t>() - 1) / property<block_dim_x_t>());

  global_function(
    runtime_options.mep_layout ? velo_calculate_number_of_candidates_kernel<true> :
                                 velo_calculate_number_of_candidates_kernel<false>)(
    grid_size, dim3(property<block_dim_x_t>().get()), context)(arguments, size<dev_event_list_t>(arguments));
}
