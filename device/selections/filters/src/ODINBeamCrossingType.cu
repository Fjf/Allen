/************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration      *
\************************************************************************/
#include "ODINBeamCrossingType.cuh"
#include "Event/ODIN.h"
#include "ODINBank.cuh"

INSTANTIATE_ALGORITHM( odin_beamcrossingtype::odin_beamcrossingtype_t)

void odin_beamcrossingtype::odin_beamcrossingtype_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{

  set_size<typename Parameters::dev_number_of_selected_events_t>(arguments, 1);
  set_size<typename Parameters::host_number_of_selected_events_t>(arguments, 1);

  set_size<dev_event_list_output_t>(arguments, size<dev_event_list_t>(arguments));
}

void odin_beamcrossingtype::odin_beamcrossingtype_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  HostBuffers&,
  const Allen::Context& context) const
{

  //initialize<host_event_list_output_t>(arguments, 0, context);
  initialize<typename Parameters::dev_event_list_output_t>(arguments, 0, context);
  //initialize<dev_event_decisions_t>(arguments, 0, context);
  initialize<typename Parameters::dev_number_of_selected_events_t>(arguments, 0, context);
  initialize<typename Parameters::host_number_of_selected_events_t>(arguments, 0, context);

  global_function(odin_beamcrossingtype)(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(arguments);

  Allen::copy<typename Parameters::host_number_of_selected_events_t, typename Parameters::dev_number_of_selected_events_t>(arguments, context);
  reduce_size<dev_event_list_output_t>(arguments, first<host_number_of_selected_events_t>(arguments));
}

__global__ void odin_beamcrossingtype::odin_beamcrossingtype(odin_beamcrossingtype::Parameters parameters)
{

  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const unsigned* event_odin_data = nullptr;
  if (parameters.dev_mep_layout[0]) {
    event_odin_data =
      odin_data_mep_t::data(parameters.dev_odin_raw_input, parameters.dev_odin_raw_input_offsets, event_number);
  }
  else {
    event_odin_data =
      odin_data_t::data(parameters.dev_odin_raw_input, parameters.dev_odin_raw_input_offsets, event_number);
  }

  __shared__ int event_decision;
  if ( threadIdx.x == 0 ) event_decision = 0;
  __syncthreads();

  const unsigned bxt = static_cast<unsigned int>(LHCb::ODIN({event_odin_data, 10}).bunchCrossingType());
  const bool dec = bxt == parameters.beam_crossing_type;
  if (dec) atomicOr(&event_decision, dec);

  if (threadIdx.x == 0 && event_decision) {
    const auto current_event = atomicAdd(parameters.dev_number_of_selected_events.get(), 1);
    parameters.dev_event_list_output[current_event] = mask_t { event_number };
  }
}
