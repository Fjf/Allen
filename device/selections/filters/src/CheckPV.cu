/************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration      *
\************************************************************************/
#include "CheckPV.cuh"

void check_pvs::check_pvs_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_number_of_selected_events_t>(arguments, 1);
  set_size<host_number_of_selected_events_t>(arguments, 1);

  set_size<dev_event_list_output_t>( arguments, size<dev_event_list_t>(arguments));
  set_size<host_event_list_output_t>( arguments, size<dev_event_list_t>(arguments));

  set_size<dev_event_decisions_t>(arguments, size<dev_event_list_t>(arguments));
}

void check_pvs::check_pvs_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  HostBuffers&,
  const Allen::Context& context) const
{

  initialize<dev_event_list_output_t>(arguments, 0, context);
  initialize<dev_event_decisions_t>(arguments, 0, context);
  initialize<host_number_of_selected_events_t>(arguments, 0, context);
  initialize<dev_number_of_selected_events_t>(arguments, 0, context);

  global_function(check_pvs)(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(arguments);

  copy<host_number_of_selected_events_t, dev_number_of_selected_events_t>(arguments, context);
  Allen::synchronize(context);

  reduce_size<dev_event_list_output_t>(arguments, first<host_number_of_selected_events_t>(arguments));
  copy<host_event_list_output_t, dev_event_list_output_t>(arguments, context);

  Allen::synchronize(context);
  reduce_size<host_event_list_output_t>(arguments, first<host_number_of_selected_events_t>(arguments));
}

__global__ void check_pvs::check_pvs(check_pvs::Parameters parameters)
{

  const auto event_number = parameters.dev_event_list[blockIdx.x];
  const PV::Vertex* vertices = parameters.dev_multi_final_vertices + event_number * PV::max_number_vertices;

  unsigned* event_decision = parameters.dev_event_decisions.get() + blockIdx.x;

  for (unsigned i = threadIdx.x; i < parameters.dev_number_of_multi_final_vertices[event_number]; i += blockDim.x) {
    const auto& pv = vertices[i];
    const bool dec = pv.position.z >= parameters.minZ and pv.position.z < parameters.maxZ;
    if (dec) atomicOr( event_decision, dec );
  }

  __syncthreads();

  if ( threadIdx.x == 0 && *event_decision ) {
    const auto current_event = atomicAdd(parameters.dev_number_of_selected_events.get(), 1);
    parameters.dev_event_list_output[current_event] = event_number;
  }
}
