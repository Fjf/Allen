/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "HostUTGEC.h"

INSTANTIATE_ALGORITHM(host_ut_gec::host_ut_gec_t)

void host_ut_gec::host_ut_gec_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  auto number_of_events = first<host_number_of_events_t>(arguments);

  set_size<host_number_of_selected_events_t>(arguments, 1);
  set_size<host_output_event_list_t>(arguments, number_of_events);
  set_size<dev_event_list_output_t>(arguments, number_of_events);
}

void host_ut_gec::host_ut_gec_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{
  // Do the host global event cut
  host_function(runtime_options.mep_layout ? host_ut_gec<true> : host_ut_gec<false>)(
    arguments, std::get<0>(runtime_options.event_interval));

  // Reduce the size of the event lists to the selected events
  reduce_size<host_output_event_list_t>(arguments, first<host_number_of_selected_events_t>(arguments));
  reduce_size<dev_event_list_output_t>(arguments, first<host_number_of_selected_events_t>(arguments));

  // Copy data to the device
  Allen::copy_async<dev_event_list_output_t, host_output_event_list_t>(arguments, context);
}
