/*****************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "EventListInversion.cuh"

INSTANTIATE_ALGORITHM(event_list_inversion::event_list_inversion_t)

void event_list_inversion::event_list_inversion_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions& runtime_options,
  const Constants&) const
{
  const auto number_of_events =
    std::get<1>(runtime_options.event_interval) - std::get<0>(runtime_options.event_interval);

  set_size<dev_event_list_output_t>(arguments, number_of_events);
  set_size<host_event_list_output_t>(arguments, number_of_events);
  set_size<host_event_list_t>(arguments, size<dev_event_list_input_t>(arguments));
}

void event_list_inversion::event_list_inversion_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{
  const auto number_of_events =
    std::get<1>(runtime_options.event_interval) - std::get<0>(runtime_options.event_interval);

  Allen::copy<host_event_list_t, dev_event_list_input_t>(arguments, context);

  // Make a list that contains all events not in host_event_list_t
  // Do not assume any order in the elements of host_event_list_t
  unsigned output_number_of_events = 0;
  for (unsigned i = 0; i < number_of_events; ++i) {
    bool exists = false;
    for (unsigned j = 0; j < size<host_event_list_t>(arguments); ++j) {
      if (i == data<host_event_list_t>(arguments)[j]) {
        exists = true;
        break;
      }
    }

    if (!exists) {
      data<host_event_list_output_t>(arguments)[output_number_of_events] = i;
      ++output_number_of_events;
    }
  }

  // Adjust the size of the output event lists
  reduce_size<host_event_list_output_t>(arguments, output_number_of_events);
  reduce_size<dev_event_list_output_t>(arguments, output_number_of_events);

  // Copy the event list to the device
  Allen::copy_async<dev_event_list_output_t, host_event_list_output_t>(arguments, context);

  if (property<verbosity_t>() >= logger::debug) {
    printf("List inversion:\n From list: ");
    for (size_t i = 0; i < size<host_event_list_t>(arguments); ++i) {
      printf("%i, ", data<host_event_list_t>(arguments)[i]);
    }
    printf("\n To list: ");
    for (size_t i = 0; i < size<host_event_list_output_t>(arguments); ++i) {
      printf("%i, ", data<host_event_list_output_t>(arguments)[i]);
    }
    printf("\n");
  }
}
