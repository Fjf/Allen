/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "HostRoutingBitsWriter.h"
#include "ProgramOptions.h"
#include "HltDecReport.cuh"

INSTANTIATE_ALGORITHM(host_routingbits_writer::host_routingbits_writer_t)

void host_routingbits_writer::host_routingbits_writer_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  // Two words for the routing bits (ODIN + HLT1)
  set_size<host_routingbits_t>(arguments, RoutingBitsDefinition::n_words * first<host_number_of_events_t>(arguments));
}

void host_routingbits_writer::host_routingbits_writer_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  HostBuffers& host_buffers,
  const Allen::Context& context) const
{
  const auto map = m_routingbit_map.get_value().get();

  for (auto const& pair : m_routingbit_map.get_value().get()) {
    debug_cout << "{" << pair.first << ": " << pair.second << "}\n";
  }
  std::map<uint32_t, boost::regex> regex_map;
  for (auto const& [bit, expr] : map) {
    boost::regex rb_regex(expr);
    regex_map[bit] = rb_regex;
  }
  host_routingbits_conf_impl(
    first<host_number_of_events_t>(arguments),
    first<host_number_of_active_lines_t>(arguments),
    data<host_names_of_active_lines_t>(arguments),
    data<host_dec_reports_t>(arguments),
    data<host_routingbits_t>(arguments),
    regex_map);
  // Copy routing bit info to the host buffer
  safe_assign_to_host_buffer<host_routingbits_t>(host_buffers.host_routingbits, arguments, context);
}

void host_routingbits_writer::host_routingbits_conf_impl(
  unsigned host_number_of_events,
  unsigned host_number_of_active_lines,
  char* host_names_of_active_lines,
  unsigned* host_dec_reports,
  unsigned* host_routing_bits,
  const std::map<uint32_t, boost::regex>& routingbit_map)
{
  auto line_names = split_string(static_cast<char const*>(host_names_of_active_lines), ",");


  for (unsigned event = 0; event < host_number_of_events; ++event) {

    unsigned* bits = host_routing_bits + RoutingBitsDefinition::n_words * event;
    unsigned const* dec_reports = host_dec_reports + (2 + host_number_of_active_lines) * event;
    for (auto const& [bit, expr] : routingbit_map) {
      int result = 0;

      for (unsigned line_index = 0; line_index < host_number_of_active_lines; line_index++) {
        HltDecReport dec_report;
        dec_report.setDecReport(dec_reports[2 + line_index]);

        if (!dec_report.getDecision()) continue;
        auto line_name = line_names[line_index];

        if (!boost::regex_match(line_name, expr))
          continue; // only works with OR logic so far. TODO: implement AND logic / * logic
        result = 1;
        debug_cout << "line " << line_name << " fired, setting " << bit << " bit " << std::endl;
      }
      int word = bit / 32;
      if (result) bits[word] |= (0x01UL << (bit - 32 * word));
    }
    if (logger::verbosity() >= logger::debug) {
      debug_cout << " HostRoutingBits: Event n. " << event << ", routing bits: " << bits[0] << "   " << bits[1] << "   "
                 << bits[2] << "   " << bits[3] << std::endl;
    }
  }
}
