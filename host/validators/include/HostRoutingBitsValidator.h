/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#pragma once

#include "BackendCommon.h"
#include "AlgorithmTypes.cuh"
#include "RoutingBitsDefinition.h"

namespace host_routingbits_validator {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_names_of_lines_t, char) host_names_of_lines;
    HOST_INPUT(host_number_of_active_lines_t, unsigned) host_number_of_active_lines;
    HOST_INPUT(host_dec_reports_t, unsigned) host_dec_reports;
    HOST_INPUT(host_routingbits_t, unsigned) host_routingbits;
    PROPERTY(
      routingbit_map_t,
      "routingbit_map",
      "mapping of expressions to routing bits",
      std::map<std::string, uint32_t>);
  };

  struct host_routingbits_validator_t : public HostAlgorithm, Parameters {
    inline void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const {}

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    Property<routingbit_map_t> m_routingbit_map {this, {}};
  };
} // namespace host_routingbits_validator
