/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "PVLumiCounters.cuh"
#include "LumiSummaryOffsets.h"

INSTANTIATE_ALGORITHM(pv_lumi_counters::pv_lumi_counters_t)

void pv_lumi_counters::pv_lumi_counters_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  // convert the size of lumi summaries to the size of velo counter infos
  set_size<dev_lumi_infos_t>(
    arguments,
    Lumi::Constants::n_pv_counters * first<host_lumi_summaries_size_t>(arguments) / Lumi::Constants::lumi_length);
}

void pv_lumi_counters::pv_lumi_counters_t::init()
{
  std::map<std::string, std::pair<unsigned, unsigned>> schema = property<lumi_counter_schema_t>();

  if (schema.find("VeloVertices") == schema.end()) {
    std::cout << "LumiSummary schema does not use VeloVertices" << std::endl;
  }
  else {
    set_property_value<velo_vertices_offset_t>(schema["VeloVertices"].first);
    set_property_value<velo_vertices_size_t>(schema["VeloVertices"].second);
  }
}

void pv_lumi_counters::pv_lumi_counters_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  // do nothing if no lumi event
  if (first<host_lumi_summaries_size_t>(arguments) == 0) return;

  global_function(pv_lumi_counters)(dim3(first<host_number_of_events_t>(arguments)), property<block_dim_t>(), context)(
    arguments, first<host_number_of_events_t>(arguments));
}

__global__ void pv_lumi_counters::pv_lumi_counters(
  pv_lumi_counters::Parameters parameters,
  const unsigned number_of_events)
{
  for (unsigned event_number = blockIdx.x * blockDim.x + threadIdx.x; event_number < number_of_events;
       event_number += blockDim.x * gridDim.x) {
    unsigned lumi_sum_offset = parameters.dev_lumi_summary_offsets[event_number];

    // skip non-lumi event
    if (lumi_sum_offset == parameters.dev_lumi_summary_offsets[event_number + 1]) continue;

    unsigned cs = parameters.velo_vertices_size;
    unsigned co = parameters.velo_vertices_offset;

    // number of PVs
    unsigned info_offset = lumi_sum_offset / Lumi::Constants::lumi_length;
    parameters.dev_lumi_infos[info_offset].size = static_cast<LHCb::LumiSummaryOffsets::V2::counterOffsets>(cs);
    parameters.dev_lumi_infos[info_offset].offset = static_cast<LHCb::LumiSummaryOffsets::V2::counterOffsets>(co);
    parameters.dev_lumi_infos[info_offset].value = parameters.dev_number_of_pvs[event_number];
  }
}
