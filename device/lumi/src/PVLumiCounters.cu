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
#include <Event/LumiSummaryOffsets_V2.h>

INSTANTIATE_ALGORITHM(pv_lumi_counters::pv_lumi_counters_t)

void pv_lumi_counters::pv_lumi_counters_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  // convert the size of lumi summaries to the size of velo counter infos
  set_size<dev_lumi_infos_t>(
    arguments,
    Lumi::Constants::n_pv_counters * first<host_lumi_summaries_size_t>(arguments) / Lumi::Constants::lumi_length);
}

void pv_lumi_counters::pv_lumi_counters_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  HostBuffers&,
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

    // number of PVs
    unsigned info_offset = lumi_sum_offset / Lumi::Constants::lumi_length;
    parameters.dev_lumi_infos[info_offset].size = LHCb::LumiSummaryOffsets::V2::VeloVerticesSize;
    parameters.dev_lumi_infos[info_offset].offset = LHCb::LumiSummaryOffsets::V2::VeloVerticesOffset;
    parameters.dev_lumi_infos[info_offset].value = parameters.dev_number_of_pvs[event_number];
  }
}
