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
#include "ConsolidateSciFi.cuh"

#ifndef ALLEN_STANDALONE

#include "Gaudi/Accumulators.h"
#include "Gaudi/Accumulators/Histogram.h"


void scifi_consolidate_tracks::scifi_consolidate_tracks_t::init_monitor()
{
  const std::string long_track_counter_name {"n_long_tracks"};
  m_long_tracks = std::make_unique<Gaudi::Accumulators::Counter<>>(this, long_track_counter_name);
}

void scifi_consolidate_tracks::scifi_consolidate_tracks_t::monitor_operator(
  const ArgumentReferences<Parameters>& arguments,
  gsl::span<unsigned> track_offsets ) const
{
  
  auto buf_tracks = m_long_tracks->buffer();
  for (auto i = 0u; i < first<host_number_of_events_t>(arguments); ++i) {
    auto n_tracks_event = track_offsets[i+1] - track_offsets[i];
    buf_tracks+=n_tracks_event;
  }
}

#endif
