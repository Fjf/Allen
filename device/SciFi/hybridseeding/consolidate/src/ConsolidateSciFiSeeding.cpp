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
#include "ConsolidateSciFiSeeding.cuh"

#ifndef ALLEN_STANDALONE

#include "Gaudi/Accumulators.h"
#include "Gaudi/Accumulators/Histogram.h"

template<int I>
using gaudi_histo_t = Gaudi::Accumulators::Histogram<I, Gaudi::Accumulators::atomicity::full, float>;

void seed_confirmTracks_consolidate::seed_confirmTracks_consolidate_t::init_monitor()
{
  const std::string seed_track_counter_name {"n_seed_tracks"};
  m_seed_tracks = std::make_unique<Gaudi::Accumulators::Counter<>>(this, seed_track_counter_name);

  histogram_n_scifi_seeds = (void*) new gaudi_histo_t<1>(
    this,
    "n_scifi_seeds_event",
    "n_scifi_seeds_event",
    Gaudi::Accumulators::Axis<float> {80, 0, (float) 200, {}, {}});
}

void seed_confirmTracks_consolidate::seed_confirmTracks_consolidate_t::monitor_operator(
  const ArgumentReferences<Parameters>& arguments,
  gsl::span<unsigned> track_offsets ) const
{
  
  auto* histogram_n_scifi_seeds_p = reinterpret_cast<gaudi_histo_t<1>*>(histogram_n_scifi_seeds);
  auto buf_tracks = m_seed_tracks->buffer();
  auto hist_buf = histogram_n_scifi_seeds_p->buffer();
  for (auto i = 0u; i < first<host_number_of_events_t>(arguments); ++i) {
    auto n_tracks_event = track_offsets[i+1] - track_offsets[i];
    buf_tracks+=n_tracks_event;
    if(n_tracks_event>200) continue;
    hist_buf[n_tracks_event]++;
  }
}

#endif
