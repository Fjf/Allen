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
#include "VeloConsolidateTracks.cuh"

#ifndef ALLEN_STANDALONE

#include "Gaudi/Accumulators.h"
#include "Gaudi/Accumulators/Histogram.h"

template<int I>
using gaudi_histo_t = Gaudi::Accumulators::Histogram<I, Gaudi::Accumulators::atomicity::full, float>;

void velo_consolidate_tracks::velo_consolidate_tracks_t::init_monitor()
{
  const std::string velo_track_counter_name {"n_velo_tracks"};
  m_velo_tracks = std::make_unique<Gaudi::Accumulators::Counter<>>(this, velo_track_counter_name);

  histogram_n_velo_tracks = (void*) new gaudi_histo_t<1>(
    this,
    "n_velo_tracks_event",
    "n_velo_tracks_event",
    Gaudi::Accumulators::Axis<float> {50, 0, (float) 200, {}, {}});
}

void velo_consolidate_tracks::velo_consolidate_tracks_t::monitor_operator(
  const ArgumentReferences<Parameters>& arguments,
  gsl::span<unsigned> track_offsets ) const
{
  
  auto* histogram_n_velo_tracks_p = reinterpret_cast<gaudi_histo_t<1>*>(histogram_n_velo_tracks);
  auto buf_tracks = m_velo_tracks->buffer();
  auto hist_buf = histogram_n_velo_tracks_p->buffer();
  for (auto i = 0u; i < first<host_number_of_events_t>(arguments); ++i) {
    auto n_tracks_event = track_offsets[i+1] - track_offsets[i];
    buf_tracks+=n_tracks_event;
    hist_buf[n_tracks_event]++;
  }
}

#endif
