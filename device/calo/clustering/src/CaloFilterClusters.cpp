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
#include "CaloFilterClusters.cuh"

#ifndef ALLEN_STANDALONE

#include "Gaudi/Accumulators.h"
#include "Gaudi/Accumulators/Histogram.h"


void calo_filter_clusters::calo_filter_clusters_t::init_monitor()
{
  const std::string calo_cluster_counter_name {"n_calo_clusters"};
  m_calo_clusters = std::make_unique<Gaudi::Accumulators::Counter<>>(this, calo_cluster_counter_name);
}

void calo_filter_clusters::calo_filter_clusters_t::monitor_operator(
  const ArgumentReferences<Parameters>& arguments,
  gsl::span<unsigned> cluster_offsets ) const
{
  
  auto buf_clusters = m_calo_clusters->buffer();
  for (auto i = 0u; i < first<host_number_of_events_t>(arguments); ++i) {
    auto n_clusters_event = cluster_offsets[i+1] - cluster_offsets[i];
    buf_clusters+=n_clusters_event;
  }
}
#endif