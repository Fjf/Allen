/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <tuple>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include "Clustering.h"

void checkClustering(
  const std::vector<char>& geometry,
  const std::vector<char>& events,
  const std::vector<unsigned>& event_offsets,
  const std::vector<std::vector<uint32_t>>& found_clusters,
  float& reconstruction_efficiency,
  float& clone_fraction,
  float& ghost_fraction,
  const bool just_check_ids = true,
  const float allowed_distance_error = 1.f);
