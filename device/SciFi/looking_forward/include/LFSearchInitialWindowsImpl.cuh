/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>
#include "SciFiDefinitions.cuh"
#include "UTDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "LookingForwardConstants.cuh"

__device__ void lf_search_initial_windows_impl(
  SciFi::ConstHits& scifi_hits,
  SciFi::ConstHitCount& scifi_hit_count,
  const MiniState& UT_state,
  const LookingForward::Constants* looking_forward_constants,
  const float* magnet_polarity,
  const float qop,
  const bool side,
  int* initial_windows,
  const int number_of_tracks,
  const unsigned event_offset,
  bool* dev_process_track,
  const unsigned ut_track_index,
  const unsigned hit_window_size);
