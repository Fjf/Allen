#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>

#include "Common.h"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "CpuHandler.cuh"
#include "SciFiEventModel.cuh"
#include "MiniState.cuh"
#include "VeloEventModel.cuh"
#include "VeloConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "SciFiDefinitions.cuh"
#include "SciFiParametrization.h"

MiniState state_at_z(const MiniState state, const float z);

int run_momentum_forward_on_CPU (
  SciFi::TrackHits* host_scifi_tracks_events,
  int* host_scifi_n_tracks,
  const uint* host_scifi_hits,
  const uint* host_scifi_hit_count,
  const char* host_scifi_geometry, 
  const std::array<float, 9>& host_inv_clus_res,
  const uint* host_velo_tracks_atomics,
  const uint* host_velo_track_hit_number,
  const char* host_velo_states,
  const int * host_atomics_ut,
  const uint* host_ut_track_hit_number,
  const float* host_ut_qop,
  const float* host_ut_x,
  const float* host_ut_tx,
  const float* host_ut_z,
  const uint* host_ut_track_velo_indices,
  const std::vector< std::vector< std::vector< uint32_t > > > scifi_ids_ut_tracks,
  const std::vector< std::vector< float > > p_events,
  const uint number_of_events);

CPU_ALGORITHM(run_momentum_forward_on_CPU, cpu_scifi_momentum_forward_t)
