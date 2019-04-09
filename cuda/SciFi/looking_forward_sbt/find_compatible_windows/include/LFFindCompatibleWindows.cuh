#pragma once

#include "PrForwardConstants.cuh"
#include "VeloConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "SciFiEventModel.cuh"
#include "SciFiDefinitions.cuh"
#include "Handler.cuh"
#include "ArgumentsVelo.cuh"
#include "ArgumentsUT.cuh"
#include "ArgumentsSciFi.cuh"
#include "LookingForwardConstants.cuh"
#include "LookingForwardTools.cuh"

__global__ void lf_find_compatible_windows(
  uint32_t* dev_scifi_hits,
  const uint32_t* dev_scifi_hit_count,
  const int* dev_atomics_ut,
  const MiniState* dev_ut_states,
  const char* dev_scifi_geometry,
  const float* dev_inv_clus_res,
  const int* dev_initial_windows,
  const uint* dev_scifi_lf_number_of_candidates,
  const short* dev_scifi_lf_candidates,
  short* dev_scifi_lf_compatible_windows,
  const LookingForward::Constants* dev_looking_forward_constants);

ALGORITHM(
  lf_find_compatible_windows,
  lf_find_compatible_windows_t,
  ARGUMENTS(
    dev_scifi_hits,
    dev_scifi_hit_count,
    dev_atomics_ut,
    dev_ut_states,
    dev_scifi_lf_initial_windows,
    dev_scifi_lf_number_of_candidates,
    dev_scifi_lf_candidates,
    dev_scifi_lf_compatible_windows))