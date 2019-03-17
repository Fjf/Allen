#pragma once

#include "LookingForwardConstants.cuh"
#include "LookingForwardTools.cuh"
#include "LFExtendTracksFirstLayersXImpl.cuh"
#include "SciFiEventModel.cuh"
#include "Handler.cuh"
#include "ArgumentsUT.cuh"
#include "ArgumentsSciFi.cuh"

__global__ void lf_extend_tracks_first_layers_x(
  const uint32_t* dev_scifi_hits,
  const uint32_t* dev_scifi_hit_count,
  const int* dev_atomics_ut,
  SciFi::TrackHits* dev_scifi_tracks,
  int* dev_atomics_scifi,
  const char* dev_scifi_geometry,
  const LookingForward::Constants* dev_looking_forward_constants,
  const float* dev_inv_clus_res,
  const uint* dev_scifi_lf_number_of_candidates,
  const short* dev_scifi_lf_candidates,
  const uint8_t relative_extrapolation_layer);

ALGORITHM(
  lf_extend_tracks_first_layers_x,
  lf_extend_tracks_first_layers_x_t,
  ARGUMENTS(
    dev_scifi_hits,
    dev_scifi_hit_count,
    dev_atomics_ut,
    dev_scifi_lf_tracks,
    dev_scifi_lf_atomics,
    dev_scifi_lf_number_of_candidates,
    dev_scifi_lf_candidates))
