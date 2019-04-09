#pragma once

#include "LookingForwardConstants.cuh"
#include "LookingForwardTools.cuh"
#include "SciFiEventModel.cuh"
#include "Handler.cuh"
#include "ArgumentsSciFi.cuh"

__global__ void lf_quality_filter_x(
  const SciFi::TrackHits* dev_scifi_lf_tracks,
  const int* dev_scifi_lf_atomics,
  SciFi::TrackHits* dev_scifi_lf_filtered_tracks,
  int* dev_scifi_lf_filtered_atomics);

ALGORITHM(
  lf_quality_filter_x,
  lf_quality_filter_x_t,
  ARGUMENTS(
    dev_scifi_lf_tracks,
    dev_scifi_lf_atomics,
    dev_scifi_lf_filtered_tracks,
    dev_scifi_lf_filtered_atomics))