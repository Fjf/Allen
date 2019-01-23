#pragma once

#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "Handler.cuh"

__device__ void make_cluster_v5 (
  const int hit_index,
  const SciFi::HitCount& hit_count,
  const SciFi::SciFiGeometry& geom,
  uint32_t chan,
  uint8_t fraction,
  uint8_t pseudoSize,
  uint32_t uniqueMat,
  SciFi::Hits& hits);

__global__ void scifi_raw_bank_decoder_v5(
  char *scifi_events,
  uint *scifi_event_offsets,
  const uint *event_list,
  uint *scifi_hit_count,
  uint *scifi_hits,
  char *scifi_geometry,
  const float* dev_inv_clus_res);

ALGORITHM(scifi_raw_bank_decoder_v5, scifi_raw_bank_decoder_v5_t)
