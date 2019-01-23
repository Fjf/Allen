#pragma once

#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "Handler.cuh"

__device__ void store_sorted_cluster_reference_v5 (
  const SciFi::HitCount& hit_count,
  const uint32_t uniqueMat,
  const uint32_t chan,
  uint32_t* shared_mat_offsets,
  const int raw_bank,
  const int it,
  const int condition_1,
  const int condition_2,
  const int delta,
  SciFi::Hits& hits);

__global__ void scifi_pre_decode_v5(
  char *scifi_events,
  uint *scifi_event_offsets,
  const uint *event_list,
  uint *scifi_hit_count,
  uint *scifi_hits,
  char *scifi_geometry,
  const float* dev_inv_clus_res);

ALGORITHM(scifi_pre_decode_v5, scifi_pre_decode_v5_t)
