#pragma once

#include "Handler.cuh"
#include "ArgumentsMuon.cuh"
#include "MuonDefinitions.cuh"
#include "FindPermutation.cuh"

__global__ void muon_sort_station_region_quarter(
  const uint* dev_storage_tile_id,
  const uint* dev_atomics_muon,
  uint* dev_permutation_srq);

ALGORITHM(
  muon_sort_station_region_quarter,
  muon_sort_station_region_quarter_t,
  ARGUMENTS(dev_storage_tile_id, dev_atomics_muon, dev_permutation_srq))
