#include "MuonDecoding.cuh"
#include <stdio.h>

using namespace Muon;

__global__ void muon_decoding(char* events, unsigned int* offsets, Muon::MuonRawToHits* muon_raw_to_hits,
    Muon::HitsSoA* unordered_muon_hits, Muon::HitsSoA* muon_hits) {
  __shared__ int currentHitIndex;
  size_t eventId = blockIdx.x;
  //if (eventId != 2) {
//    return;
//  }
  //printf("blockIdx.x = %u\n", blockIdx.x);

  size_t station = threadIdx.x / (Muon::Constants::n_regions * Muon::Constants::n_quarters);
  size_t region = (threadIdx.x % (Muon::Constants::n_regions * Muon::Constants::n_quarters)) / Muon::Constants::n_regions;
  size_t quarter = threadIdx.x % Muon::Constants::n_quarters;
  //printf("threadIdx.x = %u\n, ", threadIdx.x);
  //printf("station = %u, ", station);
  //printf("region = %u, ", region);
  //printf("quarter = %u\n", quarter);
  __shared__ unsigned int storageTileId[Constants::max_numhits_per_event];
  __shared__ unsigned int storageTdcValue[Constants::max_numhits_per_event];
  __shared__ unsigned int sortedStorageTileId[Constants::max_numhits_per_event];
  __shared__ unsigned int sortedStorageTdcValue[Constants::max_numhits_per_event];
  __shared__ Digit digits[Constants::max_numhits_per_event];
  __shared__ int currentStorageIndex;
  __shared__ int storageStationRegionQuarterOccurrences[Muon::Constants::n_stations * Muon::Constants::n_regions * Muon::Constants::n_quarters];
  __shared__ int storageStationRegionQuarterOccurrencesOffset[Muon::Constants::n_stations * Muon::Constants::n_regions * Muon::Constants::n_quarters + 1];
  __shared__ int originalStorageStationRegionQuarterOccurrencesOffset[Muon::Constants::n_stations * Muon::Constants::n_regions * Muon::Constants::n_quarters + 1];
  __shared__ bool used[Constants::max_numhits_per_event];
  __shared__ int stationOccurrences[Muon::Constants::n_stations];
  __shared__ int stationOccurrencesOffset[Muon::Constants::n_stations + 1];

  if (threadIdx.x == 0) {
    currentHitIndex = 0;
    currentStorageIndex = 0;
    memset(storageStationRegionQuarterOccurrences, 0, sizeof(storageStationRegionQuarterOccurrences));
    storageStationRegionQuarterOccurrencesOffset[0] = 0;
    originalStorageStationRegionQuarterOccurrencesOffset[0] = 0;
    memset(used, false, sizeof(used));
    memset(stationOccurrences, 0, sizeof(stationOccurrences));
    stationOccurrencesOffset[0] = 0;
  }
  __syncthreads();

  if (region == 0 && quarter == 0) {
    MuonRawEvent rawEvent = MuonRawEvent((const char*) events + offsets[eventId]);
    for (uint32_t bank_index = 0; bank_index < rawEvent.number_of_raw_banks; bank_index++) {
      unsigned int tell1Number = rawEvent.getMuonBank(bank_index).sourceID;
      size_t stationByBankNumber = (tell1Number < 4 ? 0 : tell1Number < 6 ? 1 : tell1Number < 8 ? 2 : 3);
      if (stationByBankNumber != station) {
        continue;
      }
      MuonRawBank rawBank = rawEvent.getMuonBank(bank_index);
      uint16_t* p = rawBank.data;
      int preamble_size = 2 * ((* p + 3) / 2);
      p += preamble_size;
      for (size_t i = 0; i < 4; i++) {
        uint16_t frontValue = * p;
        for (size_t shift = 1; shift < 1 + frontValue; shift++) {
          unsigned int pp = * (p + shift);
          unsigned int add = (pp & 0x0FFF);
          unsigned int tdc_value = ((pp & 0xF000) >> 12);
          unsigned int tileId = muon_raw_to_hits->muonGeometry.getADDInTell1(tell1Number, add);
          //printf("tileId = %u\n", tileId);
          if (tileId != 0) {
            int localCurrentStorageIndex = atomicAdd(&currentStorageIndex, 1);
  //          printf("localCurrentStorageIndex = %u\n", localCurrentStorageIndex);
            storageTileId[localCurrentStorageIndex] = tileId;
            storageTdcValue[localCurrentStorageIndex] = tdc_value;
          }
        }
        p += 1 + frontValue;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("currentStorageIndex = %d\n", currentStorageIndex * 10000 + eventId);
    for (size_t i = 0; i < currentStorageIndex; i++) {
      size_t stationRegionQuarter = Muon::MuonTileID::stationRegionQuarter(storageTileId[i]);
      storageStationRegionQuarterOccurrences[stationRegionQuarter]++;
    }
    for (size_t i = 0; i < Muon::Constants::n_stations * Muon::Constants::n_regions * Muon::Constants::n_quarters; i++) {
      storageStationRegionQuarterOccurrencesOffset[i + 1] =
          storageStationRegionQuarterOccurrencesOffset[i] + storageStationRegionQuarterOccurrences[i];
      originalStorageStationRegionQuarterOccurrencesOffset[i + 1] = storageStationRegionQuarterOccurrencesOffset[i + 1];
    }
    for (size_t i = 0; i < currentStorageIndex; i++) {
      size_t stationRegionQuarter = Muon::MuonTileID::stationRegionQuarter(storageTileId[i]);
      size_t index = storageStationRegionQuarterOccurrencesOffset[stationRegionQuarter];
      storageStationRegionQuarterOccurrencesOffset[stationRegionQuarter]++;
      sortedStorageTileId[index] = storageTileId[i];
      sortedStorageTdcValue[index] = storageTdcValue[i];
    }
  }
  __syncthreads();
  //printf("OFFSET = %d\n", storageStationRegionQuarterOccurrences[threadIdx.x]*10000000 + originalStorageStationRegionQuarterOccurrencesOffset[threadIdx.x] * 10000 + station*100 + region*10+quarter);
  //return;
  //if (eventId == 2) {
    muon_raw_to_hits->addCoordsCrossingMap(
        sortedStorageTileId,
        sortedStorageTdcValue,
        used,
        originalStorageStationRegionQuarterOccurrencesOffset[threadIdx.x],
        originalStorageStationRegionQuarterOccurrencesOffset[threadIdx.x + 1],
        &unordered_muon_hits[eventId],
        currentHitIndex
    );
  //}
  __syncthreads();
  //return;
  if (threadIdx.x == 0) {
    printf("currentHitIndex = %d\n", currentHitIndex * 10000 + eventId);

    for (size_t i = 0; i < currentHitIndex; i++) {
      size_t currentStation = Muon::MuonTileID::station(unordered_muon_hits[eventId].tile[i]);
      stationOccurrences[currentStation]++;
    }

    for (size_t i = 0; i < Muon::Constants::n_stations; i++) {
      stationOccurrencesOffset[i + 1] = stationOccurrencesOffset[i] + stationOccurrences[i];
    }
    for (size_t i = 0; i < Muon::Constants::n_stations; i++) {
      muon_hits->station_offsets[i] = stationOccurrencesOffset[i];
      muon_hits->number_of_hits_per_station[i] = stationOccurrences[i];
    }
    for (size_t i = 0; i < currentStorageIndex; i++) {
      size_t currentStation = Muon::MuonTileID::station(unordered_muon_hits[eventId].tile[i]);
      size_t index = stationOccurrencesOffset[currentStation];
      stationOccurrencesOffset[currentStation]++;
      //printf("index = %d\n", index);

      Muon::setAtIndex(
          &muon_hits[eventId],
          index,
          unordered_muon_hits[eventId].tile[i],
          unordered_muon_hits[eventId].x[i],
          unordered_muon_hits[eventId].dx[i],
          unordered_muon_hits[eventId].y[i],
          unordered_muon_hits[eventId].dy[i],
          unordered_muon_hits[eventId].z[i],
          unordered_muon_hits[eventId].dz[i],
          unordered_muon_hits[eventId].uncrossed[i],
          unordered_muon_hits[eventId].time[i],
          unordered_muon_hits[eventId].delta_time[i],
          unordered_muon_hits[eventId].cluster_size[i],
          unordered_muon_hits[eventId].region_id[i]
      );
    }
  }
  __syncthreads();
}