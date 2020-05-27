#pragma once

#include "CudaCommon.h"
#include "CaloRawBanks.h"

struct CaloRawEvent {
  uint32_t number_of_raw_banks;
  char* data;
  uint32_t* offsets;

  // For Allen format
  __device__ __host__ CaloRawEvent(const char* events, const uint32_t* o)
  : number_of_raw_banks{((uint32_t*) event)[0]},
    data{events},
    offsets{o}
  {
  }

  CaloRawBank bank(unsigned event, unsigned n) {
    const char* event_data = data + offsets[event];
    uint32_t* bank_offsets = ((uint32_t*)event_data) + 1;
    return CaloRawBank{event_data + (number_of_raw_banks + 2) * sizeof(uint32_t) + bank_offsets[n]};
  }
};

struct CaloMepEvent {
  uint32_t number_of_raw_banks;
  char* blocks;
  uint32_t* offsets;

  // For Allen format
  __device__ __host__ CaloMepEvent(const char* b, const uint32_t* o)
  : number_of_raw_banks{MEP::number_of_banks(o)},
    offsets{o},
    blocks{b}
  {
  }

  CaloRawBank bank(unsigned event, unsigned n) {
    return MEP::raw_bank<CaloRawBank>(blocks, offsets, event, n);
  }
};
