/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <cstdint>
#include <cassert>

namespace Muon {
  struct MuonRawBank {
    uint32_t sourceID;
    uint16_t* data;
    uint16_t* last;

    __device__ MuonRawBank(const char* raw_bank, const char* end)
    {
      const char* p = raw_bank;
      sourceID = *((uint32_t*) p);
      p += sizeof(uint32_t);
      data = (uint16_t*) p;
      last = (uint16_t*) end;
    }

    __device__ MuonRawBank(const uint32_t sID, const char* bank_start, const char* bank_end)
    {
      sourceID = sID;
      data = (uint16_t*) bank_start;
      last = (uint16_t*) bank_end;
    }

  };

  struct MuonRawEvent {
    static constexpr size_t batches_per_bank = 4;
    uint32_t number_of_raw_banks;

    uint32_t* raw_bank_offset;
    char* payload;

    __device__ MuonRawEvent(const char* event)
    {
      const char* p = event;
      number_of_raw_banks = ((uint32_t*) p)[0];
      p += sizeof(uint32_t);
      raw_bank_offset = (uint32_t*) p;
      p += (number_of_raw_banks + 1) * sizeof(uint32_t);
      payload = (char*) p;
    }

    __device__ MuonRawBank getMuonBank(const uint32_t index) const
    {
      MuonRawBank bank(payload + raw_bank_offset[index], payload + raw_bank_offset[index + 1]);
      return bank;
    }
  };
} // namespace Muon
