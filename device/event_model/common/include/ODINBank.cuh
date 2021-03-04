/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#pragma once

#include <BackendCommon.h>
#include <MEPTools.h>

struct ODINRawBank {

  uint32_t* data;

  /// Constructor from Allen layout
  __device__ __host__ ODINRawBank(const char* raw_bank) { data = (uint32_t*) (raw_bank + sizeof(uint32_t)); }

  /// Constructor from MEP layout
  __device__ __host__ ODINRawBank(const uint32_t, const char* fragment) { data = (uint32_t*) fragment; }
};

struct odin_data_t {
  static __host__ __device__ const unsigned*
  data(const char* dev_odin_data, const uint* dev_odin_offsets, const uint event_number)
  {
    // In Allen layout, the first uint is the number of banks,
    // followed by 2 offsets and then the source ID of the bank; 4
    // uint in total
    return ODINRawBank(dev_odin_data + dev_odin_offsets[event_number] + 4 * sizeof(uint32_t)).data;
  }
};

struct odin_data_mep_t {
  static __host__ __device__ const unsigned*
  data(const char* dev_odin_data, const uint* dev_odin_offsets, const uint event_number)
  {
    return MEP::raw_bank<ODINRawBank>(dev_odin_data, dev_odin_offsets, event_number, 0).data;
  }
};
