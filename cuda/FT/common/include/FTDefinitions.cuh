#pragma once

#include <stdint.h>
#include <vector>
#include <ostream>

/**
 * @brief FT geometry description typecast.
 */
namespace FT {
constexpr uint32_t number_of_zones = 24;

struct FTGeometry {
  size_t size;
  uint32_t number_of_stations;
  uint32_t number_of_layers_per_station;
  uint32_t number_of_layers;
  uint32_t number_of_quarters_per_layer;
  uint32_t number_of_quarters;
  uint32_t* number_of_modules; //for each quarter
  uint32_t number_of_mats_per_module;
  uint32_t number_of_mats;
  uint32_t number_of_tell40s;
  uint32_t* bank_first_channel;
  uint32_t max_uniqueMat;
  float* mirrorPointX;
  float* mirrorPointY;
  float* mirrorPointZ;
  float* ddxX;
  float* ddxY;
  float* ddxZ;
  float* uBegin;
  float* halfChannelPitch;
  float* dieGap;
  float* sipmPitch;
  float* dxdy;
  float* dzdy;
  float* globaldy;

  /**
   * @brief Typecast from std::vector.
   */
  FTGeometry(const std::vector<char>& geometry);

  /**
   * @brief Just typecast, no size check.
   */
  __device__ __host__ FTGeometry(
    const char* geometry
  );
};

struct FTRawBank {
  uint32_t sourceID;
  uint16_t* data;
  uint16_t* last;

  __device__ __host__ FTRawBank(const char* raw_bank, const char* end);
};

struct FTRawEvent {
  uint32_t number_of_raw_banks;
  uint32_t* raw_bank_offset;
  char* payload;

  __device__ __host__ FTRawEvent(const char* event);
  __device__ __host__ FTRawBank getFTRawBank(const uint32_t index) const;
};

namespace FTRawBankParams { //from FT/FTDAQ/src/FTRawBankParams.h
  enum shifts {
    linkShift     = 9,
    cellShift     = 2,
    fractionShift = 1,
    sizeShift     = 0,
  };

  static constexpr uint16_t nbClusMaximum   = 31;  // 5 bits
  static constexpr uint16_t nbClusFFMaximum = 10;  //
  static constexpr uint16_t fractionMaximum = 1;   // 1 bits allocted
  static constexpr uint16_t cellMaximum     = 127; // 0 to 127; coded on 7 bits
  static constexpr uint16_t sizeMaximum     = 1;   // 1 bits allocated

  enum BankProperties {
    NbBanks = 240,
    NbLinksPerBank = 24
  };

  static constexpr uint16_t clusterMaxWidth = 4;
}


struct FTChannelID {
  uint32_t channelID;
  __device__ __host__ uint32_t channel() const;
  __device__ __host__ uint32_t sipm() const;
  __device__ __host__ uint32_t mat() const;
  __device__ __host__ uint32_t uniqueMat() const;
  __device__ __host__ uint32_t module() const;
  __device__ __host__ uint32_t uniqueModule() const;
  __device__ __host__ uint32_t quarter() const;
  __device__ __host__ uint32_t uniqueQuarter() const;
  __device__ __host__ uint32_t layer() const;
  __device__ __host__ uint32_t uniqueLayer() const;
  __device__ __host__ uint32_t station() const;
  __device__ __host__ uint32_t die() const;
  __device__ __host__ bool isBottom() const;
  __device__ __host__ FTChannelID operator+=(const uint32_t& other);
  __host__ std::string toString();
  __device__ __host__ FTChannelID(const uint32_t channelID);
  //from FTChannelID.h (generated)
  enum channelIDMasks{channelMask       = 0x7fL,
                      sipmMask          = 0x180L,
                      matMask           = 0x600L,
                      moduleMask        = 0x3800L,
                      quarterMask       = 0xc000L,
                      layerMask         = 0x30000L,
                      stationMask       = 0xc0000L,
                      uniqueLayerMask   = layerMask + stationMask,
                      uniqueQuarterMask = quarterMask + layerMask + stationMask,
                      uniqueModuleMask  = moduleMask + quarterMask + layerMask + stationMask,
                      uniqueMatMask     = matMask + moduleMask + quarterMask + layerMask + stationMask,
                      uniqueSiPMMask    = sipmMask + matMask + moduleMask + quarterMask + layerMask + stationMask
};
  enum channelIDBits{channelBits       = 0,
                     sipmBits          = 7,
                     matBits           = 9,
                     moduleBits        = 11,
                     quarterBits       = 14,
                     layerBits         = 16,
                     stationBits       = 18};
};

/**
* @brief Offset and number of hits of each layer.
*/
struct FTHitCount{
  uint* layer_offsets;
  uint* n_hits_layers;

  __device__ __host__
  void typecast_before_prefix_sum(
    uint* base_pointer,
    const uint event_number
  );

  __device__ __host__
  void typecast_after_prefix_sum(
    uint* base_pointer,
    const uint event_number,
    const uint number_of_events
  );
};

struct FTHit {
  float x0;
  float z0;
  float w;
  float dxdy;
  float dzdy;
  float yMin;
  float yMax;
  float werrX;
  float coord;
  uint32_t LHCbID;
  uint32_t planeCode;
  uint32_t hitZone;
  uint32_t info;
  bool used;

  friend std::ostream& operator<<(std::ostream& stream, const FTHit& hit) {
  stream << "FT hit {"
    << hit.planeCode << ", "
    << hit.hitZone << ", "
    << hit.LHCbID << ", "
    << hit.x0 << ", "
    << hit.z0 << ", "
    << hit.w<< ", "
    << hit.dxdy << ", "
    << hit.dzdy << ", "
    << hit.yMin << ", "
    << hit.yMax << ", "
    << hit.werrX << "}";

  return stream;
}

};

struct FTHits {
  float* x0;
  float* z0;
  float* w;
  float* dxdy;
  float* dzdy;
  float* yMin;
  float* yMax;
  float* werrX;
  float* coord;
  uint32_t* LHCbID;
  uint32_t* planeCode;
  uint32_t* hitZone;
  uint32_t* info;
  bool* used;
  uint32_t* temp;


  FTHits() = default;

  /**
   * @brief Populates the FTHits object pointers from an unsorted array of data
   *        pointed by base_pointer.
   */
  __host__ __device__
  void typecast_unsorted(char* base_pointer, uint32_t total_number_of_hits);

  /**
   * @brief Populates the FTHits object pointers from a sorted array of data
   *        pointed by base_pointer.
   */
  __host__ __device__
  void typecast_sorted(char* base_pointer, uint32_t total_number_of_hits);

  /**
   * @brief Gets a hit in the FTHit format from the global hit index.
   */
  FTHit getHit(uint32_t index) const;

  // check for used hit
  __device__ __host__ bool isValid( uint32_t value ) const {
    return !used[value];
  }
};
}
