/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <stdint.h>
#include <vector>
#include <ostream>
#include <sstream>
#include "BackendCommon.h"
#include "Common.h"
#include "Logger.h"
#include "States.cuh"

#include "assert.h"

namespace SciFi {

  // need 3 arrays (size: number_of_events) for copy_and_prefix_sum_scifi_t
  static constexpr int num_atomics = 3;

  // need 1 arrays (size: number_of_events) for seeding, right?
  static constexpr int num_seeding_atomics = 1;

  namespace Constants {
    // Detector description
    // There are three stations with four layers each
    static constexpr unsigned n_stations = 3;
    static constexpr unsigned n_layers_per_station = 4;
    static constexpr unsigned n_zones = 24;
    static constexpr unsigned n_layers = 12;
    static constexpr unsigned n_xzlayers = 6;
    static constexpr unsigned n_uvlayers = 6;
    static constexpr unsigned n_mats = 1024;
    static constexpr unsigned n_parts = 2;
    static constexpr unsigned max_num_seed_tracks = 6000; // FIXME
    static constexpr int INVALID_IDX = -1;                // FIXME
    static constexpr int INVALID_ID = 0;                  // FIXME

    /**
     * The following constants are based on the number of modules per quarter.
     * There are currently 80 raw banks per SciFi station:
     *
     *   The first two stations (first 160 raw banks) encode 4 modules per quarter.//FIXME: WRONG
     *   The last station (raw banks 161 to 240) encode 5 modules per quarter.//FIXME: WRONG
     *
     * The raw data is sorted such that every four consecutive modules are either
     * monotonically increasing or monotonically decreasing, following a particular pattern.
     * Thus, it is possible to decode the first 160 raw banks in v4 in parallel since the
     * position of each hit is known by simply knowing the current iteration in the raw bank,
     * and using that information as a relative index, given the raw bank offset.
     * This kind of decoding is what we call "direct decoding".
     *
     * However, the last 80 raw banks cannot be decoded in this manner. Therefore, the
     * previous method is employed for these last raw banks, consisting in a two-step
     * decoding.
     *
     * The constants below capture this idea. The prefix sum needed contains information about
     * "mat groups" (the first 160 raw banks, since the offset of the group is enough).
     * However, for the last sector, every mat offset is stored individually.
     */
    static constexpr unsigned max_corrected_mat = 1024; // FIXME: probably smaller.
    static constexpr unsigned n_consecutive_raw_banks = 160;
    static constexpr unsigned n_mats_per_consec_raw_bank = 4;
    static constexpr unsigned n_mat_groups_and_mats = 544;
    static constexpr unsigned mat_index_substract = n_consecutive_raw_banks * 3;
    static constexpr unsigned n_mats_without_group = n_mats - n_consecutive_raw_banks * n_mats_per_consec_raw_bank;

    // FIXME_GEOMETRY_HARDCODING
    // todo: use dzdy defined in geometry, read by mat
    static constexpr float dzdy = 0.003601f;
    static constexpr float dRatio = -0.00028f;
    static constexpr float ZBegT = 7500.f * Gaudi::Units::mm;   // FIXME_GEOMETRY_HARDCODING
    static constexpr float ZEndT = 9410.f * Gaudi::Units::mm;   // FIXME_GEOMETRY_HARDCODING
    static constexpr float z_mid_t = 8520.f * Gaudi::Units::mm; // FIXME_GEOMETRY_HARDCODING

    // Looking Forward
    static constexpr int max_track_size = n_layers;
    static constexpr int max_track_candidate_size = 4;
    static constexpr int hit_layer_offset = 6;
    static constexpr int max_SciFi_tracks_per_UT_track = 1;

    // This constant is for the HostBuffer reserve method, when validating
    static constexpr int max_tracks = 1000;

    // Constants for SciFi seeding
    static constexpr int Nmax_seed_xz_per_part = 300;
    static constexpr int Nmax_seed_xz = n_parts * Nmax_seed_xz_per_part;
    static constexpr int Nmax_seeds_per_part = Nmax_seed_xz_per_part;
    static constexpr int Nmax_seeds = n_parts * Nmax_seeds_per_part;
  } // namespace Constants

  namespace SciFiRawBankParams { // from SciFi/SciFiDAQ/src/SciFiRawBankParams.h
    enum shifts {
      linkShift = 9,
      cellShift = 2,
      fractionShift = 1,
      sizeShift = 0,
    };

    static constexpr uint16_t nbClusMaximum = 31;   // 5 bits
    static constexpr uint16_t nbClusFFMaximum = 10; //
    static constexpr uint16_t fractionMaximum = 1;  // 1 bits allocted
    static constexpr uint16_t cellMaximum = 127;    // 0 to 127; coded on 7 bits
    static constexpr uint16_t sizeMaximum = 1;      // 1 bits allocated

    enum BankProperties {
      NbBanksMax = 240,
      NbBanksPerQuarter = 5,
      NbLinksPerBank = 24,
      NbLinksMax = NbLinksPerBank * NbBanksMax
    };

    static constexpr uint16_t clusterMaxWidth = 4;
  } // namespace SciFiRawBankParams

  /**
   * @brief SciFi geometry description typecast.
   */
  struct SciFiGeometry {
    size_t size;
    uint32_t number_of_stations;
    uint32_t number_of_layers_per_station;
    uint32_t number_of_layers;
    uint32_t number_of_quarters_per_layer;
    uint32_t number_of_quarters;
    uint32_t* number_of_modules; // for each quarter
    uint32_t number_of_mats_per_module;
    uint32_t number_of_mats;
    uint32_t number_of_banks;
    uint32_t version;
    uint32_t* bank_first_channel; // decoding v6
    uint32_t* source_ids;         // decoding v7
    uint32_t* bank_sipm_list;     // decoding v7
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

    __device__ __host__ SciFiGeometry() {}

    /**
     * @brief Just typecast, no size check.
     */
    __device__ __host__ SciFiGeometry(const char* geometry)
    {
      const char* p = geometry;

      number_of_stations = *((uint32_t*) p);
      p += sizeof(uint32_t);
      number_of_layers_per_station = *((uint32_t*) p);
      p += sizeof(uint32_t);
      number_of_layers = *((uint32_t*) p);
      p += sizeof(uint32_t);
      number_of_quarters_per_layer = *((uint32_t*) p);
      p += sizeof(uint32_t);
      number_of_quarters = *((uint32_t*) p);
      p += sizeof(uint32_t);
      number_of_modules = (uint32_t*) p;
      p += number_of_quarters * sizeof(uint32_t);
      number_of_mats_per_module = *((uint32_t*) p);
      p += sizeof(uint32_t);
      number_of_mats = *((uint32_t*) p);
      p += sizeof(uint32_t);
      number_of_banks = *((uint32_t*) p);
      p += sizeof(uint32_t);
      version = *((uint32_t*) p);
      p += sizeof(uint32_t);
      if (version == 0) {
        bank_first_channel = (uint32_t*) p;
        p += number_of_banks * sizeof(uint32_t);
      }
      else {
        source_ids = (uint32_t*) p;
        p += number_of_banks * sizeof(uint32_t);
        bank_sipm_list = (uint32_t*) p;
        p += number_of_banks * SciFiRawBankParams::BankProperties::NbLinksPerBank * sizeof(uint32_t);
      }
      max_uniqueMat = *((uint32_t*) p);
      p += sizeof(uint32_t);
      mirrorPointX = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      mirrorPointY = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      mirrorPointZ = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      ddxX = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      ddxY = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      ddxZ = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      uBegin = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      halfChannelPitch = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      dieGap = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      sipmPitch = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      dxdy = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      dzdy = (float*) p;
      p += sizeof(float) * max_uniqueMat;
      globaldy = (float*) p;
      p += sizeof(float) * max_uniqueMat;

      size = p - geometry;
    }

    /**
     * @brief Typecast from std::vector.
     */
    SciFiGeometry(const std::vector<char>& geometry) : SciFiGeometry::SciFiGeometry(geometry.data()) {}
  };

  struct SciFiChannelID {
    uint32_t channelID;
    __host__ std::string toString() const
    {
      std::ostringstream s;
      s << "{ SciFiChannelID : "
        << " channel =" << std::to_string(channel()) << " sipm =" << std::to_string(sipm())
        << " mat =" << std::to_string(mat()) << " module=" << std::to_string(module())
        << " quarter=" << std::to_string(quarter()) << " layer=" << std::to_string(layer())
        << " station=" << std::to_string(station()) << " }";
      return s.str();
    }

    __device__ __host__ uint32_t channel() const { return (uint32_t)((channelID & channelMask) >> channelBits); }

    __device__ __host__ uint32_t sipm() const { return ((channelID & sipmMask) >> sipmBits); }

    __device__ __host__ uint32_t mat() const { return ((channelID & matMask) >> matBits); }

    __device__ __host__ uint32_t module() const { return ((channelID & moduleMask) >> moduleBits); }

    __device__ __host__ uint32_t localModuleIdx() const
    {
      // Returns local module ID in ascending x order.
      // There may be a faster way to do this.
      uint32_t module_count = station() >= 3 ? 6 : 5;
      return (isRight()) ? module_count - 1 - module() : module();
    }

    __device__ __host__ uint32_t quarter() const { return ((channelID & quarterMask) >> quarterBits); }

    __device__ __host__ uint32_t layer() const { return ((channelID & layerMask) >> layerBits); }

    __device__ __host__ uint32_t station() const { return ((channelID & stationMask) >> stationBits); }

    __device__ __host__ uint32_t globalLayerID() const { return ((channelID & uniqueLayerMask) >> layerBits); }
    __device__ __host__ uint32_t globalLayerIdx() const { return globalLayerID() - 4; }

    __device__ __host__ uint32_t globalQuarterID() const { return ((channelID & uniqueQuarterMask) >> quarterBits); }
    __device__ __host__ uint32_t globalQuarterIdx() const { return globalQuarterID() - 16; }

    __device__ __host__ uint32_t globalModuleID() const { return ((channelID & uniqueModuleMask) >> moduleBits); }
    __device__ __host__ uint32_t globalModuleIdx() const
    {
      auto quarterIdx = globalQuarterIdx();
      return quarterIdx * 5 + (quarterIdx >= 32 ? quarterIdx - 32 : 0) + localModuleIdx();
    }
    __device__ __host__ uint32_t globalMatID() const { return ((channelID & uniqueMatMask) >> matBits); }
    __device__ __host__ uint32_t globalMatID_shift() const { return globalMatID() - 512; }
    __device__ __host__ uint32_t globalMatIdx_Xorder() const
    {
      // Returns global mat ID in ascending x order without any gaps.
      // Geometry dependent. No idea how to not hardcode this.
      assert(globalModuleIdx() * 4 + (reversedZone() ? 3 - mat() : mat() < SciFi::Constants::max_corrected_mat));
      return globalModuleIdx() * 4 + (reversedZone() ? 3 - mat() : mat());
    }

    __device__ __host__ uint32_t die() const { return ((channelID & 0x40) >> 6); }

    __device__ __host__ bool isBottom() const { return (quarter() == 0 || quarter() == 1); }

    __device__ __host__ bool isRight() const { return (quarter() == 0 || quarter() == 2); }

    __device__ __host__ bool reversedZone() const
    {
      unsigned zone = ((globalQuarterIdx()) >> 1) % 4;
      return zone == 1 || zone == 2;
    }

    __device__ __host__ SciFiChannelID(const uint32_t channelID) : channelID(channelID) {}

    static constexpr uint32_t kInvalidChannelID = 14336;

    // from FTChannelID.h (generated)
    enum channelIDMasks {
      channelMask = 0x7fL,
      sipmMask = 0x180L,
      matMask = 0x600L,
      moduleMask = 0x3800L,
      quarterMask = 0xc000L,
      layerMask = 0x30000L,
      stationMask = 0xc0000L,
      uniqueLayerMask = layerMask | stationMask,
      uniqueQuarterMask = quarterMask | layerMask | stationMask,
      uniqueModuleMask = moduleMask | quarterMask | layerMask | stationMask,
      uniqueMatMask = matMask | moduleMask | quarterMask | layerMask | stationMask,
      uniqueSiPMMask = sipmMask | matMask | moduleMask | quarterMask | layerMask | stationMask
    };
    enum channelIDBits {
      channelBits = 0,
      sipmBits = 7,
      matBits = 9,
      moduleBits = 11,
      quarterBits = 14,
      layerBits = 16,
      stationBits = 18
    };
  };

  __device__ inline uint32_t channelInBank(const uint32_t c) { return (c >> SciFiRawBankParams::cellShift); }

  __device__ inline uint32_t channelInLink(const uint32_t c)
  {
    return (c >> SciFiRawBankParams::cellShift & SciFiChannelID::channelIDMasks::channelMask);
  }

  __device__ inline uint16_t getLinkInBank(const uint16_t c) { return (c >> SciFiRawBankParams::linkShift); }

  __device__ inline int cell(const uint16_t c)
  {
    return (c >> SciFiRawBankParams::cellShift) & SciFiRawBankParams::cellMaximum;
  }

  __device__ inline int fraction(const uint16_t c)
  {
    return (c >> SciFiRawBankParams::fractionShift) & SciFiRawBankParams::fractionMaximum;
  }

  __device__ inline bool cSize(const uint16_t c)
  {
    return (c >> SciFiRawBankParams::sizeShift) & SciFiRawBankParams::sizeMaximum;
  }

  __device__ inline unsigned int iSource(const SciFi::SciFiGeometry& geom, unsigned int sourceID)
  {
    if (geom.version == 0) return sourceID;
    unsigned int output(geom.number_of_banks);
    for (uint32_t i = 0; i < geom.number_of_banks; i++)
      if (geom.source_ids[i] == sourceID) {
        output = i;
        break;
      }
    return output;
  }

  __device__ inline uint32_t
  getGlobalSiPMFromIndex(const SciFi::SciFiGeometry& geom, const unsigned int iRowInDB, const uint16_t c)
  {
    auto localLinkIdx = getLinkInBank(c);
    if (localLinkIdx >= SciFi::SciFiRawBankParams::BankProperties::NbLinksPerBank) {
      // Corrupted data
      return SciFi::SciFiChannelID::kInvalidChannelID;
    }
    uint32_t globalSipmID =
      geom.bank_sipm_list[iRowInDB * SciFi::SciFiRawBankParams::BankProperties::NbLinksPerBank + localLinkIdx];
    return globalSipmID;
  }

  namespace ClusterTypes {
    constexpr unsigned int NullCluster = 0x00;
    constexpr unsigned int SmallCluster = 0x01;
    constexpr unsigned int LastCluster = 0x02;
    constexpr unsigned int BigCluster = 0x03;
    constexpr unsigned int EdgeCluster = 0x04;
    constexpr unsigned int SizeLt8Cluster = 0x05;
  }; // namespace ClusterTypes

  namespace ClusterReference {
    static constexpr uint32_t maxRawBank = 0xFF;
    static constexpr uint32_t maxICluster = 0xFF;
    static constexpr uint32_t maxCond = 0x07;
    static constexpr uint32_t maxDelta = 0xFF;
    static constexpr int rawBankShift = 24;
    static constexpr int iClusterShift = 16;
    static constexpr int condShift = 13;
    __device__ inline uint32_t
    makeClusterReference(const int raw_bank, const int it, const int condition, const int delta)
    {
      return (raw_bank & maxRawBank) << rawBankShift | (it & maxICluster) << iClusterShift |
             (condition & maxCond) << condShift | (delta & maxDelta);
    };
    __device__ inline int getRawBank(uint32_t cluster_reference)
    {
      return (cluster_reference >> rawBankShift) & maxRawBank;
    }
    __device__ inline int getICluster(uint32_t cluster_reference)
    {
      return (cluster_reference >> iClusterShift) & maxICluster;
    }
    __device__ inline int getCond(uint32_t cluster_reference) { return (cluster_reference >> condShift) & maxCond; }
    __device__ inline int getDelta(uint32_t cluster_reference) { return (cluster_reference) &maxDelta; }

  }; // namespace ClusterReference

  __device__ inline bool lastClusterSiPM(unsigned c, unsigned c2, const uint16_t* it, const uint16_t* last)
  {
    return (it + 1 == last || SciFi::getLinkInBank(c) != SciFi::getLinkInBank(c2));
  }

  template<int decoding_version>
  __device__ inline bool startLargeCluster(unsigned c)
  {
    if constexpr (decoding_version == 7) {
      return SciFi::cSize(c) && !SciFi::fraction(c);
    }
    return SciFi::cSize(c) && SciFi::fraction(c);
  }

  template<int decoding_version>
  __device__ inline bool endLargeCluster(unsigned c)
  {
    if constexpr (decoding_version == 7) {
      return SciFi::cSize(c);
    }
    return SciFi::cSize(c) && !SciFi::fraction(c);
  }

  __device__ inline bool wellOrdered(unsigned c, unsigned c2) { return SciFi::cell(c) < SciFi::cell(c2); }

} // namespace SciFi
