/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <ostream>
#include <stdint.h>

#include "States.cuh"
#include "LHCbID.cuh"
#include "SciFiDefinitions.cuh"

namespace SciFi {
  // Bitfields to manipulate triplets
  // Size of lf_triplet must be 64 bits
  struct lf_triplet {
    using t = uint32_t;

    unsigned h2_rel : 6;
    unsigned h1_rel : 6;
    unsigned h0_rel : 6;
    unsigned triplet_seed : 1;
    unsigned left_right_side : 1;
    unsigned chi2 : 12;

    lf_triplet() = default;

    __host__ __device__ lf_triplet(const t i)
    {
      auto* this_t = reinterpret_cast<t*>(this);
      this_t[0] = i;
    }

    __host__ __device__ lf_triplet(
      const unsigned h0,
      const unsigned h1,
      const unsigned h2,
      const unsigned _triplet_seed,
      const unsigned _left_right_side,
      const unsigned _chi2) :
      h2_rel(h2),
      h1_rel(h1), h0_rel(h0), triplet_seed(_triplet_seed), left_right_side(_left_right_side), chi2(_chi2 >> 3)
    {}

    __host__ __device__ operator uint32_t() const { return *reinterpret_cast<const t*>(this); }
  };
  static_assert(sizeof(lf_triplet) == sizeof(lf_triplet::t));

  //----------------------------
  // Struct for hit information.
  //----------------------------
  struct Hit {
    float x0;
    float z0;
    float endPointY;
    unsigned channel;

    // Cluster reference:
    //   raw bank: 8 bits
    //   element (it): 8 bits
    //   Condition 1-2-3: 2 bits
    //   Condition 2.1-2.2: 1 bit
    //   Condition 2.1: log2(n+1) - 8 bits
    unsigned assembled_datatype;

    friend std::ostream& operator<<(std::ostream& stream, const Hit& hit)
    {
      stream << "SciFi hit {" << hit.x0 << ", " << hit.z0 << ", " << hit.channel << ", " << hit.assembled_datatype
             << "}";

      return stream;
    }
  };

  /**
   * @brief Offset and number of hits of each layer.
   */
  template<typename T>
  struct HitCount_t {
  private:
    Allen::forward_type_t<T, unsigned>* m_mat_offsets;
    // TODO: Add "total number of hits" to information of this struct

  public:
    __host__ __device__ HitCount_t(Allen::forward_type_t<T, unsigned>* base_pointer, const unsigned event_number) :
      m_mat_offsets(base_pointer + event_number * SciFi::Constants::n_mat_groups_and_mats)
    {}

    __host__ __device__ void inline set_mat_offsets(const unsigned mat_number, const unsigned value)
    {
      assert(mat_number < SciFi::Constants::n_mats);
      m_mat_offsets[mat_number] = value;
    }

    __host__ __device__ inline unsigned mat_offsets(const unsigned mat_number) const
    {
      assert(
        mat_number >= SciFi::Constants::n_consecutive_raw_banks * SciFi::Constants::n_mats_per_consec_raw_bank &&
        mat_number < SciFi::Constants::n_mats);
      const unsigned corrected_mat_number = mat_number - SciFi::Constants::mat_index_substract;
      return m_mat_offsets[corrected_mat_number];
    }

    __host__ __device__ inline Allen::forward_type_t<T, unsigned>* mat_offsets_p(const unsigned mat_number) const
    {
      return m_mat_offsets + mat_number;
    }

    __host__ __device__ inline unsigned mat_number_of_hits(const unsigned mat_number) const
    {
      assert(mat_number >= SciFi::Constants::n_consecutive_raw_banks * SciFi::Constants::n_mats_per_consec_raw_bank);
      assert(mat_number < SciFi::Constants::n_mats);
      const unsigned corrected_mat_number = mat_number - SciFi::Constants::mat_index_substract;
      return m_mat_offsets[corrected_mat_number + 1] - m_mat_offsets[corrected_mat_number];
    }

    __host__ __device__ inline unsigned mat_group_offset(const unsigned mat_group_number) const
    {
      assert(mat_group_number < SciFi::Constants::n_consecutive_raw_banks);
      return m_mat_offsets[mat_group_number];
    }

    __host__ __device__ inline unsigned mat_group_number_of_hits(const unsigned mat_group_number) const
    {
      assert(mat_group_number < SciFi::Constants::n_consecutive_raw_banks);
      return m_mat_offsets[mat_group_number + 1] - m_mat_offsets[mat_group_number];
    }

    __host__ __device__ inline unsigned mat_group_or_mat_number_of_hits(const unsigned mat_or_mat_group_number) const
    {
      assert(mat_or_mat_group_number < SciFi::Constants::n_mat_groups_and_mats);
      return m_mat_offsets[mat_or_mat_group_number + 1] - m_mat_offsets[mat_or_mat_group_number];
    }

    __host__ __device__ inline unsigned zone_offset(const unsigned zone_number) const
    {
      // TODO: Make this a constant
      // constexpr uint32_t first_corrected_unique_mat_in_zone[] = {
      //   0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640,
      //   688, 736, 784, 832, 880, 928, 976, 1024};
      constexpr uint32_t first_corrected_unique_mat_in_zone[] = {0,   10,  20,  30,  40,  50,  60,  70,  80,
                                                                 90,  100, 110, 120, 130, 140, 150, 160, 208,
                                                                 256, 304, 352, 400, 448, 496, 544};
      return m_mat_offsets[first_corrected_unique_mat_in_zone[zone_number]];
    }

    __host__ __device__ inline unsigned zone_number_of_hits(const unsigned zone_number) const
    {
      return zone_offset(zone_number + 1) - zone_offset(zone_number);
    }

    __host__ __device__ inline unsigned event_number_of_hits() const
    {
      return m_mat_offsets[SciFi::Constants::n_mat_groups_and_mats] - m_mat_offsets[0];
    }

    __host__ __device__ inline unsigned number_of_hits_in_zones_without_mat_groups() const
    {
      return m_mat_offsets[SciFi::Constants::n_mat_groups_and_mats] -
             m_mat_offsets[SciFi::Constants::n_consecutive_raw_banks];
    }

    __host__ __device__ inline unsigned event_offset() const { return m_mat_offsets[0]; }

    __host__ __device__ inline unsigned offset_zones_without_mat_groups() const
    {
      return m_mat_offsets[SciFi::Constants::n_consecutive_raw_banks];
    }
  };

  typedef const HitCount_t<const char> ConstHitCount;
  typedef HitCount_t<char> HitCount;

  template<typename T>
  struct Hits_t {
  private:
    Allen::forward_type_t<T, float>* m_base_pointer;
    const unsigned m_total_number_of_hits;

  public:
    static constexpr unsigned number_of_arrays = 5;

    __host__ __device__ Hits_t(T* base_pointer, const unsigned total_number_of_hits, const unsigned offset = 0) :
      m_base_pointer(reinterpret_cast<Allen::forward_type_t<T, float>*>(base_pointer) + offset),
      m_total_number_of_hits(total_number_of_hits)
    {
      static_assert(sizeof(float) == sizeof(unsigned));
      assert((((size_t) base_pointer) & sizeof(float)) == 0);
    }

    // Const and lvalue accessors
    __host__ __device__ inline float x0(const unsigned index) const
    {
      assert(index < m_total_number_of_hits);
      return m_base_pointer[index];
    }

    __host__ __device__ inline float& x0(const unsigned index)
    {
      assert(index < m_total_number_of_hits);
      return m_base_pointer[index];
    }

    __host__ __device__ inline float z0(const unsigned index) const
    {
      assert(index < m_total_number_of_hits);
      return m_base_pointer[m_total_number_of_hits + index];
    }

    __host__ __device__ inline float& z0(const unsigned index)
    {
      assert(index < m_total_number_of_hits);
      return m_base_pointer[m_total_number_of_hits + index];
    }

    __host__ __device__ inline float endPointY(const unsigned index) const
    {
      assert(index < m_total_number_of_hits);
      return m_base_pointer[2 * m_total_number_of_hits + index];
    }

    __host__ __device__ inline float& endPointY(const unsigned index)
    {
      assert(index < m_total_number_of_hits);
      return m_base_pointer[2 * m_total_number_of_hits + index];
    }

    __host__ __device__ inline unsigned channel(const unsigned index) const
    {

      assert(index < m_total_number_of_hits);
      return reinterpret_cast<Allen::forward_type_t<T, unsigned>*>(m_base_pointer)[3 * m_total_number_of_hits + index];
    }

    __host__ __device__ inline unsigned& channel(const unsigned index)
    {
      assert(index < m_total_number_of_hits);
      return reinterpret_cast<Allen::forward_type_t<T, unsigned>*>(m_base_pointer)[3 * m_total_number_of_hits + index];
    }

    __host__ __device__ inline unsigned assembled_datatype(const unsigned index) const
    {
      assert(index < m_total_number_of_hits);
      return reinterpret_cast<Allen::forward_type_t<T, unsigned>*>(m_base_pointer)[4 * m_total_number_of_hits + index];
    }

    __host__ __device__ inline unsigned& assembled_datatype(const unsigned index)
    {
      assert(index < m_total_number_of_hits);
      return reinterpret_cast<Allen::forward_type_t<T, unsigned>*>(m_base_pointer)[4 * m_total_number_of_hits + index];
    }

    __host__ __device__ inline unsigned id(const unsigned index) const
    {
      return lhcb_id::set_detector_type_id(lhcb_id::LHCbIDType::FT, channel(index));
    }

    __host__ __device__ inline unsigned mat(const unsigned index) const { return assembled_datatype(index) & 0x7ff; };

    __host__ __device__ inline unsigned pseudoSize(const unsigned index) const
    {
      return (assembled_datatype(index) >> 11) & 0xf;
    };

    __host__ __device__ inline unsigned planeCode(const unsigned index) const
    {
      return (assembled_datatype(index) >> 15) & 0x1f;
    };

    __host__ __device__ inline unsigned fraction(const unsigned index) const
    {
      return (assembled_datatype(index) >> 20) & 0x1;
    };

    __host__ __device__ inline Hit get(const unsigned hit_number) const
    {
      return SciFi::Hit {
        x0(hit_number), z0(hit_number), endPointY(hit_number), channel(hit_number), assembled_datatype(hit_number)};
    }

    // Pointer accessor for binary search
    __host__ __device__ inline Allen::forward_type_t<T, float>* x0_p(const unsigned index) const
    {
      return m_base_pointer + index;
    }
  };

  typedef const Hits_t<const char> ConstHits;
  typedef Hits_t<char> Hits;

  template<typename T>
  struct ExtendedHits_t : public Hits_t<T> {
  private:
    const float* m_inv_clus_res;
    const SciFiGeometry* m_geom;

  public:
    __host__ __device__ ExtendedHits_t(
      T* base_pointer,
      const unsigned total_number_of_hits,
      const float* inv_clus_res,
      const SciFiGeometry* geom,
      const unsigned offset = 0) :
      Hits_t<T>(base_pointer, total_number_of_hits, offset),
      m_inv_clus_res(inv_clus_res), m_geom(geom)
    {}

    using Hits_t<T>::pseudoSize;
    using Hits_t<T>::endPointY;
    using Hits_t<T>::channel;
    using Hits_t<T>::mat;

    // Additional accessors provided by having inv clus res and geometry information
    __host__ __device__ inline float w(const unsigned index) const
    {
      assert(pseudoSize(index) < 9 && "Wrong pseudo size.");
      const auto werrX = m_inv_clus_res[pseudoSize(index)];
      return werrX * werrX;
    };

    __host__ __device__ inline float dxdy(const unsigned index) const { return m_geom->dxdy[mat(index)]; };

    __host__ __device__ inline float dzdy(const unsigned index) const { return m_geom->dzdy[mat(index)]; };

    __host__ __device__ inline float yMin(const unsigned index) const
    {
      const SciFiChannelID id(channel(index));
      return endPointY(index) + id.isBottom() * m_geom->globaldy[mat(index)];
    };

    __host__ __device__ inline float yMax(const unsigned index) const
    {
      const SciFiChannelID id(channel(index));
      return endPointY(index) + !id.isBottom() * m_geom->globaldy[mat(index)];
    };

    // Deprecated code?
    // __host__ __device__ float endPointY(const unsigned index) const
    // {
    //   const SciFiChannelID id(channel[index]);
    //   float uFromChannel =
    //     m_geom->uBegin[mat(index)] + (2 * id.channel() + 1 + fraction(index)) * m_geom->halfChannelPitch[mat(index)];
    //   if (id.die()) uFromChannel += m_geom->dieGap[mat(index)];
    //   return m_geom->mirrorPointY[mat(index)] + m_geom->ddxY[mat(index)] * uFromChannel;
    // }
  };

  typedef const ExtendedHits_t<const char> ConstExtendedHits;
  typedef ExtendedHits_t<char> ExtendedHits;

  /**
   * Track object used for storing tracks
   */
  struct TrackCandidate {
    float quality = 0.f;
    float qop;
    uint16_t input_track_index;
    uint16_t hits[SciFi::Constants::max_track_candidate_size];
    uint8_t hitsNum = 0;

    TrackCandidate() = default;
    TrackCandidate(const TrackCandidate&) = default;

    __host__ __device__ TrackCandidate(
      const uint16_t h0,
      const uint16_t h1,
      const uint16_t param_input_track_index,
      const float param_qop) :
      quality(0.f),
      qop(param_qop), input_track_index(param_input_track_index), hitsNum(2)
    {
      hits[0] = h0;
      hits[1] = h1;
    };

    __host__ __device__ inline void add_hit(uint16_t hit_index)
    {
      assert(hitsNum < SciFi::Constants::max_track_candidate_size);
      hits[hitsNum++] = hit_index;
    }

    __host__ __device__ inline void add_hit_with_quality(uint16_t hit_index, float chi2)
    {
      assert(hitsNum < SciFi::Constants::max_track_candidate_size);
      hits[hitsNum++] = hit_index;
      quality += chi2;
    }
  };

  /**
   * Track object used for storing tracks
   */
  struct TrackHits {
    float quality = 0.f;
    float qop;
    uint16_t input_track_index;
    uint16_t charge_seed;
    uint16_t hits[SciFi::Constants::max_track_size];
    uint8_t hitsNum = 0;

    TrackHits() = default;
    TrackHits(const TrackHits&) = default;
    TrackHits& operator=(const TrackHits&) = default;

    __host__ __device__ TrackHits(const TrackCandidate& candidate) :
      quality(candidate.quality), qop(candidate.qop), input_track_index(candidate.input_track_index),
      hitsNum(candidate.hitsNum)
    {
      for (int i = 0; i < hitsNum; ++i) {
        hits[i] = candidate.hits[i];
      }
    }

    __host__ __device__ TrackHits(
      const uint16_t h0,
      const uint16_t h1,
      const uint16_t h2,
      const float chi2,
      const float qop,
      const uint16_t input_track_index,
      const uint16_t charge_seed) :
      quality(chi2),
      qop(qop), input_track_index(input_track_index), charge_seed(charge_seed)
    {
      hitsNum = 3;
      hits[0] = h0;
      hits[1] = h1;
      hits[2] = h2;
    }

    __host__ __device__ TrackHits(
      const uint16_t h0,
      const uint16_t h1,
      const uint16_t h2,
      const uint16_t layer_h0,
      const uint16_t layer_h1,
      const uint16_t layer_h2,
      const float chi2,
      const float qop,
      const uint16_t input_track_index,
      const uint16_t charge_seed) :
      quality(chi2),
      qop(qop), input_track_index(input_track_index), charge_seed(charge_seed)
    {
      hitsNum = 3;
      hits[0] = h0;
      hits[1] = h1;
      hits[2] = h2;
      hits[SciFi::Constants::hit_layer_offset] = layer_h0;
      hits[SciFi::Constants::hit_layer_offset + 1] = layer_h1;
      hits[SciFi::Constants::hit_layer_offset + 2] = layer_h2;
    }

    __host__ __device__ inline uint16_t get_layer(uint8_t index) const
    {
      assert(hitsNum <= SciFi::Constants::hit_layer_offset);
      return hits[SciFi::Constants::hit_layer_offset + index];
    }

    __host__ __device__ inline void add_hit(uint16_t hit_index)
    {
      assert(hitsNum < SciFi::Constants::max_track_size);
      hits[hitsNum++] = hit_index;
    }

    __host__ __device__ inline void add_hit_with_quality(uint16_t hit_index, float chi2)
    {
      assert(hitsNum < SciFi::Constants::max_track_size);
      hits[hitsNum++] = hit_index;
      quality += chi2;
    }

    __host__ __device__ inline void add_hit_with_layer_and_quality(uint16_t hit_index, uint16_t layer, float chi2)
    {
      assert(hitsNum < SciFi::Constants::max_track_size);
      hits[hitsNum] = hit_index;
      hits[SciFi::Constants::hit_layer_offset + hitsNum++] = layer;
      quality += chi2;
    }

    __host__ __device__ inline float get_quality() const
    {
      assert(hitsNum > 2);
      return quality / ((float) hitsNum - 2);
    }

    __host__ __device__ inline void print(int event_number = -1) const
    {
      printf("Track with %i hits:", hitsNum);
      for (int i = 0; i < hitsNum; ++i) {
        printf(" %i,", hits[i]);
      }
      printf(
        " qop %f, quality %f, UT track %i", static_cast<double>(qop), static_cast<double>(quality), input_track_index);
      if (event_number >= 0) {
        printf(" (event %i)", event_number);
      }
      printf("\n");
    }
  };

  struct LongCheckerTrack {
    using LHCbID = unsigned;
    LHCbID allids[42];
    unsigned total_number_of_hits = 0;
    unsigned velo_track_index = 0;
    float p = 0.f, pt = 0.f, rho = 0.f, qop = 0.f;
  };
  struct KalmanCheckerTrack {
    using LHCbID = unsigned;
    LHCbID allids[42];
    unsigned total_number_of_hits = 0;
    unsigned velo_track_index = 0;
    // Kalman information.
    float z = 0.f, x = 0.f, y = 0.f, tx = 0.f, ty = 0.f, qop = 0.f;
    float first_qop = 0.f, best_qop = 0.f;
    float chi2 = 0.f, chi2V = 0.f, chi2T = 0.f;
    unsigned ndof = 0, ndofV = 0, ndofT = 0;
    float kalman_ip = 0.f, kalman_ip_chi2 = 0.f, kalman_ipx = 0.f, kalman_ipy = 0.f;
    float kalman_docaz = 0.f;
    float velo_ip = 0.f, velo_ip_chi2 = 0.f, velo_ipx = 0.f, velo_ipy = 0.f;
    float velo_docaz = 0.f;
    float long_ip = 0.f, long_ip_chi2 = 0.f, long_ipx = 0.f, long_ipy = 0.f;
    std::size_t n_matched_total = 0;
    float p = 0.f, pt = 0.f, rho = 0.f;
  };
  namespace Seeding {
    struct TrackXZ {
      int number_of_hits;
      unsigned int ids[6];
      int idx[6];
      unsigned int hits[6];
      float chi2;
      float ax;
      float bx;
      float cx;
    };

    struct Track {
      int number_of_hits = 0;
      // unsigned int ids[SciFi::Constants::n_layers] = {SciFi::Constants::INVALID_ID};
      unsigned int hits[SciFi::Constants::n_layers] = {SciFi::Constants::INVALID_ID};
      float chi2X, chi2Y;
      float ax;
      float bx;
      float cx;
      float ay;
      float by;

      __host__ __device__ float x(float z) const
      {
        const float dz = z - SciFi::Constants::z_mid_t;
        return ax + dz * (bx + dz * cx * (1.f + SciFi::Constants::dRatio * dz));
      }
      __host__ __device__ float y(float z) const { return (ay + by * (z - SciFi::Constants::z_mid_t)); }

      __host__ __device__ float xFromDz(float dz) const
      {
        return ax + dz * (bx + dz * cx * (1.f + SciFi::Constants::dRatio * dz));
      }
      __host__ __device__ float yFromDz(float dz) const { return ay + by * dz; }
      __host__ __device__ float xSlopeFromDz(float dz) const
      {
        return bx + 2.f * dz * cx + 3.f * dz * dz * cx * SciFi::Constants::dRatio;
      }
      __host__ __device__ float ySlope() const { return by; }
    };
  } // namespace Seeding

  struct MatchedTrack {
    uint16_t velo_track_index;
    uint16_t scifi_track_index;
    int number_of_hits_velo = 0;
    int number_of_hits_ut = 0;
    int number_of_hits_scifi = 0;
    float chi2_matching;
    float qop;
  };
} // namespace SciFi
