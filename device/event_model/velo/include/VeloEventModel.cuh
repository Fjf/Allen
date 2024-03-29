/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <stdint.h>
#include "BackendCommon.h"
#include "VeloDefinitions.cuh"
#include "Vector.h"

namespace Velo {
  struct ModulePair {
    unsigned hit_start;
    unsigned hit_num;
    // z of both modules in module pair
    float z[2];

    __device__ ModulePair() {}

    __device__ ModulePair(
      const unsigned param_hit_start,
      const unsigned param_hit_num,
      const float param_z0,
      const float param_z1) :
      hit_start(param_hit_start),
      hit_num(param_hit_num), z {param_z0, param_z1}
    {}
  };

  struct HitBase { // 3 * 4 = 16 B
    float x;
    float y;
    float z;

    __host__ __device__ HitBase() {}

    __host__ __device__ HitBase(const float _x, const float _y, const float _z) : x(_x), y(_y), z(_z) {}
  };

  struct Hit : public HitBase { // 4 * 4 = 16 B
    unsigned LHCbID;

    __host__ __device__ Hit() {}

    __host__ __device__ Hit(const float _x, const float _y, const float _z, const unsigned _LHCbID) :
      HitBase(_x, _y, _z), LHCbID(_LHCbID)
    {}
  };

  /**
   * @brief TrackletHits struct
   */
  struct TrackletHits { // 3 * 2 = 6 B
    uint16_t hits[3];

    __host__ __device__ TrackletHits() {}

    __host__ __device__ TrackletHits(const uint16_t h0, const uint16_t h1, const uint16_t h2)
    {
      hits[0] = h0;
      hits[1] = h1;
      hits[2] = h2;
    }
  };

  /* Structure containing indices to hits within hit array */
  struct TrackHits { // 1 + 26 * 2 = 53 B
    uint16_t hits[Velo::Constants::max_track_size];
    uint8_t hitsNum = 3;

    __host__ __device__ TrackHits() {}

    __host__ __device__ TrackHits(const uint16_t _h0, const uint16_t _h1, const uint16_t _h2)
    {
      hits[0] = _h0;
      hits[1] = _h1;
      hits[2] = _h2;
    }

    __host__ __device__ TrackHits(const TrackletHits& tracklet)
    {
      hits[0] = tracklet.hits[0];
      hits[1] = tracklet.hits[1];
      hits[2] = tracklet.hits[2];
    }
  };

  /**
   * @brief Structure to access VELO clusters.
   */
  template<typename T>
  struct Clusters_t {
  protected:
    typename ForwardType<T, half_t>::t* m_base_pointer;
    unsigned m_total_number_of_hits;
    unsigned m_offset;

  public:
    constexpr static unsigned element_size = sizeof(unsigned) + sizeof(int16_t) + 3 * sizeof(half_t);
    constexpr static unsigned offset_coordinates = sizeof(unsigned) / sizeof(half_t);
    constexpr static unsigned offset_phi = (sizeof(unsigned) + 3 * sizeof(half_t)) / sizeof(int16_t);

    Clusters_t() = default;
    Clusters_t(const Clusters_t&) = default;
    Clusters_t& operator=(const Clusters_t&) = default;

    __host__ __device__
    Clusters_t(T* base_pointer, const unsigned total_estimated_number_of_clusters, const unsigned offset = 0) :
      m_base_pointer(reinterpret_cast<typename ForwardType<T, half_t>::t*>(base_pointer)),
      m_total_number_of_hits(total_estimated_number_of_clusters), m_offset(offset)
    {}

    __host__ __device__ unsigned id(const unsigned index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return reinterpret_cast<typename ForwardType<T, unsigned>::t*>(m_base_pointer)[m_offset + index];
    }

    __host__ __device__ void set_id(const unsigned index, const unsigned value)
    {
      assert(m_offset + index < m_total_number_of_hits);
      reinterpret_cast<typename ForwardType<T, unsigned>::t*>(m_base_pointer)[m_offset + index] = value;
    }

    __host__ __device__ float x(const unsigned index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return static_cast<typename ForwardType<T, float>::t>(
        m_base_pointer[offset_coordinates * m_total_number_of_hits + 3 * (m_offset + index)]);
    }

    __host__ __device__ void set_x(const unsigned index, const half_t value)
    {
      assert(m_offset + index < m_total_number_of_hits);
      m_base_pointer[offset_coordinates * m_total_number_of_hits + 3 * (m_offset + index)] = value;
    }

    __host__ __device__ float y(const unsigned index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return static_cast<typename ForwardType<T, float>::t>(
        m_base_pointer[offset_coordinates * m_total_number_of_hits + 3 * (m_offset + index) + 1]);
    }

    __host__ __device__ void set_y(const unsigned index, const half_t value)
    {
      assert(m_offset + index < m_total_number_of_hits);
      m_base_pointer[offset_coordinates * m_total_number_of_hits + 3 * (m_offset + index) + 1] = value;
    }

    __host__ __device__ float z(const unsigned index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return static_cast<typename ForwardType<T, float>::t>(
        m_base_pointer[offset_coordinates * m_total_number_of_hits + 3 * (m_offset + index) + 2]);
    }

    __host__ __device__ void set_z(const unsigned index, const half_t value)
    {
      assert(m_offset + index < m_total_number_of_hits);
      m_base_pointer[offset_coordinates * m_total_number_of_hits + 3 * (m_offset + index) + 2] = value;
    }

    __host__ __device__ int16_t phi(const unsigned index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return reinterpret_cast<typename ForwardType<T, int16_t>::t*>(
        m_base_pointer)[m_total_number_of_hits * offset_phi + m_offset + index];
    }

    __host__ __device__ void set_phi(const unsigned index, const int16_t value)
    {
      assert(m_offset + index < m_total_number_of_hits);
      reinterpret_cast<typename ForwardType<T, int16_t>::t*>(
        m_base_pointer)[m_total_number_of_hits * offset_phi + m_offset + index] = value;
    }

    // Pointer accessor for binary search
    __host__ __device__ typename ForwardType<T, int16_t>::t* phi_begin() const
    {
      return reinterpret_cast<typename ForwardType<T, int16_t>::t*>(m_base_pointer) +
             m_total_number_of_hits * offset_phi + m_offset;
    }
  };

  typedef const Clusters_t<const char> ConstClusters;
  typedef Clusters_t<char> Clusters;

  /**
   * @brief Returns the track offset of an event.
   */
  __host__ __device__ inline unsigned track_offset(const unsigned* offsets, const unsigned event_number)
  {
    const auto offset_event = offsets[event_number * Velo::Constants::n_module_pairs];
    return offset_event * Velo::Constants::max_number_of_tracks_per_cluster;
  }
} // namespace Velo
