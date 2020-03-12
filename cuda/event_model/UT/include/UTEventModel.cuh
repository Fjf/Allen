#pragma once

#include <cstdint>
#include <ostream>
#include "UTDefinitions.cuh"

namespace UT {
  // Hit base containing just the geometrical information about the hit.
  struct Hit {
    float yBegin;
    float yEnd;
    float zAtYEq0;
    float xAtYEq0;
    float weight;
    uint32_t LHCbID;
    uint8_t plane_code;

    __device__ Hit() {}

    __device__ Hit(
      const float _yBegin,
      const float _yEnd,
      const float _zAtYEq0,
      const float _xAtYEq0,
      const float _weight,
      const uint32_t _LHCbID,
      const uint8_t _plane_code) :
      yBegin(_yBegin),
      yEnd(_yEnd), zAtYEq0(_zAtYEq0), xAtYEq0(_xAtYEq0), weight(_weight), LHCbID(_LHCbID), plane_code(_plane_code)
    {}

    bool operator==(const Hit& h) const { return LHCbID == h.LHCbID; }

    bool operator!=(const Hit& h) const { return !operator==(h); }

    friend std::ostream& operator<<(std::ostream& stream, const Hit& ut_hit)
    {
      stream << "UT hit {" << ut_hit.LHCbID << ", " << ut_hit.yBegin << ", " << ut_hit.yEnd << ", " << ut_hit.zAtYEq0
             << ", " << ut_hit.xAtYEq0 << ", " << ut_hit.weight << ut_hit.plane_code << "}";

      return stream;
    }
  };

  struct TrackHits {
    float qop;
    float x, z;
    float tx;
    short hits[UT::Constants::max_track_size];
    unsigned short hits_num = 0;
    unsigned short velo_track_index;
  };

  /**
   * @brief Offset and number of hits of each layer.
   */
  struct HitOffsets {
    const uint* m_unique_x_sector_layer_offsets;
    const uint* m_ut_hit_offsets;
    const uint m_number_of_unique_x_sectors;

    __device__ __host__ HitOffsets(
      const uint* base_pointer,
      const uint event_number,
      const uint number_of_unique_x_sectors,
      const uint* unique_x_sector_layer_offsets) :
      m_unique_x_sector_layer_offsets(unique_x_sector_layer_offsets),
      m_ut_hit_offsets(base_pointer + event_number * number_of_unique_x_sectors),
      m_number_of_unique_x_sectors(number_of_unique_x_sectors)
    {}

    __device__ __host__ uint sector_group_offset(const uint sector_group) const
    {
      assert(sector_group <= m_number_of_unique_x_sectors);
      return m_ut_hit_offsets[sector_group];
    }

    __device__ __host__ uint sector_group_number_of_hits(const uint sector_group) const
    {
      assert(sector_group < m_number_of_unique_x_sectors);
      return m_ut_hit_offsets[sector_group + 1] - m_ut_hit_offsets[sector_group];
    }

    __device__ __host__ uint layer_offset(const uint layer_number) const
    {
      assert(layer_number < 4);
      return m_ut_hit_offsets[m_unique_x_sector_layer_offsets[layer_number]];
    }

    __device__ __host__ uint layer_number_of_hits(const uint layer_number) const
    {
      assert(layer_number < 4);
      return m_ut_hit_offsets[m_unique_x_sector_layer_offsets[layer_number + 1]] -
             m_ut_hit_offsets[m_unique_x_sector_layer_offsets[layer_number]];
    }

    __device__ __host__ uint event_offset() const { return m_ut_hit_offsets[0]; }

    __device__ __host__ uint event_number_of_hits() const
    {
      return m_ut_hit_offsets[m_number_of_unique_x_sectors] - m_ut_hit_offsets[0];
    }
  };

  /*
     SoA for hit variables
     The hits for every layer are written behind each other, the offsets
     are stored for access;
     one Hits structure exists per event
  */
  template<typename T>
  struct Hits_t {
  protected:
    typename ForwardType<T, float>::t* m_base_pointer;
    const uint m_total_number_of_hits;
    const uint m_offset;

  public:
    constexpr static uint element_size = 5 * sizeof(float) + sizeof(uint);
    /**
     * @brief Populates the UTHits object pointers to an array of data
     *        pointed by base_pointer.
     */
    __host__ __device__ Hits_t(T* base_pointer, const uint total_number_of_hits, const uint offset = 0) :
      m_base_pointer(reinterpret_cast<typename ForwardType<T, float>::t*>(base_pointer)),
      m_total_number_of_hits(total_number_of_hits), m_offset(offset)
    {}

    // Const and lvalue accessors
    __host__ __device__ float yBegin(const uint index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return m_base_pointer[m_offset + index];
    }

    __host__ __device__ float& yBegin(const uint index)
    {
      assert(m_offset + index < m_total_number_of_hits);
      return m_base_pointer[m_offset + index];
    }

    __host__ __device__ float yEnd(const uint index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return m_base_pointer[m_offset + m_total_number_of_hits + index];
    }

    __host__ __device__ float& yEnd(const uint index)
    {
      assert(m_offset + index < m_total_number_of_hits);
      return m_base_pointer[m_offset + m_total_number_of_hits + index];
    }

    __host__ __device__ float zAtYEq0(const uint index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return m_base_pointer[m_offset + 2 * m_total_number_of_hits + index];
    }

    __host__ __device__ float& zAtYEq0(const uint index)
    {
      assert(m_offset + index < m_total_number_of_hits);
      return m_base_pointer[m_offset + 2 * m_total_number_of_hits + index];
    }

    __host__ __device__ float xAtYEq0(const uint index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return m_base_pointer[m_offset + 3 * m_total_number_of_hits + index];
    }

    __host__ __device__ float& xAtYEq0(const uint index)
    {
      assert(m_offset + index < m_total_number_of_hits);
      return m_base_pointer[m_offset + 3 * m_total_number_of_hits + index];
    }

    __host__ __device__ float weight(const uint index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return m_base_pointer[m_offset + 4 * m_total_number_of_hits + index];
    }

    __host__ __device__ float& weight(const uint index)
    {
      assert(m_offset + index < m_total_number_of_hits);
      return m_base_pointer[m_offset + 4 * m_total_number_of_hits + index];
    }

    __host__ __device__ uint id(const uint index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return reinterpret_cast<typename ForwardType<T, uint>::t*>(
        m_base_pointer)[m_offset + 5 * m_total_number_of_hits + index];
    }

    __host__ __device__ uint& id(const uint index)
    {
      assert(m_offset + index < m_total_number_of_hits);
      return reinterpret_cast<typename ForwardType<T, uint>::t*>(
        m_base_pointer)[m_offset + 5 * m_total_number_of_hits + index];
    }

    /**
     * @brief Gets a hit in the UT::Hit format from the global hit index.
     */
    __host__ __device__ Hit getHit(const uint index) const
    {
      return {yBegin(index), yEnd(index), zAtYEq0(index), xAtYEq0(index), weight(index), id(index), 0};
    }

    __host__ __device__ bool isYCompatible(const uint index, const float y, const float tol) const
    {
      return yMin(index) - tol <= y && y <= yMax(index) + tol;
    }

    __host__ __device__ bool isNotYCompatible(const uint index, const float y, const float tol) const
    {
      return yMin(index) - tol > y || y > yMax(index) + tol;
    }

    __host__ __device__ float cosT(const uint index, const float dxDy) const
    {
      return (fabsf(xAtYEq0(index)) < 1.0e-9f) ? 1.f / sqrtf(1.f + dxDy * dxDy) : cosf(dxDy);
    }

    __host__ __device__ float sinT(const uint index, const float dxDy) const { return tanT(dxDy) * cosT(index, dxDy); }

    __host__ __device__ float tanT(const float dxDy) const { return -1.f * dxDy; }

    __host__ __device__ float xAt(const uint index, const float globalY, const float dxDy) const
    {
      return xAtYEq0(index) + globalY * dxDy;
    }

    __host__ __device__ float yMax(const uint index) const { return fmaxf(yBegin(index), yEnd(index)); }

    __host__ __device__ float yMid(const uint index) const { return 0.5f * (yBegin(index) + yEnd(index)); }

    __host__ __device__ float yMin(const uint index) const { return fminf(yBegin(index), yEnd(index)); }

    // Pointer accessors for binary search
    __host__ __device__ typename ForwardType<T, float>::t* yBegin_p(const uint index) const
    {
      assert(m_offset + index < m_total_number_of_hits);
      return m_base_pointer + m_offset + index;
    }

    __host__ __device__ typename ForwardType<T, float>::t* yEnd_p(const uint index) const
    {
      assert(m_offset + index <= m_total_number_of_hits);
      return m_base_pointer + m_offset + m_total_number_of_hits + index;
    }
  };

  typedef const Hits_t<const char> ConstHits;
  typedef Hits_t<char> Hits;

  /**
   * @brief Pre decoded hits datatype
   * @details This datatype is used for the predecoding steps of the UT.
   */
  template<typename T>
  struct PreDecodedHits_t {
  private:
    typename ForwardType<T, float>::t* m_base_pointer;
    const uint m_total_number_of_hits;

  public:
    constexpr static uint element_size = sizeof(float) + sizeof(uint);

    /**
     * @brief Populates the UTHits object pointers to an array of data
     *        pointed by base_pointer.
     */
    __host__ __device__ PreDecodedHits_t(T* base_pointer, const uint total_number_of_hits) :
      m_base_pointer(reinterpret_cast<typename ForwardType<T, float>::t*>(base_pointer)),
      m_total_number_of_hits(total_number_of_hits)
    {}

    // Const and lvalue accessors
    __host__ __device__ float sort_key(const uint index) const
    {
      assert(index < m_total_number_of_hits);
      return m_base_pointer[index];
    }

    __host__ __device__ float& sort_key(const uint index)
    {
      assert(index < m_total_number_of_hits);
      return m_base_pointer[index];
    }

    __host__ __device__ uint index(const uint index) const
    {
      assert(index < m_total_number_of_hits);
      return reinterpret_cast<typename ForwardType<T, uint>::t*>(m_base_pointer)[m_total_number_of_hits + index];
    }

    __host__ __device__ uint& index(const uint index)
    {
      assert(index < m_total_number_of_hits);
      return reinterpret_cast<typename ForwardType<T, uint>::t*>(m_base_pointer)[m_total_number_of_hits + index];
    }

    // Pointer accessors for binary search
    __host__ __device__ typename ForwardType<T, float>::t* sort_key_p(const uint index) const
    {
      assert(index < m_total_number_of_hits);
      return m_base_pointer + index;
    }
  };

  typedef const PreDecodedHits_t<const char> ConstPreDecodedHits;
  typedef PreDecodedHits_t<char> PreDecodedHits;
} // namespace UT
