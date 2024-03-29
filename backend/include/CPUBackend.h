/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/

#pragma once

#ifdef TARGET_DEVICE_CPU

#include <stdexcept>
#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <regex>

#ifdef __linux__
#include <ext/stdio_filebuf.h>
#endif

using std::copysignf;
using std::max;
using std::min;
using std::signbit;

#define __host__
#define __device__
#define __shared__
#define __global__
#define __constant__
#define __syncthreads()
#define __syncwarp()
#define __launch_bounds__(_i)
#define __popc __builtin_popcount
#define __popcll __builtin_popcountll
#define __ffs __builtin_ffs
#define __clz __builtin_clz
#define __forceinline__ inline
#define copysignf_impl copysignf
#define fmaxf_impl fmaxf
#define fminf_impl fminf
constexpr int warp_size = 1;

unsigned inline __ballot_sync(unsigned mask, int predicate) { return predicate & mask; }

// Support for dynamic shared memory buffers
#define DYNAMIC_SHARED_MEMORY_BUFFER(_type, _instance, _config)                                                  \
  auto _dynamic_shared_memory_buffer = std::vector<_type>(_config.dynamic_shared_memory_size() / sizeof(_type)); \
  auto _instance = _dynamic_shared_memory_buffer.data();

struct float3 {
  float x;
  float y;
  float z;
};

struct float2 {
  float x;
  float y;
};

struct dim3 {
  unsigned int x = 1;
  unsigned int y = 1;
  unsigned int z = 1;

  dim3() = default;
  dim3(const dim3&) = default;
  dim3(const unsigned int& x) : x(x) {}
  dim3(const unsigned int& x, const unsigned int& y) : x(x), y(y) {}
  dim3(const unsigned int& x, const unsigned int& y, const unsigned int& z) : x(x), y(y), z(z) {}
};

struct GridDimensions {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

struct BlockIndices {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

struct BlockDimensions {
  unsigned int x = 1;
  unsigned int y = 1;
  unsigned int z = 1;
};

struct ThreadIndices {
  unsigned int x = 0;
  unsigned int y = 0;
  unsigned int z = 0;
};

extern thread_local GridDimensions gridDim;
extern thread_local BlockIndices blockIdx;
constexpr BlockDimensions blockDim {1, 1, 1};
constexpr ThreadIndices threadIdx {0, 0, 0};

/**
 * @brief Emulates atomic operations so they work on the CPU backend.
 * @details The CPU backend executes sequentially, so there
 *          is no need to provide atomicity in any of the
 *          implementations below.
 */
template<class T, class S>
T atomicAdd(T* address, S val)
{
  const T old = *address;
  *address += val;
  return old;
}

template<class T, class S>
T atomicOr(T* address, S val)
{
  const T old = *address;
  *address |= val;
  return old;
}

inline unsigned int atomicInc(unsigned int* address, unsigned int val)
{
  unsigned int old = *address;
  *address = ((old >= val) ? 0 : (old + 1));
  return old;
}

template<class T>
inline T atomicMin(T* address, T val)
{
  const int old = *address;
  *address = std::min(old, val);
  return old;
}

template<class T>
inline T atomicMax(T* address, T val)
{
  const T old = *address;
  *address = std::max(old, val);
  return old;
}

uint16_t __float2half(const float f);

float __half2float(const uint16_t h);

#ifdef CPU_USE_REAL_HALF

/**
 * @brief half_t with int16_t backend (real half).
 */
struct half_t {
private:
  uint16_t m_value;

public:
  half_t() = default;
  half_t(const half_t&) = default;

  half_t(const float f) { m_value = __float2half(f); }

  inline operator float() const { return __half2float(m_value); }

  inline bool operator<(const half_t& a) const
  {
    const auto sign = (m_value >> 15) & 0x01;
    const auto sign_a = (a.get() >> 15) & 0x01;
    return (sign & sign_a & operator!=(a)) ^ (m_value < a.get());
  }

  inline bool operator>(const half_t& a) const
  {
    const auto sign = (m_value >> 15) & 0x01;
    const auto sign_a = (a.get() >> 15) & 0x01;
    return (sign & sign_a & operator!=(a)) ^ (m_value > a.get());
  }

  inline bool operator<=(const half_t& a) const { return !operator>(a); }

  inline bool operator>=(const half_t& a) const { return !operator<(a); }

  inline bool operator==(const half_t& a) const { return m_value == a.get(); }

  inline bool operator!=(const half_t& a) const { return !operator==(a); }
};

#else

/**
 * @brief half_t with float backend.
 */
using half_t = float;

#endif

namespace Allen {
  struct KernelInvocationConfiguration {
  private:
    unsigned m_dynamic_shared_memory_size = 0;

  public:
    KernelInvocationConfiguration() = default;

    KernelInvocationConfiguration(const dim3&, const dim3&, const unsigned dynamic_shared_memory_size) :
      m_dynamic_shared_memory_size(dynamic_shared_memory_size)
    {}

    unsigned dynamic_shared_memory_size() const { return m_dynamic_shared_memory_size; }
  };

  // Big enough alignment to align with 512-bit vectors
  constexpr static unsigned cpu_alignment = 64;

  struct Context {
    void initialize() {}
  };

  void inline malloc(void** devPtr, size_t size)
  {
    if (posix_memalign(devPtr, cpu_alignment, size) != 0) *devPtr = nullptr;
  }

  void inline malloc_host(void** ptr, size_t size) { malloc(ptr, size); }

  void inline memcpy(void* dst, const void* src, size_t count, enum Allen::memcpy_kind)
  {
    std::memcpy(dst, src, count);
  }

  void inline memcpy_async(void* dst, const void* src, size_t count, enum Allen::memcpy_kind kind, const Context&)
  {
    memcpy(dst, src, count, kind);
  }

  void inline memset(void* devPtr, int value, size_t count) { std::memset(devPtr, value, count); }

  void inline memset_async(void* ptr, int value, size_t count, const Context&) { memset(ptr, value, count); }

  void inline free_host(void* ptr) { ::free(ptr); }

  void inline free(void* ptr) { free_host(ptr); }

  void inline synchronize(const Context&) {}

  void inline device_reset() {}

  void inline peek_at_last_error() {}

  void inline host_unregister(void*) {}

  void inline host_register(void*, size_t, host_register_kind) {}

  std::tuple<bool, std::string, unsigned> set_device(int, size_t);

  void inline print_device_memory_consumption() {}

  std::tuple<bool, int> inline get_device_id(const std::string&) { return {true, 0}; }

  namespace device {
    template<class To, class From>
    std::enable_if_t<
      sizeof(To) == sizeof(From) && alignof(To) == alignof(From) && std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>,
      To>
    bit_cast(const From& src) noexcept
    {
      To dst;
      std::memcpy(&dst, &src, sizeof(To));
      return dst;
    }
  } // namespace device
} // namespace Allen

inline float __int_as_float(int a) { return Allen::device::bit_cast<float>(a); }

inline int __float_as_int(float a) { return Allen::device::bit_cast<int>(a); }

#endif
