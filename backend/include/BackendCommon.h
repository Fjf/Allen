/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/

#pragma once

// Host / device compiler identification
#if defined(TARGET_DEVICE_CPU) || (defined(TARGET_DEVICE_CUDA) && defined(__CUDACC__)) || \
  (defined(TARGET_DEVICE_HIP) && (defined(__HCC__) || defined(__HIP__)))
#define DEVICE_COMPILER
#endif

#if defined(TARGET_DEVICE_CUDA) || defined(TARGET_DEVICE_HIP)
#define TARGET_DEVICE_CUDAHIP
#endif

#include <tuple>
#include <string>
#include <cassert>
#include <array>
#include <gsl/gsl>
#include "AllenTypeTraits.h"
#include "BackendCommonInterface.h"

// Dispatch to the right backend
#if defined(TARGET_DEVICE_CPU)
#include "CPUBackend.h"
#elif defined(TARGET_DEVICE_HIP)
#include "HIPBackend.h"
#elif defined(TARGET_DEVICE_CUDA)
#include "CUDABackend.h"
#endif

#if defined(DEVICE_COMPILER)
namespace Allen {
  namespace device {
    // Dispatcher targets
    namespace target {
      struct Default {
      };
      struct CPU {
      };
      struct HIP {
      };
      struct CUDA {
      };
    } // namespace target

    /**
     * @brief Allows to write several functions specialized per target.
     * @details Usage:
     *
     *          dispatch<target::Default, target::CPU>(fn0, fn1)(arguments...);
     *
     *          The concrete target on the executing platform is sought first. If none is
     *          available, the default one is chosen. If the default doesn't exist, a static_assert
     *          fails.
     *
     *          List of possible targets:
     *
     *          * Default
     *          * CPU
     *          * HIP
     *          * CUDA
     */
    template<typename... Ts, typename... Fns>
    __device__ constexpr auto dispatch(Fns&&... fns)
    {
      using targets_t = std::tuple<Ts...>;
#if !defined(ALWAYS_DISPATCH_TO_DEFAULT)
#if defined(TARGET_DEVICE_CPU)
      constexpr auto configured_target = index_of_v<target::CPU, targets_t>;
#elif defined(TARGET_DEVICE_CUDA)
      constexpr auto configured_target = index_of_v<target::CUDA, targets_t>;
#elif defined(TARGET_DEVICE_HIP)
      constexpr auto configured_target = index_of_v<target::HIP, targets_t>;
#endif
      if constexpr (configured_target == std::tuple_size<targets_t>::value) {
#endif
        // Dispatch to the default target, check with a static_assert its existence
        constexpr auto default_target = index_of_v<target::Default, targets_t>;
        static_assert(default_target != std::tuple_size<targets_t>::value, "target available for current platform");
        const auto fn = std::get<default_target>(std::tuple<Fns...> {fns...});
        return [fn](auto&&... args) { return fn(args...); };
#if !defined(ALWAYS_DISPATCH_TO_DEFAULT)
      }
      else {
        // Dispatch to the specific target
        const auto fn = std::get<configured_target>(std::tuple<Fns...> {fns...});
        return [fn](auto&&... args) { return fn(args...); };
      }
#endif
    }
  } // namespace device
} // namespace Allen
#else
#include <cmath>
#endif

// Replacement for gsl::span in device code when building with HIP,
// gsl::span works for CUDA and CPU
namespace Allen::device {
#if defined(TARGET_DEVICE_HIP)
  template<class T>
  struct span {
  private:
    T* m_ptr = nullptr;
    std::size_t m_size = 0;

  public:
    constexpr span() = default;

    constexpr __device__ __host__ span(T* ptr, std::size_t size) : m_ptr(ptr), m_size(size) {}

    template<std::size_t N>
    constexpr __device__ __host__ span(std::array<T, N>& a) : m_ptr(std::data(a)), m_size(N)
    {}

    template<std::size_t N>
    constexpr __device__ __host__ span(const std::array<std::remove_const_t<T>, N>& a) : m_ptr(std::data(a)), m_size(N)
    {}

    constexpr __device__ __host__ bool empty() const { return size() == 0; }
    constexpr __device__ __host__ T* data() const { return m_ptr; }
    constexpr __device__ __host__ size_t size() const { return m_size; }
    constexpr __device__ __host__ size_t size_bytes() const { return m_size * sizeof(T); }
    constexpr __device__ __host__ T& operator[](int i) { return m_ptr[i]; }
    constexpr __device__ __host__ const T& operator[](int i) const { return m_ptr[i]; }
    constexpr __device__ __host__ span<T> subspan(const std::size_t offset, const std::size_t count) const
    {
      if (count == 0) {
        return {m_ptr + offset, m_size - offset};
      }
      else {
        assert(offset + count <= m_size);
        return {m_ptr + offset, count};
      }
    }

    constexpr __device__ __host__ span<T> subspan(const std::size_t offset) const
    {
      return {m_ptr + offset, m_size - offset};
    }

    constexpr __device__ __host__ T* begin() const { return m_ptr; }

    constexpr __device__ __host__ T* end() const { return m_ptr + m_size; }

    constexpr __device__ __host__ T* rbegin() const { return m_ptr + m_size - 1; }

    constexpr __device__ __host__ T* rend() const { return m_ptr - 1; }
  };
#else
  using gsl::span;
#endif
} // namespace Allen::device

using DeviceDimensions = std::array<unsigned, 3>;

// Helper structure to deal with constness of T
template<typename T, typename U>
struct ForwardType {
  using t = U;
};

template<typename T, typename U>
struct ForwardType<const T, U> {
  using t = std::add_const_t<U>;
};

struct HashNotPopulatedException : public std::exception {
private:
  std::string m_algorithm_name;

public:
  HashNotPopulatedException(const std::string& algorithm_name) :
    m_algorithm_name("Pre or post-scaler hash not populated in selection algorithm " + algorithm_name)
  {}

  const char* what() const noexcept override { return m_algorithm_name.c_str(); }
};

__device__ inline float signselect(const float& s, const float& a, const float& b) { return (s > 0) ? a : b; }
