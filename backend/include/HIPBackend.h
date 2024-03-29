/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <Logger.h>
#include "BackendCommonInterface.h"

#if !defined(__HCC__) && !defined(__HIP__)
#define __HIP_PLATFORM_HCC__

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-result"
#elif __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include <hip/hip_runtime_api.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#else

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#elif __GNUC__ >= 8
#pragma GCC diagnostic pop
#endif

#include <hip/hip_runtime.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <hip/math_functions.h>
#endif

#if !defined(DEVICE_COMPILER)
#include <cmath>
#endif

#include <iomanip>
#include <hip/hip_fp16.h>
#define half_t half

// Intrinsics
constexpr int warp_size = 64;
#define __shfl_down_sync(_, val, offset) __shfl_down(val, offset)
#define __shfl_sync(_, val, offset) __shfl(val, offset)
#define __ballot_sync(_, predicate) __ballot(predicate)

// Support for dynamic shared memory buffers
#define DYNAMIC_SHARED_MEMORY_BUFFER(_type, _instance, _config) HIP_DYNAMIC_SHARED(_type, _instance)

// syncwarp is not supported in HIP. Use syncthreads instead
#define __syncwarp __syncthreads

/**
 * @brief Macro to check hip calls.
 */
#define hipCheck(stmt)                                                                                            \
  {                                                                                                               \
    hipError_t err = stmt;                                                                                        \
    if (err != hipSuccess) {                                                                                      \
      fprintf(                                                                                                    \
        stderr, "Failed to run %s\n%s (%d) at %s: %d\n", #stmt, hipGetErrorString(err), err, __FILE__, __LINE__); \
      throw std::invalid_argument("hipCheck failed");                                                             \
    }                                                                                                             \
  }

#define hipCheckKernelCall(stmt)                                                                                  \
  {                                                                                                               \
    hipError_t err = stmt;                                                                                        \
    if (err != hipSuccess) {                                                                                      \
      fprintf(                                                                                                    \
        stderr, "Failed to invoke kernel\n%s (%d) at %s: %d\n", hipGetErrorString(err), err, __FILE__, __LINE__); \
      throw std::invalid_argument("hipCheckKernelCall failed");                                                   \
    }                                                                                                             \
  }

namespace Allen {
  struct KernelInvocationConfiguration {
    KernelInvocationConfiguration() = default;
    KernelInvocationConfiguration(const dim3&, const dim3&, const unsigned) {}
  };

#ifdef SYNCHRONOUS_DEVICE_EXECUTION
  struct Context {
    void initialize() {}
  };
#else
  struct Context {
  private:
    hipStream_t m_stream;

  public:
    Context() {}

    void initialize() { hipCheck(hipStreamCreate(&m_stream)); }

    hipStream_t inline stream() const { return m_stream; }
  };
#endif

  // Convert kind from Allen::memcpy_kind to hipMemcpyKind
  hipMemcpyKind inline convert_allen_to_hip_kind(Allen::memcpy_kind kind)
  {
    switch (kind) {
    case memcpyHostToHost: return hipMemcpyHostToHost;
    case memcpyHostToDevice: return hipMemcpyHostToDevice;
    case memcpyDeviceToHost: return hipMemcpyDeviceToHost;
    case memcpyDeviceToDevice: return hipMemcpyDeviceToDevice;
    default: return hipMemcpyDefault;
    }
  }

  unsigned inline convert_allen_to_hip_host_register_kind(Allen::host_register_kind kind)
  {
    switch (kind) {
    case hostRegisterPortable: return hipHostRegisterPortable;
    case hostRegisterMapped: return hipHostRegisterMapped;
    case hostRegisterReadOnly:
      warning_cout << "hostRegisterReadOnly is not supported by HIP\n";
      return hipHostRegisterDefault;
    case hostRegisterIoMemory:
      warning_cout << "hostRegisterIoMemory is not supported by HIP\n";
      return hipHostRegisterDefault;
    default: return hipHostRegisterDefault;
    }
  }

  void inline malloc(void** devPtr, size_t size) { hipCheck(hipMalloc(devPtr, size)); }

  void inline malloc_host(void** ptr, size_t size) { hipCheck(hipHostMalloc(ptr, size)); }

  void inline memcpy(void* dst, const void* src, size_t count, Allen::memcpy_kind kind)
  {
    hipCheck(hipMemcpy(dst, src, count, convert_allen_to_hip_kind(kind)));
  }

#ifdef SYNCHRONOUS_DEVICE_EXECUTION
  void inline memcpy_async(void* dst, const void* src, size_t count, Allen::memcpy_kind kind, const Context&)
  {
    memcpy(dst, src, count, kind);
  }
#else
  void inline memcpy_async(void* dst, const void* src, size_t count, Allen::memcpy_kind kind, const Context& context)
  {
    hipCheck(hipMemcpyAsync(dst, src, count, convert_allen_to_hip_kind(kind), context.stream()));
  }
#endif

  void inline memset(void* devPtr, int value, size_t count) { hipCheck(hipMemset(devPtr, value, count)); }

#ifdef SYNCHRONOUS_DEVICE_EXECUTION
  void inline memset_async(void* ptr, int value, size_t count, const Context&) { memset(ptr, value, count); }
#else
  void inline memset_async(void* ptr, int value, size_t count, const Context& context)
  {
    hipCheck(hipMemsetAsync(ptr, value, count, context.stream()));
  }
#endif

  void inline free_host(void* ptr) { hipCheck(hipHostFree(ptr)); }

  void inline free(void* ptr) { hipCheck(hipFree(ptr)); }

#ifdef SYNCHRONOUS_DEVICE_EXECUTION
  void inline synchronize(const Context&) {}
#else
  void inline synchronize(const Context& context) { hipCheck(hipStreamSynchronize(context.stream())); }
#endif

  void inline device_reset() { hipCheck(hipDeviceReset()); }

  void inline peek_at_last_error() { hipCheckKernelCall(hipPeekAtLastError()); }

  void inline host_unregister(void* ptr) { hipCheck(hipHostUnregister(ptr)); }

  void inline host_register(void* ptr, size_t size, host_register_kind flags)
  {
    hipCheck(hipHostRegister(ptr, size, convert_allen_to_hip_host_register_kind(flags)));
  }

  namespace device {
    template<class To, class From>
    __host__ __device__ std::enable_if_t<
      sizeof(To) == sizeof(From) && alignof(To) == alignof(From) && std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>,
      To>
    bit_cast(const From& src) noexcept
    {
      return *reinterpret_cast<const To*>(&src);
    }
  } // namespace device
} // namespace Allen
