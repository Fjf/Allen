/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <tuple>
#include <array>

// Avoid warnings from the hana library from nvcc and clang
#ifdef __CUDACC__
#pragma push
#pragma diag_suppress = expr_has_no_effect
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdouble-promotion"
#endif

#include <boost/hana/members.hpp>
#include <boost/hana/define_struct.hpp>
#include <boost/hana/ext/std/tuple.hpp>

#ifdef __CUDACC__
#pragma pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

// Datatypes can be host or device.
// Note: These structs need to be not templated.
struct host_datatype {
  virtual void set_size(size_t) {}
  virtual size_t size() const { return 0; }
  virtual std::string name() const { return ""; }
  virtual void set_offset(char*) {}
  virtual char* offset() const { return nullptr; }
  virtual ~host_datatype() {}
};

struct device_datatype {
  virtual void set_size(size_t) {}
  virtual size_t size() const { return 0; }
  virtual std::string name() const { return ""; }
  virtual void set_offset(char*) {}
  virtual char* offset() const { return nullptr; }
  virtual ~device_datatype() {}
};

// A generic datatype* data holder.
template<typename internal_t>
struct datatype {
  using type = internal_t;
  __host__ __device__ datatype(type* value) : m_value(value) {}
  __host__ __device__ datatype() {}

protected:
  type* m_value;
};

// Input datatypes have read-only accessors.
template<typename internal_t>
struct input_datatype : datatype<internal_t> {
  using type = typename datatype<internal_t>::type;
  __host__ __device__ input_datatype() {}
  __host__ __device__ input_datatype(type* value) : datatype<internal_t>(value) {}
  __host__ __device__ operator const type*() const { return const_cast<const type*>(this->m_value); }
  __host__ __device__ const type* get() const { return const_cast<const type*>(this->m_value); }
};

// Output datatypes return pointers that can be modified.
template<typename internal_t>
struct output_datatype : datatype<internal_t> {
  using type = typename datatype<internal_t>::type;
  __host__ __device__ output_datatype() {}
  __host__ __device__ output_datatype(type* value) : datatype<internal_t>(value) {}
  __host__ __device__ operator type*() const { return this->m_value; }
  __host__ __device__ type* get() const { return this->m_value; }
};

// Inputs / outputs have an additional parameter method to be able to parse it with libclang.
#define DEVICE_INPUT(ARGUMENT_NAME, ...)                                       \
  struct ARGUMENT_NAME : public device_datatype, input_datatype<__VA_ARGS__> { \
    using input_datatype<__VA_ARGS__>::input_datatype;                         \
    void inline parameter(__VA_ARGS__) const {}                                \
  }

#define DEVICE_OUTPUT(ARGUMENT_NAME, ...)                                       \
  struct ARGUMENT_NAME : public device_datatype, output_datatype<__VA_ARGS__> { \
    using output_datatype<__VA_ARGS__>::output_datatype;                        \
    void inline parameter(__VA_ARGS__) {}                                       \
  }

#define HOST_INPUT(ARGUMENT_NAME, ...)                                       \
  struct ARGUMENT_NAME : public host_datatype, input_datatype<__VA_ARGS__> { \
    using input_datatype<__VA_ARGS__>::input_datatype;                       \
    void inline parameter(__VA_ARGS__) const {}                              \
  }

#define HOST_OUTPUT(ARGUMENT_NAME, ...)                                       \
  struct ARGUMENT_NAME : public host_datatype, output_datatype<__VA_ARGS__> { \
    using output_datatype<__VA_ARGS__>::output_datatype;                      \
    void inline parameter(__VA_ARGS__) {}                                     \
  }

// Struct that mimics std::array<unsigned, 3> and works with CUDA.
struct DeviceDimensions {
  unsigned x;
  unsigned y;
  unsigned z;

  constexpr DeviceDimensions() : x(1), y(1), z(1) {}
  constexpr DeviceDimensions(const std::array<unsigned, 3>& v) : x(v[0]), y(v[1]), z(v[2]) {}
};

// A property datatype data holder.
template<typename T>
struct property_datatype {
  using t = T;

  __host__ __device__ constexpr property_datatype(const t& value) : m_value(value) {}
  __host__ __device__ constexpr property_datatype() {}
  __host__ __device__ operator t() const { return this->m_value; }
  __host__ __device__ t get() const { return this->m_value; }
  __host__ __device__ operator dim3() const;

protected:
  t m_value;
};

template<>
struct property_datatype<DeviceDimensions> {
  using t = DeviceDimensions;

  __host__ __device__ constexpr property_datatype(const t& value) : m_value(value) {}
  __host__ __device__ constexpr property_datatype() {}
  __host__ __device__ operator t() const { return this->m_value; }
  __host__ __device__ t get() const { return this->m_value; }
  __host__ __device__ operator dim3() const { return {this->m_value.x, this->m_value.y, this->m_value.z}; }

protected:
  t m_value;
};

// Properties have an additional property method to be able to parse it with libclang.
// libclang relies on name and description being 2nd and 3rd arguments of this macro function.
#define PROPERTY(ARGUMENT_NAME, NAME, DESCRIPTION, ...)      \
  struct ARGUMENT_NAME : property_datatype<__VA_ARGS__> {    \
    constexpr static auto name {NAME};                       \
    constexpr static auto description {DESCRIPTION};         \
    using property_datatype<__VA_ARGS__>::property_datatype; \
    void inline property(__VA_ARGS__) {}                     \
  }

/**
 * @brief Dependencies for an algorithm, after
 *        being processed by the scheduler machinery.
 */
template<typename T, typename ArgumentsTuple>
struct ScheduledDependencies {
  using Algorithm = T;
  using Arguments = ArgumentsTuple;
};

#define DEFINE_PARAMETERS(CLASS_NAME, ...)             \
  struct CLASS_NAME {                                  \
    BOOST_HANA_DEFINE_STRUCT(CLASS_NAME, __VA_ARGS__); \
  };
