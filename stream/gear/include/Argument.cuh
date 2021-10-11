/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <array>

// Struct to hold the types of the dependencies (libClang)
namespace {
  template<typename... T>
  struct dependencies {
  };
} // namespace

// Datatypes can be host, device or aggregates.
// Note: These structs need to be not templated (libClang).
struct host_datatype {
};
struct device_datatype {
};
struct aggregate_datatype {
};
struct optional_datatype {
};

// Checks all Ts inherit from host_datatype
template<typename... Ts>
constexpr bool all_host_v = (std::is_base_of_v<host_datatype, Ts> && ...);
// Checks all Ts inherit from device_datatype
template<typename... Ts>
constexpr bool all_device_v = (std::is_base_of_v<device_datatype, Ts> && ...);
// Checks all Ts either inherit from host_datatype or all inherit from device_datatype
template<typename... Ts>
constexpr bool all_host_or_all_device_v = all_host_v<Ts...> || all_device_v<Ts...>;

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
template<typename T>
struct input_datatype : datatype<T> {
  using type = typename datatype<T>::type;
  __host__ __device__ input_datatype() {}
  __host__ __device__ input_datatype(type* value) : datatype<T>(value) {}
  __host__ __device__ operator const type*() const { return const_cast<const type*>(this->m_value); }
  __host__ __device__ const type* get() const { return const_cast<const type*>(this->m_value); }
};

// Output datatypes return pointers that can be modified.
template<typename T>
struct output_datatype : datatype<T> {
  using type = typename datatype<T>::type;
  __host__ __device__ output_datatype() {}
  __host__ __device__ output_datatype(type* value) : datatype<T>(value) {}
  __host__ __device__ operator type*() const { return this->m_value; }
  __host__ __device__ type* get() const { return this->m_value; }
};

// Inputs / outputs have an additional parsable method required for libclang parsing.
#define DEVICE_INPUT(ARGUMENT_NAME, ...)                                       \
  struct ARGUMENT_NAME : public device_datatype, input_datatype<__VA_ARGS__> { \
    using input_datatype<__VA_ARGS__>::input_datatype;                         \
    void parameter(__VA_ARGS__) const;                                         \
  }

#define HOST_INPUT(ARGUMENT_NAME, ...)                                       \
  struct ARGUMENT_NAME : public host_datatype, input_datatype<__VA_ARGS__> { \
    using input_datatype<__VA_ARGS__>::input_datatype;                       \
    void parameter(__VA_ARGS__) const;                                       \
  }

#define DEVICE_OUTPUT(ARGUMENT_NAME, ...)                                       \
  struct ARGUMENT_NAME : public device_datatype, output_datatype<__VA_ARGS__> { \
    using output_datatype<__VA_ARGS__>::output_datatype;                        \
    void parameter(__VA_ARGS__);                                                \
  }

#define HOST_OUTPUT(ARGUMENT_NAME, ...)                                       \
  struct ARGUMENT_NAME : public host_datatype, output_datatype<__VA_ARGS__> { \
    using output_datatype<__VA_ARGS__>::output_datatype;                      \
    void parameter(__VA_ARGS__);                                              \
  }

#define DEVICE_OUTPUT_WITH_DEPENDENCIES(ARGUMENT_NAME, DEPS, ...)               \
  struct ARGUMENT_NAME : public device_datatype, output_datatype<__VA_ARGS__> { \
    using output_datatype<__VA_ARGS__>::output_datatype;                        \
    DEPS parameter(__VA_ARGS__);                                                \
  }

#define HOST_OUTPUT_WITH_DEPENDENCIES(ARGUMENT_NAME, DEPS, ...)               \
  struct ARGUMENT_NAME : public host_datatype, output_datatype<__VA_ARGS__> { \
    using output_datatype<__VA_ARGS__>::output_datatype;                      \
    DEPS parameter(__VA_ARGS__);                                              \
  }

// Support for masks
// Masks are unsigned inputs / outputs, which the parser and multi ev scheduler
// deal with in a special way. A maximum of one input mask and one output mask per algorithm
// is allowed.
struct mask_t {
  unsigned m_data = 0;

  __host__ __device__ operator unsigned() const { return m_data; }
};

#define MASK_INPUT(ARGUMENT_NAME)                                         \
  struct ARGUMENT_NAME : public device_datatype, input_datatype<mask_t> { \
    using input_datatype<mask_t>::input_datatype;                         \
    void parameter(mask_t) const;                                         \
  }

#define MASK_OUTPUT(ARGUMENT_NAME)                                         \
  struct ARGUMENT_NAME : public device_datatype, output_datatype<mask_t> { \
    using output_datatype<mask_t>::output_datatype;                        \
    void parameter(mask_t);                                                \
  }

// Support for optional input aggregates
#define DEVICE_INPUT_OPTIONAL(ARGUMENT_NAME, ...)                                                 \
  struct ARGUMENT_NAME : public device_datatype, optional_datatype, input_datatype<__VA_ARGS__> { \
    using input_datatype<__VA_ARGS__>::input_datatype;                                            \
    void parameter(__VA_ARGS__) const;                                                            \
  }

#define HOST_INPUT_OPTIONAL(ARGUMENT_NAME, ...)                                                 \
  struct ARGUMENT_NAME : public host_datatype, optional_datatype, input_datatype<__VA_ARGS__> { \
    using input_datatype<__VA_ARGS__>::input_datatype;                                          \
    void parameter(__VA_ARGS__) const;                                                          \
  }

#define DEPENDENCIES(...) dependencies<__VA_ARGS__>

using DeviceDimensions = std::array<unsigned, 3>;

/**
 * @brief A property datatype data holder.
 */
template<typename T>
struct property_datatype {
  using t = T;

  constexpr property_datatype(const t& value) : m_value(value) {}
  constexpr property_datatype() {}
  __host__ __device__ operator t() const { return this->m_value; }
  __host__ __device__ t get() const { return this->m_value; }
  __host__ __device__ operator dim3() const;

protected:
  t m_value;
};

/**
 * @brief std::string properties cannot be accessed on the device,
 *        as any retrieval of the value results in a copy constructor
 *        invocation, which is not header only.
 */
template<>
struct property_datatype<std::string> {
  using t = std::string;

  // Constructors cannot be constexpr for std::string
  property_datatype(const t& value) : m_value(value) {}
  property_datatype() {}
  operator t() const { return this->m_value; }
  t get() const { return this->m_value; }

protected:
  t m_value;
};

/**
 * @brief DeviceDimension specialization.
 *
 *        A separate specialization is provided for DeviceDimensions to support
 *        conversion to dim3.
 */
template<>
struct property_datatype<DeviceDimensions> {
  using t = DeviceDimensions;

  property_datatype(const t& value) : m_value(value) {}
  property_datatype() {}
  __host__ __device__ operator t() const { return this->m_value; }
  __host__ __device__ t get() const { return this->m_value; }
  __host__ __device__ operator dim3() const { return {this->m_value[0], this->m_value[1], this->m_value[2]}; }

protected:
  t m_value = {0, 0, 0};
};

// Properties have an additional property method to be able to parse it with libclang.
// libclang relies on name and description being 2nd and 3rd arguments of this macro function.
#define PROPERTY(ARGUMENT_NAME, NAME, DESCRIPTION, ...)      \
  struct ARGUMENT_NAME : property_datatype<__VA_ARGS__> {    \
    constexpr static auto name {NAME};                       \
    constexpr static auto description {DESCRIPTION};         \
    using property_datatype<__VA_ARGS__>::property_datatype; \
    void property(__VA_ARGS__) {}                            \
  }
