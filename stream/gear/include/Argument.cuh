#pragma once

#include <tuple>

// Datatype
template<typename internal_t>
struct input_datatype {
  using type = internal_t;

  operator const type*() const { return const_cast<const type*>(m_value); }
  
  const type* get() const { return const_cast<const type*>(m_value); }

  input_datatype(type* value) : m_value(value) {}

  input_datatype() = default;

  input_datatype(const input_datatype&) = default;

private:
  type* m_value;
};

template<typename internal_t>
struct output_datatype {
  using type = internal_t;

  operator type*() const { return m_value; }

  type* get() const { return m_value; }

  output_datatype(type* value) : m_value(value) {}

  output_datatype() = default;

  output_datatype(const output_datatype&) = default;

private:
  type* m_value;
};

#define ARG(ARGUMENT_NAME, ...)         \
  struct ARGUMENT_NAME : __VA_ARGS__ {           \
    constexpr static auto name {#ARGUMENT_NAME}; \
    size_t size;                                 \
    char* offset;                                \
  };

/**
 * @brief Macro for defining arguments in a Handler.
 */
#define ARGUMENTS(...) std::tuple<__VA_ARGS__>

/**
 * @brief Macro for defining arguments. An argument has an identifier
 *        and a type.
 */
#define ARGUMENT(ARGUMENT_NAME, ARGUMENT_TYPE)   \
  struct ARGUMENT_NAME {                         \
    constexpr static auto name {#ARGUMENT_NAME}; \
    using type = ARGUMENT_TYPE;                  \
    size_t size;                                 \
    char* offset;                                \
  };

/**
 * @brief Defines dependencies for an algorithm.
 *
 * @tparam T The algorithm type.
 * @tparam Args The dependencies.
 */
template<typename T, typename ArgumentsTuple>
struct AlgorithmDependencies {
  using Algorithm = T;
  using Arguments = ArgumentsTuple;
};

/**
 * @brief Dependencies for an algorithm, after
 *        being processed by the scheduler machinery.
 */
template<typename T, typename ArgumentsTuple>
struct ScheduledDependencies {
  using Algorithm = T;
  using Arguments = ArgumentsTuple;
};
