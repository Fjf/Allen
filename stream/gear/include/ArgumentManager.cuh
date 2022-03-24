/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <tuple>
#include <gsl/gsl>
#include <vector>
#include <cstring>
#include <unordered_map>
#include "BackendCommon.h"
#include "Logger.h"
#include "AllenTypeTraits.h"
#include "Argument.cuh"
#include "StructToTuple.cuh"

enum class ArgumentScope { Host, Device, Invalid };

/**
 * @brief Contains the data of an argument, namely its name, data pointer and size.
 *
 */
struct ArgumentData {
private:
  std::string m_name = "";
  ArgumentScope m_scope = ArgumentScope::Invalid;
  void* m_pointer = nullptr;
  size_t m_size = 0;
  size_t m_type_size = 0;

public:
  ArgumentData() = default;
  ArgumentData(const ArgumentData&) = default;
  ArgumentData(const std::string& name) : m_name(name) {}
  ArgumentData(const std::string& name, ArgumentScope scope) : m_name(name), m_scope(scope) {}

  virtual void* pointer() const { return m_pointer; }
  virtual size_t size() const { return m_size; }
  virtual size_t sizebytes() const { return m_size * m_type_size; }
  virtual std::string name() const { return m_name; }
  virtual ArgumentScope scope() const { return m_scope; }
  virtual void set_pointer(void* pointer) { m_pointer = pointer; }
  virtual void set_size(size_t size) { m_size = size; }
  virtual void set_type_size(size_t type_size) { m_type_size = type_size; }
  virtual void set_name(const std::string& name) { m_name = name; }
  virtual void set_scope(ArgumentScope scope) { m_scope = scope; }
  virtual ~ArgumentData() {}
};

/**
 * @brief Allen argument manager
 */
class UnorderedStore {
  std::unordered_map<std::string, ArgumentData> m_store;

public:
  ArgumentData& at(const std::string& k) { return m_store.at(k); }

  void emplace(const std::string& k, ArgumentData&& t)
  {
    const auto& [ret, ok] = m_store.try_emplace(k, std::forward<ArgumentData>(t));
    if (!ok) {
      throw std::runtime_error("store emplace failed, entry already exists");
    }
  }
};

/**
 * @brief Manager of argument references for every handler.
 */
template<
  typename ParametersAndPropertiesTuple,
  typename ParameterTuple,
  typename ParameterStruct,
  typename InputAggregatesTuple = std::tuple<>>
struct ArgumentRefManager {
public:
  using parameters_and_properties_tuple_t = ParametersAndPropertiesTuple;
  using parameters_tuple_t = ParameterTuple;
  using parameters_struct_t = ParameterStruct;
  using input_aggregates_t = InputAggregatesTuple;
  using store_ref_t = std::array<std::reference_wrapper<ArgumentData>, std::tuple_size_v<parameters_tuple_t>>;

private:
  mutable store_ref_t m_store_ref;
  input_aggregates_t m_input_aggregates;

public:
  ArgumentRefManager(store_ref_t store_ref, input_aggregates_t input_aggregates) :
    m_store_ref(store_ref), m_input_aggregates(input_aggregates)
  {}

  ArgumentRefManager(store_ref_t store_ref) : m_store_ref(store_ref) {}

  template<typename T, std::enable_if_t<!std::is_base_of_v<aggregate_datatype, T>, bool> = true>
  typename T::type* pointer() const
  {
    constexpr auto index_of_T = index_of_v<T, parameters_tuple_t>;
    static_assert(index_of_T < std::tuple_size_v<parameters_tuple_t> && "Index of T is in bounds");
    auto pointer = m_store_ref[index_of_T].get().pointer();
    return reinterpret_cast<typename T::type*>(pointer);
  }

  template<typename T, std::enable_if_t<!std::is_base_of_v<aggregate_datatype, T>, bool> = true>
  typename T::type first() const
  {
    static_assert(std::is_base_of_v<host_datatype, T> && "first can only access host datatypes");
    if constexpr (std::is_base_of_v<optional_datatype, T>) {
      if (pointer<T>() == nullptr) {
        return 0;
      }
    }
    return pointer<T>()[0];
  }

  template<typename T, std::enable_if_t<!std::is_base_of_v<aggregate_datatype, T>, bool> = true>
  size_t size() const
  {
    constexpr auto index_of_T = index_of_v<T, parameters_tuple_t>;
    static_assert(index_of_T < std::tuple_size_v<parameters_tuple_t> && "Index of T is in bounds");
    return m_store_ref[index_of_T].get().size();
  }

  template<typename T, std::enable_if_t<!std::is_base_of_v<aggregate_datatype, T>, bool> = true>
  void set_size(const size_t size)
  {
    static_assert(!Allen::is_template_base_of_v<input_datatype, T> && "set_size can only be used on output datatypes");
    constexpr auto index_of_T = index_of_v<T, parameters_tuple_t>;
    static_assert(index_of_T < std::tuple_size_v<parameters_tuple_t> && "Index of T is in bounds");
    m_store_ref[index_of_T].get().set_size(size);
    m_store_ref[index_of_T].get().set_type_size(sizeof(typename T::type));
  }

  /**
   * @brief Reduces the size of the container.
   * @details Reducing the size can be done in the operator(), hence this method is const.
   */
  template<typename T, std::enable_if_t<!std::is_base_of_v<aggregate_datatype, T>, bool> = true>
  void reduce_size(const size_t size) const
  {
    static_assert(
      !Allen::is_template_base_of_v<input_datatype, T> && "reduce_size can only be used on output datatypes");
    constexpr auto index_of_T = index_of_v<T, parameters_tuple_t>;
    static_assert(index_of_T < std::tuple_size_v<parameters_tuple_t> && "Index of T is in bounds");
    assert(size <= m_store_ref[index_of_T].get().size());
    m_store_ref[index_of_T].get().set_size(size);
  }

  template<typename T, std::enable_if_t<!std::is_base_of_v<aggregate_datatype, T>, bool> = true>
  std::string name() const
  {
    constexpr auto index_of_T = index_of_v<T, parameters_tuple_t>;
    static_assert(index_of_T < std::tuple_size_v<parameters_tuple_t> && "Index of T is in bounds");
    return m_store_ref[index_of_T].get().name();
  }

  template<typename T, std::enable_if_t<std::is_base_of_v<aggregate_datatype, T>, bool> = true>
  auto input_aggregate() const
  {
    return std::get<T>(m_input_aggregates).value();
  }
};

/**
 * @brief Aggregate datatype
 */
template<typename T>
struct InputAggregate {
private:
  std::vector<std::reference_wrapper<ArgumentData>> m_argument_data_v;

public:
  using type = T;

  InputAggregate() = default;

  InputAggregate(const std::vector<std::reference_wrapper<ArgumentData>>& argument_data_v) :
    m_argument_data_v(argument_data_v)
  {}

  template<typename Tuple, std::size_t... Is>
  InputAggregate(Tuple t, std::index_sequence<Is...>) : m_argument_data_v {std::get<Is>(t)...}
  {}

  T* data(const unsigned index) const
  {
    assert(index < m_argument_data_v.size() && "Index is in bounds");
    auto pointer = m_argument_data_v[index].get().pointer();
    return reinterpret_cast<T*>(pointer);
  }

  T first(const unsigned index) const
  {
    assert(index < m_argument_data_v.size() && "Index is in bounds");
    return data(index)[0];
  }

  size_t size(const unsigned index) const
  {
    assert(index < m_argument_data_v.size() && "Index is in bounds");
    return m_argument_data_v[index].get().size();
  }

  gsl::span<T> span(const unsigned index) const { return {data(index), size(index)}; }

  std::string name(const unsigned index) const
  {
    assert(index < m_argument_data_v.size() && "Index is in bounds");
    return m_argument_data_v[index].get().name();
  }

  size_t size_of_aggregate() const { return m_argument_data_v.size(); }
};

template<typename T, typename... Ts>
static auto makeInputAggregate(std::tuple<Ts&...> tp)
{
  return InputAggregate<T> {tp, std::make_index_sequence<sizeof...(Ts)>()};
}

// Macro
#define INPUT_AGGREGATE(HOST_DEVICE, ARGUMENT_NAME, ...)                                             \
  struct ARGUMENT_NAME : public aggregate_datatype, HOST_DEVICE {                                    \
    using type = InputAggregate<__VA_ARGS__>;                                                        \
    void parameter(__VA_ARGS__) const;                                                               \
    ARGUMENT_NAME() = default;                                                                       \
    ARGUMENT_NAME(const type& input_aggregate) : m_value(input_aggregate) {}                         \
    template<typename... Ts>                                                                         \
    ARGUMENT_NAME(std::tuple<Ts&...> value) : m_value(makeInputAggregate<__VA_ARGS__, Ts...>(value)) \
    {}                                                                                               \
    const type& value() const { return m_value; }                                                    \
                                                                                                     \
  private:                                                                                           \
    type m_value {};                                                                                 \
  }

#define HOST_INPUT_AGGREGATE(ARGUMENT_NAME, ...) INPUT_AGGREGATE(host_datatype, ARGUMENT_NAME, __VA_ARGS__)

#define DEVICE_INPUT_AGGREGATE(ARGUMENT_NAME, ...) INPUT_AGGREGATE(device_datatype, ARGUMENT_NAME, __VA_ARGS__)

/**
 * @brief Tuple wrapper that extracts tuples out of the
 *        Parameters struct. It extracts a tuple of all parameters and properties
 *        (parameters_and_properties_tuple_t), and a tuple of parameters (parameters_tuple_t).
 */
template<typename Tuple, typename Enabled = void>
struct WrappedTuple;

template<>
struct WrappedTuple<std::tuple<>, void> {
  using parameters_and_properties_tuple_t = std::tuple<>;
  using parameters_tuple_t = std::tuple<>;
  using aggregates_tuple_t = std::tuple<>;
};

template<typename T, typename... R>
struct WrappedTuple<
  std::tuple<T, R...>,
  std::enable_if_t<(std::is_base_of_v<device_datatype, T> || std::is_base_of_v<host_datatype, T>) &&!std::
                     is_base_of_v<aggregate_datatype, T>>> {
  using prev_wrapped_tuple = WrappedTuple<std::tuple<R...>>;
  using prev_parameters_and_properties_tuple_t = typename prev_wrapped_tuple::parameters_and_properties_tuple_t;
  using parameters_and_properties_tuple_t = prepend_to_tuple_t<T, prev_parameters_and_properties_tuple_t>;
  using prev_parameters_tuple_t = typename prev_wrapped_tuple::parameters_tuple_t;
  using parameters_tuple_t = prepend_to_tuple_t<T, prev_parameters_tuple_t>;
  using aggregates_tuple_t = typename prev_wrapped_tuple::aggregates_tuple_t;
};

template<typename T, typename... R>
struct WrappedTuple<std::tuple<T, R...>, std::enable_if_t<std::is_base_of_v<aggregate_datatype, T>>> {
  using prev_wrapped_tuple = WrappedTuple<std::tuple<R...>>;
  using parameters_and_properties_tuple_t =
    prepend_to_tuple_t<T, typename prev_wrapped_tuple::parameters_and_properties_tuple_t>;
  using parameters_tuple_t = typename prev_wrapped_tuple::parameters_tuple_t;
  using aggregates_tuple_t = prepend_to_tuple_t<T, typename prev_wrapped_tuple::aggregates_tuple_t>;
};

template<typename T, typename... R>
struct WrappedTuple<
  std::tuple<T, R...>,
  std::enable_if_t<
    !std::is_base_of_v<device_datatype, T> && !std::is_base_of_v<host_datatype, T> &&
    !std::is_base_of_v<aggregate_datatype, T>>> {
  using prev_wrapped_tuple = WrappedTuple<std::tuple<R...>>;
  using prev_parameters_and_properties_tuple_t = typename prev_wrapped_tuple::parameters_and_properties_tuple_t;
  using parameters_and_properties_tuple_t = prepend_to_tuple_t<T, prev_parameters_and_properties_tuple_t>;
  using parameters_tuple_t = typename prev_wrapped_tuple::parameters_tuple_t;
  using aggregates_tuple_t = typename prev_wrapped_tuple::aggregates_tuple_t;
};

template<typename T>
using ArgumentReferences = ArgumentRefManager<
  typename WrappedTuple<decltype(struct_to_tuple(T {}))>::parameters_and_properties_tuple_t,
  typename WrappedTuple<decltype(struct_to_tuple(T {}))>::parameters_tuple_t,
  T,
  typename WrappedTuple<decltype(struct_to_tuple(T {}))>::aggregates_tuple_t>;

namespace Allen {
  template<size_t... Is>
  auto gen_input_aggregates_tuple(
    const std::vector<std::vector<std::reference_wrapper<ArgumentData>>>& input_aggregates,
    std::index_sequence<Is...>)
  {
    return std::make_tuple(input_aggregates[Is]...);
  }
} // namespace Allen
