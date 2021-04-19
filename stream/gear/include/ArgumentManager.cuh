/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <tuple>
#include <vector>
#include <cstring>
#include "BackendCommon.h"
#include "Logger.h"
#include "AllenTypeTraits.cuh"
#include "Argument.cuh"
#include "ArgumentOps.cuh"
#include "StructToTuple.cuh"

/**
 * @brief Contains the data of an argument, namely its name, data pointer and size.
 *
 */
struct ArgumentData {
private:
  char* m_pointer = nullptr;
  size_t m_size = 0;
  std::string m_name = "";

public:
  virtual char* pointer() const { return m_pointer; }
  virtual size_t size() const { return m_size; }
  virtual std::string name() const { return m_name; }
  virtual void set_pointer(char* pointer) { m_pointer = pointer; }
  virtual void set_size(size_t size) { m_size = size; }
  virtual void set_name(const std::string& name) { m_name = name; }
  virtual ~ArgumentData() {}
};

/**
 * @brief Holds a database of arguments, and provides accessors for their pointers and size.
 */
template<typename Tuple>
struct ArgumentManager {
private:
  std::array<ArgumentData, std::tuple_size_v<Tuple>> m_tuple_to_argument_data;

public:
  std::array<ArgumentData, std::tuple_size_v<Tuple>>& argument_database() { return m_tuple_to_argument_data; }

  template<typename T>
  typename T::type* pointer() const
  {
    constexpr auto index_of_T = index_of_v<T, Tuple>;
    static_assert(index_of_T < std::tuple_size_v<Tuple> && "Index of T is in bounds");
    auto pointer = m_tuple_to_argument_data[index_of_T].pointer();
    return reinterpret_cast<typename T::type*>(pointer);
  }

  template<typename T>
  void set_pointer(char* pointer)
  {
    constexpr auto index_of_T = index_of_v<T, Tuple>;
    static_assert(index_of_T < std::tuple_size_v<Tuple> && "Index of T is in bounds");
    m_tuple_to_argument_data[index_of_T].set_pointer(pointer);
  }

  template<typename T>
  size_t size() const
  {
    constexpr auto index_of_T = index_of_v<T, Tuple>;
    static_assert(index_of_T < std::tuple_size_v<Tuple> && "Index of T is in bounds");
    return m_tuple_to_argument_data[index_of_T].size();
  }

  template<typename T>
  void set_size(const size_t size)
  {
    constexpr auto index_of_T = index_of_v<T, Tuple>;
    static_assert(index_of_T < std::tuple_size_v<Tuple> && "Index of T is in bounds");
    m_tuple_to_argument_data[index_of_T].set_size(size);
  }

  template<typename T>
  std::string name() const
  {
    constexpr auto index_of_T = index_of_v<T, Tuple>;
    static_assert(index_of_T < std::tuple_size_v<Tuple> && "Index of T is in bounds");
    return m_tuple_to_argument_data[index_of_T].name();
  }

  template<typename T>
  void set_name(const std::string& name)
  {
    constexpr auto index_of_T = index_of_v<T, Tuple>;
    static_assert(index_of_T < std::tuple_size_v<Tuple> && "Index of T is in bounds");
    return m_tuple_to_argument_data[index_of_T].set_name(name);
  }
};

/**
 * @brief Manager of argument references for every handler.
 */
template<typename ParametersAndPropertiesTuple, typename ParameterTuple = void, typename ParameterStruct = void>
struct ArgumentRefManager {
public:
  using parameters_and_properties_tuple_t = ParametersAndPropertiesTuple;
  using parameters_tuple_t = ParameterTuple;
  using parameters_struct_t = ParameterStruct;

private:
  mutable std::array<std::reference_wrapper<ArgumentData>, std::tuple_size_v<parameters_tuple_t>>
    m_tuple_to_argument_data;

public:
  ArgumentRefManager(
    std::array<std::reference_wrapper<ArgumentData>, std::tuple_size_v<parameters_tuple_t>> tuple_to_argument_data) :
    m_tuple_to_argument_data(tuple_to_argument_data)
  {}

  template<typename T>
  typename T::type* pointer() const
  {
    constexpr auto index_of_T = index_of_v<T, parameters_tuple_t>;
    static_assert(index_of_T < std::tuple_size_v<parameters_tuple_t> && "Index of T is in bounds");
    auto pointer = m_tuple_to_argument_data[index_of_T].get().pointer();
    return reinterpret_cast<typename T::type*>(pointer);
  }

  template<typename T>
  typename T::type first() const
  {
    return pointer<T>()[0];
  }

  template<typename T>
  size_t size() const
  {
    constexpr auto index_of_T = index_of_v<T, parameters_tuple_t>;
    static_assert(index_of_T < std::tuple_size_v<parameters_tuple_t> && "Index of T is in bounds");
    return m_tuple_to_argument_data[index_of_T].get().size();
  }

  template<typename T>
  void set_size(const size_t size)
  {
    constexpr auto index_of_T = index_of_v<T, parameters_tuple_t>;
    static_assert(index_of_T < std::tuple_size_v<parameters_tuple_t> && "Index of T is in bounds");
    m_tuple_to_argument_data[index_of_T].get().set_size(size);
  }

  /**
   * @brief Reduces the size of the container.
   * @details Reducing the size can be done in the operator(), hence this method is const.
   */
  template<typename T>
  void reduce_size(const size_t size) const
  {
    constexpr auto index_of_T = index_of_v<T, parameters_tuple_t>;
    static_assert(index_of_T < std::tuple_size_v<parameters_tuple_t> && "Index of T is in bounds");
    assert(size <= m_tuple_to_argument_data[index_of_T].get().size());
    m_tuple_to_argument_data[index_of_T].get().set_size(size);
  }

  template<typename T>
  std::string name() const
  {
    constexpr auto index_of_T = index_of_v<T, parameters_tuple_t>;
    static_assert(index_of_T < std::tuple_size_v<parameters_tuple_t> && "Index of T is in bounds");
    return m_tuple_to_argument_data[index_of_T].get().name();
  }
};

/**
 * @brief Aggregate datatype
 */
template<typename T>
struct InputAggregate {
private:
  std::vector<ArgumentData> m_argument_data_v;

public:
  InputAggregate(
    const std::vector<ArgumentData>& argument_data_v) :
    m_argument_data_v(argument_data_v)
  {}

  template<typename Tuple, std::size_t... Is>
  InputAggregate(Tuple t, std::index_sequence<Is...>) : 
    m_argument_data_v{std::get<Is>(t)...}
  {}

  T* pointer(const int index) const
  {
    assert(index < m_argument_data_v.size() && "Index is in bounds");
    auto pointer = m_argument_data_v[index].pointer();
    return reinterpret_cast<T*>(pointer);
  }

  T first(const int index) const
  {
    assert(index < m_argument_data_v.size() && "Index is in bounds");
    return m_argument_data_v[index][0];
  }

  size_t size(const int index) const
  {
    assert(index < m_argument_data_v.size() && "Index is in bounds");
    return m_argument_data_v[index].size();
  }

  std::string name(const int index) const
  {
    assert(index < m_argument_data_v.size() && "Index is in bounds");
    return m_argument_data_v[index].name();
  }
};

template<typename... Ts>
static auto makeInputAggregate(std::tuple<Ts...> tuple) {
    return InputAggregate{tuple, std::make_index_sequence<sizeof...(Ts)>()};
}

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
};

template<typename T, typename... R>
struct WrappedTuple<
  std::tuple<T, R...>,
  std::enable_if_t<(std::is_base_of_v<device_datatype, T> || std::is_base_of_v<host_datatype, T>) &&!std::
                     is_base_of_v<aggregate_datatype, T>>> {
  using prev_parameters_and_properties_tuple_t =
    typename WrappedTuple<std::tuple<R...>>::parameters_and_properties_tuple_t;
  using parameters_and_properties_tuple_t = prepend_to_tuple_t<T, prev_parameters_and_properties_tuple_t>;
  using prev_parameters_tuple_t = typename WrappedTuple<std::tuple<R...>>::parameters_tuple_t;
  using parameters_tuple_t = prepend_to_tuple_t<T, prev_parameters_tuple_t>;
};

template<typename T, typename... R>
struct WrappedTuple<std::tuple<T, R...>, std::enable_if_t<std::is_base_of_v<aggregate_datatype, T>>> {
  using prev_parameters_and_properties_tuple_t =
    typename WrappedTuple<std::tuple<R...>>::parameters_and_properties_tuple_t;
  using parameters_and_properties_tuple_t = cat_tuples_t<typename T::type, prev_parameters_and_properties_tuple_t>;
  using prev_parameters_tuple_t = typename WrappedTuple<std::tuple<R...>>::parameters_tuple_t;
  using parameters_tuple_t = cat_tuples_t<typename T::type, prev_parameters_tuple_t>;
};

template<typename T, typename... R>
struct WrappedTuple<
  std::tuple<T, R...>,
  std::enable_if_t<!std::is_base_of_v<device_datatype, T> && !std::is_base_of_v<host_datatype, T>>> {
  using prev_parameters_and_properties_tuple_t =
    typename WrappedTuple<std::tuple<R...>>::parameters_and_properties_tuple_t;
  using parameters_and_properties_tuple_t = prepend_to_tuple_t<T, prev_parameters_and_properties_tuple_t>;
  using parameters_tuple_t = typename WrappedTuple<std::tuple<R...>>::parameters_tuple_t;
};

template<typename T>
using ArgumentReferences = ArgumentRefManager<
  typename WrappedTuple<decltype(struct_to_tuple(T {}))>::parameters_and_properties_tuple_t,
  typename WrappedTuple<decltype(struct_to_tuple(T {}))>::parameters_tuple_t,
  T>;
