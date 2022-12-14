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

namespace Allen::Store {

  /**
   * @brief Aggregate datatype
   */
  template<typename T>
  struct InputAggregate {
  private:
    std::vector<std::reference_wrapper<BaseArgument>> m_argument_data_v;

  public:
    using type = T;

    InputAggregate() = default;

    InputAggregate(const std::vector<std::reference_wrapper<BaseArgument>>& argument_data_v) :
      m_argument_data_v(argument_data_v)
    {}

    template<typename Tuple, std::size_t... Is>
    InputAggregate(Tuple t, std::index_sequence<Is...>) : m_argument_data_v {std::get<Is>(t)...}
    {}

    gsl::span<T> get(const unsigned index) const
    {
      assert(index < m_argument_data_v.size() && "Index is in bounds");
      return m_argument_data_v[index].get();
    }

    auto data(const unsigned index) const { return get(index).data(); }

    auto first(const unsigned index) const { return get(index)[0]; }

    auto size(const unsigned index) const { return get(index).size(); }

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
#define INPUT_AGGREGATE(HOST_OR_DEVICE, ARGUMENT_NAME, ...)                        \
  struct ARGUMENT_NAME : public Allen::Store::aggregate_datatype, HOST_OR_DEVICE { \
    using type = Allen::Store::InputAggregate<__VA_ARGS__>;                        \
    void parameter(__VA_ARGS__) const;                                             \
  }

#define HOST_INPUT_AGGREGATE(ARGUMENT_NAME, ...) \
  INPUT_AGGREGATE(Allen::Store::host_datatype, ARGUMENT_NAME, __VA_ARGS__)

#define DEVICE_INPUT_AGGREGATE(ARGUMENT_NAME, ...) \
  INPUT_AGGREGATE(Allen::Store::device_datatype, ARGUMENT_NAME, __VA_ARGS__)

} // namespace Allen::Store
