/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "BackendCommon.h"
#include "Logger.h"
#include "BaseTypes.cuh"
#include "TargetFunction.cuh"
#include "Argument.cuh"
#include "Contract.h"
#include "RuntimeOptions.h"
#include "Constants.cuh"
#include "HostBuffers.cuh"
#include <any>

namespace {
  // Get the ArgumentRefManagerType from the function operator()
  template<typename Function>
  struct FunctionTraits;

  template<typename Function, typename... Ts, typename... OtherArguments>
  struct FunctionTraits<void (Function::*)(const ArgumentRefManager<Ts...>&, OtherArguments...) const> {
    using ArgumentRefManagerType = ArgumentRefManager<Ts...>;
  };

  template<typename Algorithm>
  struct AlgorithmTraits {
    using ArgumentRefManagerType = typename FunctionTraits<decltype(&Algorithm::operator())>::ArgumentRefManagerType;
  };

  // Creates a std::array store out of the vector one
  template<std::size_t... Is>
  auto create_store_ref(
    const std::vector<std::reference_wrapper<ArgumentData>>& vector_store_ref,
    std::index_sequence<Is...>)
  {
    return std::array {vector_store_ref[Is]...};
  }
} // namespace

namespace Allen {
  template<typename ContractsTuple, typename Enabled = void>
  struct AlgorithmContracts;

  template<>
  struct AlgorithmContracts<std::tuple<>, void> {
    using preconditions = std::tuple<>;
    using postconditions = std::tuple<>;
  };

  template<typename A, typename... T>
  struct AlgorithmContracts<
    std::tuple<A, T...>,
    std::enable_if_t<std::is_base_of_v<Allen::contract::Precondition, A>>> {
    using recursive_contracts = AlgorithmContracts<std::tuple<T...>>;
    using preconditions = append_to_tuple_t<typename recursive_contracts::preconditions, A>;
    using postconditions = typename recursive_contracts::postconditions;
  };

  template<typename A, typename... T>
  struct AlgorithmContracts<
    std::tuple<A, T...>,
    std::enable_if_t<std::is_base_of_v<Allen::contract::Postcondition, A>>> {
    using recursive_contracts = AlgorithmContracts<std::tuple<T...>>;
    using preconditions = typename recursive_contracts::preconditions;
    using postconditions = append_to_tuple_t<typename recursive_contracts::postconditions, A>;
  };

  // Type-erased algorithm
  class TypeErasedAlgorithm {

    struct vtable {
      std::string (*name)(void const*) = nullptr;
      std::any (*create_arg_ref_manager)(
        std::vector<std::reference_wrapper<ArgumentData>>,
        std::vector<std::vector<std::reference_wrapper<ArgumentData>>>) = nullptr;
      void (*set_arguments_size)(void*, std::any&, const RuntimeOptions&, const Constants&, const HostBuffers&) =
        nullptr;
      void (
        *invoke)(void const*, std::any&, const RuntimeOptions&, const Constants&, HostBuffers&, const Allen::Context&) =
        nullptr;
      void (*init)(void*) = nullptr;
      void (*set_properties)(void*, const std::map<std::string, std::string>&) = nullptr;
      std::map<std::string, std::string> (*get_properties)(void const*) = nullptr;
      std::string (*scope)() = nullptr;
      void (*dtor)(void*) = nullptr;
      void (*run_preconditions)(
        void* p,
        std::any& arg_ref_manager,
        const RuntimeOptions& runtime_options,
        const Constants& constants,
        const Allen::Context& context) = nullptr;
      void (*run_postconditions)(
        void* p,
        std::any& arg_ref_manager,
        const RuntimeOptions& runtime_options,
        const Constants& constants,
        const Allen::Context& context) = nullptr;
    };

    void* instance = nullptr;
    vtable table = {};

  public:
    template<typename ALGORITHM>
    TypeErasedAlgorithm(std::in_place_type_t<ALGORITHM>, const std::string& name)
    {
      auto p = new ALGORITHM {};
      p->set_name(name);
      instance = p;
      table = vtable {
        [](void const* p) { return static_cast<ALGORITHM const*>(p)->name(); },
        [](
          std::vector<std::reference_wrapper<ArgumentData>> vector_store_ref,
          std::vector<std::vector<std::reference_wrapper<ArgumentData>>> input_aggregates) {
          using arg_ref_mgr_t = typename AlgorithmTraits<ALGORITHM>::ArgumentRefManagerType;
          using store_ref_t = typename arg_ref_mgr_t::store_ref_t;
          using input_aggregates_t = typename arg_ref_mgr_t::input_aggregates_t;
          if (std::tuple_size_v<store_ref_t> != vector_store_ref.size()) {
            throw std::runtime_error("unexpected number of arguments");
          }
          auto store_ref =
            create_store_ref(vector_store_ref, std::make_index_sequence<std::tuple_size_v<store_ref_t>> {});
          auto input_agg_store = input_aggregates_t {Allen::gen_input_aggregates_tuple(
            input_aggregates, std::make_index_sequence<std::tuple_size_v<input_aggregates_t>> {})};
          return std::any {arg_ref_mgr_t {store_ref, input_agg_store}};
        },
        [](
          void* p,
          std::any& arg_ref_manager,
          const RuntimeOptions& runtime_options,
          const Constants& constants,
          const HostBuffers& host_buffers) {
          using arg_ref_mgr_t = typename AlgorithmTraits<ALGORITHM>::ArgumentRefManagerType;
          static_cast<ALGORITHM*>(p)->set_arguments_size(
            std::any_cast<arg_ref_mgr_t&>(arg_ref_manager), runtime_options, constants, host_buffers);
        },
        [](
          const void* p,
          std::any& arg_ref_manager,
          const RuntimeOptions& runtime_options,
          const Constants& constants,
          HostBuffers& host_buffers,
          const Allen::Context& context) {
          using arg_ref_mgr_t = typename AlgorithmTraits<ALGORITHM>::ArgumentRefManagerType;
          static_cast<ALGORITHM const*>(p)->operator()(
            std::any_cast<arg_ref_mgr_t&>(arg_ref_manager), runtime_options, constants, host_buffers, context);
        },
        [](void* p) {
          if constexpr (Allen::has_init_member_fn<ALGORITHM>::value) {
            initialize_algorithm(*static_cast<ALGORITHM*>(p));
          }
          else {
            _unused(p);
          }
        },
        [](void* p, const std::map<std::string, std::string>& algo_config) {
          static_cast<ALGORITHM*>(p)->set_properties(algo_config);
        },
        [](void const* p) { return static_cast<ALGORITHM const*>(p)->get_properties(); },
        []() -> std::string { return ALGORITHM::algorithm_scope; },
        [](void* p) { delete static_cast<ALGORITHM*>(p); },
        [](
          void* p,
          std::any& arg_ref_manager,
          const RuntimeOptions& runtime_options,
          const Constants& constants,
          const Allen::Context& context) {
          using arg_ref_mgr_t = typename AlgorithmTraits<ALGORITHM>::ArgumentRefManagerType;
          using preconditions_t = typename AlgorithmContracts<typename ALGORITHM::contracts>::preconditions;
          if constexpr (std::tuple_size_v<preconditions_t>> 0) {
            auto preconditions = preconditions_t {};
            const auto location = static_cast<ALGORITHM const*>(p)->name();
            std::apply(
              [&](auto&... contract) { (contract.set_location(location, demangle<decltype(contract)>()), ...); },
              preconditions);
            std::apply(
              [&](const auto&... contract) {
                (std::invoke(
                   contract, std::any_cast<arg_ref_mgr_t&>(arg_ref_manager), runtime_options, constants, context),
                 ...);
              },
              preconditions);
          }
        },
        [](
          void* p,
          std::any& arg_ref_manager,
          const RuntimeOptions& runtime_options,
          const Constants& constants,
          const Allen::Context& context) {
          using arg_ref_mgr_t = typename AlgorithmTraits<ALGORITHM>::ArgumentRefManagerType;
          using postconditions_t = typename AlgorithmContracts<typename ALGORITHM::contracts>::postconditions;
          if constexpr (std::tuple_size_v<postconditions_t>> 0) {
            auto postconditions = postconditions_t {};
            const auto location = static_cast<ALGORITHM const*>(p)->name();
            std::apply(
              [&](auto&... contract) { (contract.set_location(location, demangle<decltype(contract)>()), ...); },
              postconditions);
            std::apply(
              [&](const auto&... contract) {
                (std::invoke(
                   contract, std::any_cast<arg_ref_mgr_t&>(arg_ref_manager), runtime_options, constants, context),
                 ...);
              },
              postconditions);
          }
        }};
    }
    ~TypeErasedAlgorithm() { (table.dtor)(instance); }
    TypeErasedAlgorithm(const TypeErasedAlgorithm&) = delete;
    TypeErasedAlgorithm(TypeErasedAlgorithm&& arg) : instance {std::exchange(arg.instance, nullptr)}, table {arg.table}
    {}
    TypeErasedAlgorithm& operator=(const TypeErasedAlgorithm&) = delete;
    TypeErasedAlgorithm& operator=(TypeErasedAlgorithm&&) = delete;

    std::string name() const { return (table.name)(instance); }
    std::any create_arg_ref_manager(
      std::vector<std::reference_wrapper<ArgumentData>> vector_store_ref,
      std::vector<std::vector<std::reference_wrapper<ArgumentData>>> input_aggregates)
    {
      return (table.create_arg_ref_manager)(std::move(vector_store_ref), std::move(input_aggregates));
    }
    void set_arguments_size(
      std::any& arg_ref_manager,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const HostBuffers& host_buffers)
    {
      (table.set_arguments_size)(instance, arg_ref_manager, runtime_options, constants, host_buffers);
    }
    void invoke(
      std::any& arg_ref_manager,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      const Allen::Context& context)
    {
      (table.invoke)(instance, arg_ref_manager, runtime_options, constants, host_buffers, context);
    }
    void init() { (table.init)(instance); }
    void set_properties(const std::map<std::string, std::string>& algo_config)
    {
      (table.set_properties)(instance, algo_config);
    }
    std::map<std::string, std::string> get_properties() const { return (table.get_properties)(instance); }
    std::string scope() const { return (table.scope)(); }
    void run_preconditions(
      std::any& arg_ref_manager,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const Allen::Context& context)
    {
      return (table.run_preconditions)(instance, arg_ref_manager, runtime_options, constants, context);
    }
    void run_postconditions(
      std::any& arg_ref_manager,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const Allen::Context& context)
    {
      return (table.run_postconditions)(instance, arg_ref_manager, runtime_options, constants, context);
    }
  };

  // Tool to instantiate algorithms
  template<typename T>
  TypeErasedAlgorithm instantiate_algorithm(const std::string& name);

#define INSTANTIATE_ALGORITHM(TYPE)                                                      \
  template<>                                                                             \
  Allen::TypeErasedAlgorithm Allen::instantiate_algorithm<TYPE>(const std::string& name) \
  {                                                                                      \
    return TypeErasedAlgorithm {std::in_place_type<TYPE>, name};                         \
  }

  // Forward declare to use in Algorithm
  template<typename V>
  class Property;

  /**
   * @brief      In addition to functionality in BaseAlgorithm, algorithms may need to access properties shared with
   * other algorithms
   *
   */
  class Algorithm : public BaseAlgorithm {
  public:
    // Define empty contract container by default
    using contracts = std::tuple<>;

    template<typename T>
    using Property = Allen::Property<T>;

    Algorithm() = default;
    Algorithm(const Algorithm&) = delete;
    Algorithm& operator=(const Algorithm&) = delete;
    Algorithm(Algorithm&&) = delete;
    Algorithm& operator=(Algorithm&&) = delete;

    void set_properties(const std::map<std::string, std::string>& algo_config) override
    {
      for (auto kv : algo_config) {
        auto it = m_properties.find(kv.first);

        if (it == m_properties.end()) {
          std::cerr << "could not set " << kv.first << "=" << kv.second << "\n";
          const std::string error_message = "property " + kv.first + " does not exist";
          throw std::runtime_error {error_message};
        }
        else {
          it->second->from_string(kv.second);
        }
      }
    }

    template<typename T, typename R>
    void set_property_value(const R& value)
    {
      auto prop = const_cast<Allen::Property<T>*>(dynamic_cast<Allen::Property<T> const*>(get_prop(T::name)));
      prop->set_value(value);
    }

    // Gets the value of property with type T
    template<typename T>
    T property() const
    {
      const auto base_prop = get_prop(T::name);
      const auto prop = dynamic_cast<const Property<T>*>(base_prop);
      if (!prop) {
        const std::string error_message =
          "property " + std::string(T::name) + " not defined, perhaps member definition is missing";
        throw std::runtime_error {error_message};
      }
      return prop->get_value();
    }

    std::map<std::string, std::string> get_properties() const override
    {
      std::map<std::string, std::string> properties;
      for (const auto& kv : m_properties) {
        properties.emplace(kv.first, kv.second->to_string());
      }
      return properties;
    }

    bool register_property(std::string const& name, BaseProperty* property) override
    {
      auto r = m_properties.emplace(name, property);
      if (!std::get<1>(r)) {
        const std::string error_message = "could not register property " + name;
        throw std::runtime_error {error_message};
      }
      return std::get<1>(r);
    }

    // Setter and getter of name of the algorithm
    void set_name(const std::string& name) { m_name = name; }

    std::string name() const { return m_name; }

    template<typename Fn>
    auto host_function(const Fn& fn) const
    {
      return HostFunction<Fn> {m_properties, fn};
    }

    template<typename Fn>
    auto global_function(const Fn& fn) const
    {
      return GlobalFunction<Fn> {m_properties, fn};
    }

    PROPERTY(verbosity_t, "verbosity", "verbosity of algorithm", int);

  protected:
    BaseProperty const* get_prop(const std::string& prop_name) const override
    {
      if (m_properties.find(prop_name) != m_properties.end()) {
        return m_properties.at(prop_name);
      }
      return nullptr;
    }

  private:
    std::map<std::string, BaseProperty*> m_properties;
    std::string m_name = "";
    Property<verbosity_t> m_verbosity = {this, 3};
  };
} // namespace Allen
