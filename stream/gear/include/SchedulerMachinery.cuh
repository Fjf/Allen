/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <tuple>
#include <functional>
#include <type_traits>
#include "Logger.h"
#include "Argument.cuh"
#include "ArgumentManager.cuh"
#include "AllenTypeTraits.cuh"

namespace {
  // SFINAE-based invocation of member function iff class provides it.
  // This is just one way to write a type trait, it's not necessarily
  // the best way. You could use the Detection Idiom, for example
  // (http://en.cppreference.com/w/cpp/experimental/is_detected).
  template<typename T, typename = void>
  struct has_member_fn : std::false_type {
  };

  // std::void_t is a C++17 library feature. It can be replaced
  // with your own implementation of void_t, or often by making the
  // decltype expression void, whether by casting or by comma operator
  // (`decltype(expr, void())`)
  template<typename T>
  struct has_member_fn<T, std::void_t<decltype(std::declval<T>().init())>> : std::true_type {
  };

  // SFINAE-based check of View. Require a type "deps" in it.
  template<typename T, typename = void>
  struct is_view : std::false_type {
  };

  template<typename T>
  struct is_view<T, std::void_t<typename T::type::deps>> : std::true_type {
  };
} // namespace

namespace Sch {
  // Get the ArgumentRefManagerType from the function operator()
  template<typename Function>
  struct FunctionTraits;

  template<typename Function, typename T, typename S, typename R, typename... OtherArguments>
  struct FunctionTraits<void (Function::*)(const ArgumentRefManager<T, S, R>&, OtherArguments...) const> {
    using ArgumentRefManagerType = ArgumentRefManager<T, S, R>;
  };

  template<typename Algorithm>
  struct AlgorithmTraits {
    using ArgumentRefManagerType = typename FunctionTraits<decltype(&Algorithm::operator())>::ArgumentRefManagerType;
  };

  template<typename T, typename Tuple>
  struct TupleContainsDecay;

  template<typename T, typename... Ts>
  struct TupleContainsDecay<T, std::tuple<Ts...>>
    : std::bool_constant<((std::is_base_of_v<std::decay_t<Ts>, std::decay_t<T>> || ...))> {
  };

  template<typename T, typename Tuple, typename = void>
  struct TupleContainsWithViews;

  template<typename T>
  struct TupleContainsWithViews<T, std::tuple<>, void> : std::bool_constant<false> {
  };

  template<typename T, typename OtherT, typename... Ts>
  struct TupleContainsWithViews<T, std::tuple<OtherT, Ts...>, std::enable_if_t<!is_view<OtherT>::value>>
    : std::bool_constant<std::is_same_v<T, OtherT> || TupleContainsWithViews<T, std::tuple<Ts...>>::value> {
  };

  template<typename T, typename OtherT, typename... Ts>
  struct TupleContainsWithViews<T, std::tuple<OtherT, Ts...>, std::enable_if_t<is_view<OtherT>::value>>
    : std::bool_constant<
        std::is_same_v<T, OtherT> || TupleContainsWithViews<T, std::tuple<Ts...>>::value ||
        TupleContainsDecay<T, typename OtherT::type::deps>::value> {
  };

  // Checks whether an argument T is in any of the arguments specified in the Algorithms
  template<typename T, typename Arguments>
  struct IsInAnyArgumentTuple;

  template<typename T, typename... Arguments>
  struct IsInAnyArgumentTuple<T, std::tuple<Arguments...>> : std::disjunction<TupleContainsWithViews<T, Arguments>...> {
  };

  // A mechanism to only return the arguments in Algorithm
  // that are not on any of the other RestOfArguments
  template<typename Arguments, typename RestOfArguments>
  struct ArgumentsNotIn;

  // If there are no other RestOfArguments, return all the types
  template<typename... Arguments>
  struct ArgumentsNotIn<std::tuple<Arguments...>, std::tuple<>> {
    using t = std::tuple<Arguments...>;
  };

  // Weird case: No dependencies in algo
  template<typename... RestOfArguments>
  struct ArgumentsNotIn<std::tuple<>, std::tuple<RestOfArguments...>> {
    using t = std::tuple<>;
  };

  template<typename Arg, typename... Args, typename OtherArguments, typename... RestOfArguments>
  struct ArgumentsNotIn<std::tuple<Arg, Args...>, std::tuple<OtherArguments, RestOfArguments...>> {
    // Types unused from Args...
    using previous_t = typename ArgumentsNotIn<std::tuple<Args...>, std::tuple<OtherArguments, RestOfArguments...>>::t;

    // We append Arg only if it is _not_ on the previous algorithms
    using t = std::conditional_t<
      IsInAnyArgumentTuple<Arg, std::tuple<OtherArguments, RestOfArguments...>>::value,
      previous_t,
      append_to_tuple_t<previous_t, Arg>>;
  };

  // Consume the algorithms and put their dependencies one by one
  template<typename Arguments>
  struct OutDependenciesImpl;

  template<typename Arguments>
  struct OutDependenciesImpl<std::tuple<Arguments>> {
    using t = std::tuple<>;
  };

  template<typename Arguments, typename... NextArguments>
  struct OutDependenciesImpl<std::tuple<Arguments, NextArguments...>> {
    static_assert(sizeof...(NextArguments) != 0);
    using previous_t = typename OutDependenciesImpl<std::tuple<NextArguments...>>::t;
    using t = append_to_tuple_t<previous_t, typename ArgumentsNotIn<Arguments, std::tuple<NextArguments...>>::t>;
  };

  // Helper to calculate OUT dependencies
  template<typename ConfiguredSequence>
  struct OutDependencies;

  template<typename... Arguments>
  struct OutDependencies<std::tuple<Arguments...>> {
    static_assert(sizeof...(Arguments) != 0);
    using t = reverse_tuple_t<
      append_to_tuple_t<typename OutDependenciesImpl<typename std::tuple<Arguments...>>::t, std::tuple<>>>;
  };

  // Consume the algorithms and put their dependencies one by one
  template<typename Arguments>
  struct InDependenciesImpl;

  template<>
  struct InDependenciesImpl<std::tuple<>> {
    using t = std::tuple<>;
  };

  template<typename Arguments, typename... RestOfArguments>
  struct InDependenciesImpl<std::tuple<Arguments, RestOfArguments...>> {
    using previous_t = typename InDependenciesImpl<std::tuple<RestOfArguments...>>::t;
    using t = append_to_tuple_t<previous_t, typename ArgumentsNotIn<Arguments, std::tuple<RestOfArguments...>>::t>;
  };

  template<typename ConfiguredArguments>
  using InDependencies = InDependenciesImpl<reverse_tuple_t<ConfiguredArguments>>;

  // Helper to just print the arguments
  template<typename Arguments>
  struct PrintArguments;

  template<>
  struct PrintArguments<std::tuple<>> {
    static constexpr void print() {}
  };

  template<typename Argument>
  struct PrintArguments<std::tuple<Argument>> {
    static constexpr void print() { info_cout << "'" << Argument::name << "'"; }
  };

  template<typename Argument, typename ArgumentSecond, typename... Arguments>
  struct PrintArguments<std::tuple<Argument, ArgumentSecond, Arguments...>> {
    static constexpr void print()
    {
      info_cout << "'" << Argument::name << "', ";
      PrintArguments<std::tuple<ArgumentSecond, Arguments...>>::print();
    }
  };

  // Iterate the types (In or Out) and print them for each iteration
  template<typename Dependencies>
  struct PrintAlgorithmDependencies;

  template<>
  struct PrintAlgorithmDependencies<std::tuple<>> {
    static constexpr void print() {};
  };

  template<typename Algorithm, typename... Arguments, typename... Dependencies>
  struct PrintAlgorithmDependencies<
    std::tuple<ScheduledDependencies<Algorithm, std::tuple<Arguments...>>, Dependencies...>> {
    static constexpr void print()
    {
      // info_cout << "  ['" << Algorithm::name << "', [";
      // PrintArguments<std::tuple<Arguments...>>::print();
      // info_cout << "]]," << std::endl;

      PrintAlgorithmDependencies<std::tuple<Dependencies...>>::print();
    }
  };

  // Print the configured sequence
  template<typename Dependencies>
  struct PrintAlgorithmSequence;

  template<>
  struct PrintAlgorithmSequence<std::tuple<>> {
    static constexpr void print() {};
  };

  template<typename Algorithm, typename... Algorithms>
  struct PrintAlgorithmSequence<std::tuple<Algorithm, Algorithms...>> {
    static constexpr void print()
    {
      // info_cout << " " << Algorithm::name << std::endl;
      PrintAlgorithmSequence<std::tuple<Algorithms...>>::print();
    }
  };

  template<typename Dependencies>
  struct PrintAlgorithmSequenceDetailed;

  template<>
  struct PrintAlgorithmSequenceDetailed<std::tuple<>> {
    static constexpr void print() {};
  };

  template<typename Algorithm, typename... Algorithms>
  struct PrintAlgorithmSequenceDetailed<std::tuple<Algorithm, Algorithms...>> {
    static constexpr void print()
    {
      // info_cout << "  ['" << Algorithm::name << "', [";
      // PrintArguments<typename Algorithm::Arguments>::print();
      // info_cout << "]]," << std::endl;

      PrintAlgorithmSequenceDetailed<std::tuple<Algorithms...>>::print();
    }
  };

  /**
   * @brief Produces a list of argument references.
   */
  template<typename ArgumentsTuple, typename ArgumentRefManager, typename ConfiguredArguments>
  struct ProduceArgumentsTupleHelper;

  template<typename ArgumentsTuple, typename ArgumentRefManager, typename... ConfiguredArguments>
  struct ProduceArgumentsTupleHelper<ArgumentsTuple, ArgumentRefManager, std::tuple<ConfiguredArguments...>> {
    constexpr static auto produce(std::array<ArgumentData, std::tuple_size_v<ArgumentsTuple>>& arguments_array)
    {
      return ArgumentRefManager {{arguments_array[index_of_v<ConfiguredArguments, ArgumentsTuple>]...}};
    }
  };

  /**
   * @brief Produces a single algorithm with references to arguments.
   */
  template<typename ArgumentsTuple, typename Algorithm, typename ConfiguredArguments>
  struct ProduceArgumentsTuple {
    constexpr static auto produce(std::array<ArgumentData, std::tuple_size_v<ArgumentsTuple>>& arguments_database)
    {
      return ProduceArgumentsTupleHelper<
        ArgumentsTuple,
        typename AlgorithmTraits<Algorithm>::ArgumentRefManagerType,
        ConfiguredArguments>::produce(arguments_database);
    }
  };

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

  // Checks whether the sequence contains any validation algorithm
  template<typename AlgorithmT, typename Dependencies>
  struct ContainsAlgorithmType;

  template<typename T, typename... Ts>
  struct ContainsAlgorithmType<T, std::tuple<Ts...>>
    : std::bool_constant<((std::is_base_of_v<std::decay_t<T>, std::decay_t<Ts>> || ...))> {
  };
} // namespace Sch
