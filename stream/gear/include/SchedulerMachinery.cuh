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
#include "TupleTools.cuh"

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
} // namespace

namespace Sch {
  // Motivation:
  //
  // I need somehow a struct with the inputs (things to malloc),
  // and another one for the outputs (things to free), like so:
  //
  // typedef std::tuple<
  //   In<a_t, dev_a, dev_b>,
  //   In<b_t, dev_c, dev_d>,
  //   In<c_t>
  // > input_t;
  //
  // typedef std::tuple<
  //   Out<a_t>,
  //   Out<b_t, dev_a>,
  //   Out<c_t, dev_c, dev_b, dev_d>,
  // > output_t;

  // Get the ParameterTuple from the function operator()
  template<typename Function>
  struct FunctionTraits;

  template<typename Function, typename T, typename S, typename R, typename... OtherArguments>
  struct FunctionTraits<void (Function::*)(const ArgumentRefManager<T, S, R>&, OtherArguments...) const> {
    using ParameterTuple = T;
    using ArgumentRefManagerType = ArgumentRefManager<T, S, R>;
  };

  template<typename Algorithm>
  struct AlgorithmTraits {
    using ParameterTuple = typename FunctionTraits<decltype(&Algorithm::operator())>::ParameterTuple;
    using ArgumentRefManagerType = typename FunctionTraits<decltype(&Algorithm::operator())>::ArgumentRefManagerType;
  };

  // Checks whether an argument T is in any of the arguments specified in the Algorithms
  template<typename T, typename Arguments>
  struct IsInAnyArgumentTuple;

  template<typename T>
  struct IsInAnyArgumentTuple<T, std::tuple<>> : std::false_type {
  };

  template<typename T, typename Arguments, typename... RestOfArguments>
  struct IsInAnyArgumentTuple<T, std::tuple<Arguments, RestOfArguments...>>
    : std::conditional_t<
        TupleContains<T, Arguments>::value,
        std::true_type,
        IsInAnyArgumentTuple<T, std::tuple<RestOfArguments...>>> {
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
    using t = typename std::conditional_t<
      IsInAnyArgumentTuple<Arg, std::tuple<OtherArguments, RestOfArguments...>>::value,
      previous_t,
      typename TupleAppend<previous_t, Arg>::t>;
  };

  // Consume the algorithms and put their dependencies one by one
  template<typename Arguments>
  struct OutDependenciesImpl;

  template<typename Arguments>
  struct OutDependenciesImpl<std::tuple<Arguments>> {
    using t = std::tuple<>;
  };

  template<typename Arguments, typename NextArguments, typename... RestOfArguments>
  struct OutDependenciesImpl<std::tuple<Arguments, NextArguments, RestOfArguments...>> {
    using previous_t = typename OutDependenciesImpl<std::tuple<NextArguments, RestOfArguments...>>::t;
    using t = typename TupleAppend<
      previous_t,
      typename ArgumentsNotIn<Arguments, std::tuple<NextArguments, RestOfArguments...>>::t>::t;
  };

  // Helper to calculate OUT dependencies
  template<typename ConfiguredSequence>
  struct OutDependencies;

  template<typename Arguments, typename... RestOfArguments>
  struct OutDependencies<std::tuple<Arguments, RestOfArguments...>> {
    using t = typename TupleReverse<typename TupleAppend<
      typename OutDependenciesImpl<typename std::tuple<Arguments, RestOfArguments...>>::t,
      std::tuple<>>::t>::t;
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
    using t =
      typename TupleAppend<previous_t, typename ArgumentsNotIn<Arguments, std::tuple<RestOfArguments...>>::t>::t;
  };

  template<typename ConfiguredArguments>
  using InDependencies = InDependenciesImpl<typename TupleReverse<ConfiguredArguments>::t>;

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

  // Configure constants for algorithms in the sequence
  template<typename Dependencies, typename Indices>
  struct ConfigureAlgorithmSequenceImpl;

  template<typename Tuple>
  struct ConfigureAlgorithmSequenceImpl<Tuple, std::index_sequence<>> {
    static constexpr void configure(Tuple&, const std::map<std::string, std::map<std::string, std::string>>&) {};
  };

  template<typename Tuple, unsigned long I, unsigned long... Is>
  struct ConfigureAlgorithmSequenceImpl<Tuple, std::index_sequence<I, Is...>> {
    static constexpr void configure(
      Tuple& algs,
      const std::map<std::string, std::map<std::string, std::string>>& config)
    {
      // Initialization-time:
      // * Set properties
      const auto algorithm_name = std::get<I>(algs).name();
      auto& algorithm = std::get<I>(algs);
      if (config.find(algorithm_name) != config.end()) {
        algorithm.set_properties(config.at(algorithm_name));
      }

      // * Invoke void initialize() const, iff it exists
      if constexpr (has_member_fn<decltype(algorithm)>::value) {
        algorithm.init();
      };

      ConfigureAlgorithmSequenceImpl<Tuple, std::index_sequence<Is...>>::configure(algs, config);
    }
  };

  template<typename Tuple>
  struct ConfigureAlgorithmSequence {
    static constexpr void configure(
      Tuple& algs,
      const std::map<std::string, std::map<std::string, std::string>>& config)
    {
      ConfigureAlgorithmSequenceImpl<Tuple, std::make_index_sequence<std::tuple_size<Tuple>::value>>::configure(
        algs, config);
    }
  };

  // Return constants for algorithms in the sequence
  template<typename Dependencies, typename Indices>
  struct GetSequenceConfigurationImpl;

  template<typename Tuple>
  struct GetSequenceConfigurationImpl<Tuple, std::index_sequence<>> {
    static auto get(Tuple const&, std::map<std::string, std::map<std::string, std::string>>& config) { return config; };
  };

  template<typename Tuple, unsigned long I, unsigned long... Is>
  struct GetSequenceConfigurationImpl<Tuple, std::index_sequence<I, Is...>> {
    static auto get(Tuple const& algs, std::map<std::string, std::map<std::string, std::string>>& config)
    {
      const auto& algorithm = std::get<I>(algs);
      auto props = algorithm.get_properties();
      config.emplace(algorithm.name(), props);
      return GetSequenceConfigurationImpl<Tuple, std::index_sequence<Is...>>::get(algs, config);
    }
  };

  template<typename Tuple>
  struct GetSequenceConfiguration {
    static auto get(Tuple const& algs, std::map<std::string, std::map<std::string, std::string>>& config)
    {
      return GetSequenceConfigurationImpl<Tuple, std::make_index_sequence<std::tuple_size<Tuple>::value>>::get(
        algs, config);
    }
  };

  /**
   * @brief Produces a single argument reference.
   */
  template<typename ArgumentsTuple, typename ConfiguredArgument>
  struct ProduceSingleArgument {
    constexpr static auto& produce(std::array<ArgumentData, std::tuple_size_v<ArgumentsTuple>>& arguments_tuple)
    {
      constexpr auto index_of_T = index_of_v<ConfiguredArgument, ArgumentsTuple>;
      return std::get<index_of_T>(arguments_tuple);
    }
  };

  /**
   * @brief Produces a list of argument references.
   */
  template<typename ArgumentsTuple, typename ArgumentRefManager, typename ConfiguredArguments>
  struct ProduceArgumentsTupleHelper;

  template<typename ArgumentsTuple, typename ArgumentRefManager, typename... ConfiguredArguments>
  struct ProduceArgumentsTupleHelper<ArgumentsTuple, ArgumentRefManager, std::tuple<ConfiguredArguments...>> {
    constexpr static ArgumentRefManager produce(
      std::array<ArgumentData, std::tuple_size_v<ArgumentsTuple>>& arguments_tuple)
    {
      return ArgumentRefManager {
        {ProduceSingleArgument<ArgumentsTuple, ConfiguredArguments>::produce(arguments_tuple)...}};
    }
  };

  /**
   * @brief Produces a single algorithm with references to arguments.
   */
  template<typename ArgumentsTuple, typename Algorithm, typename ConfiguredArguments>
  struct ProduceArgumentsTuple {
    constexpr static typename AlgorithmTraits<Algorithm>::ArgumentRefManagerType produce(
      std::array<ArgumentData, std::tuple_size_v<ArgumentsTuple>>& arguments_database)
    {
      return ProduceArgumentsTupleHelper<
        ArgumentsTuple,
        typename AlgorithmTraits<Algorithm>::ArgumentRefManagerType,
        ConfiguredArguments>::produce(arguments_database);
    }
  };

  /**
   * @brief Runs the sequence tuple (implementation).
   */
  template<typename Scheduler, typename SetSizeArguments, typename VisitArguments, typename Indices>
  struct RunSequenceTupleImpl;

  template<typename Scheduler, typename... SetSizeArguments, typename... VisitArguments>
  struct RunSequenceTupleImpl<
    Scheduler,
    std::tuple<SetSizeArguments...>,
    std::tuple<VisitArguments...>,
    std::index_sequence<>> {
    constexpr static void run(Scheduler&, SetSizeArguments&&..., VisitArguments&&...) {}
  };

  template<
    typename Scheduler,
    typename... SetSizeArguments,
    typename... VisitArguments,
    unsigned long I,
    unsigned long... Is>
  struct RunSequenceTupleImpl<
    Scheduler,
    std::tuple<SetSizeArguments...>,
    std::tuple<VisitArguments...>,
    std::index_sequence<I, Is...>> {
    constexpr static void
    run(Scheduler& scheduler, SetSizeArguments&&... set_size_arguments, VisitArguments&&... visit_arguments)
    {
      using t = typename std::tuple_element<I, typename Scheduler::configured_sequence_t>::type;
      using configured_arguments_t =
        typename std::tuple_element<I, typename Scheduler::configured_sequence_arguments_t>::type;
      auto arguments_tuple =
        ProduceArgumentsTuple<typename Scheduler::arguments_tuple_t, t, configured_arguments_t>::produce(
          scheduler.argument_manager.argument_database());

      // Sets the arguments sizes, setups the scheduler and visits the algorithm.
      std::get<I>(scheduler.sequence_tuple)
        .set_arguments_size(arguments_tuple, std::forward<SetSizeArguments>(set_size_arguments)...);

      scheduler.template setup<I>();

      std::get<I>(scheduler.sequence_tuple)
        .operator()(arguments_tuple, std::forward<VisitArguments>(visit_arguments)...);

      RunSequenceTupleImpl<
        Scheduler,
        std::tuple<SetSizeArguments...>,
        std::tuple<VisitArguments...>,
        std::index_sequence<Is...>>::
        run(
          scheduler,
          std::forward<SetSizeArguments>(set_size_arguments)...,
          std::forward<VisitArguments>(visit_arguments)...);
    }
  };

  /**
   * @brief Runs a sequence of algorithms.
   *
   * @tparam Tuple            Sequence of algorithms
   * @tparam SetSizeArguments Arguments to set_arguments_size
   * @tparam VisitArguments   Arguments to visit
   */
  template<typename Scheduler, typename SetSizeArguments, typename VisitArguments>
  struct RunSequenceTuple;

  template<typename Scheduler, typename... SetSizeArguments, typename... VisitArguments>
  struct RunSequenceTuple<Scheduler, std::tuple<SetSizeArguments...>, std::tuple<VisitArguments...>> {
    constexpr static void
    run(Scheduler& scheduler, SetSizeArguments&&... set_size_arguments, VisitArguments&&... visit_arguments)
    {
      RunSequenceTupleImpl<
        Scheduler,
        std::tuple<SetSizeArguments...>,
        std::tuple<VisitArguments...>,
        std::make_index_sequence<std::tuple_size<typename Scheduler::configured_sequence_t>::value>>::
        run(
          scheduler,
          std::forward<SetSizeArguments>(set_size_arguments)...,
          std::forward<VisitArguments>(visit_arguments)...);
    }
  };

  /**
   * @brief Runs the PrChecker for all configured algorithms in the sequence.
   */
  template<typename ConfiguredSequence, typename Arguments>
  struct RunChecker;

  template<typename... Arguments>
  struct RunChecker<std::tuple<>, std::tuple<Arguments...>> {
    constexpr static void check(Arguments&&...) {}
  };

  template<typename Algorithm, typename... Algorithms, typename... Arguments>
  struct RunChecker<std::tuple<Algorithm, Algorithms...>, std::tuple<Arguments...>> {
    constexpr static void check(Arguments&&... arguments)
    {
      SequenceVisitor<Algorithm>::check(std::forward<Arguments>(arguments)...);

      RunChecker<std::tuple<Algorithms...>, std::tuple<Arguments...>>::check(std::forward<Arguments>(arguments)...);
    }
  };
} // namespace Sch
