/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <tuple>
#include <functional>
#include <type_traits>

/**
 * @brief Checks if a tuple contains type T, and obtains index.
 *        index will be the length of the tuple if the type was not found.
 *
 *        Some examples of its usage:
 *
 *        if (TupleContains<int, decltype(t)>::value) {
 *          std::cout << "t contains int" << std::endl;
 *        }
 *
 *        std::cout << "int in index " << TupleContains<int, decltype(t)>::index << std::endl;
 */

template<typename T, typename Tuple>
struct TupleContains;

template<typename T, typename... Ts>
struct TupleContains<T, std::tuple<Ts...>> : std::bool_constant<((std::is_same_v<T, Ts> || ...))> {
  static constexpr auto index()
  {
    int idx = 0;
// FIXME: remove the pragma workaround when we move to clang 10.
// Clang 8 wrongfully flags the logical or as unsequenced.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsequenced"
#endif
    bool contains = ((++idx, std::is_same_v<T, Ts>) || ...);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    return contains ? idx - 1 : idx;
  }
};

template<typename T, typename Tuple>
inline constexpr std::size_t index_of_v = TupleContains<T, Tuple>::index();

// Appends a Tuple with the Element
namespace details {
  template<typename, typename>
  struct TupleAppend;

  template<typename... T, typename E>
  struct TupleAppend<std::tuple<T...>, E> {
    using type = std::tuple<T..., E>;
  };
} // namespace details
template<typename Tuple, typename Element>
using append_to_tuple_t = typename details::TupleAppend<Tuple, Element>::type;

// Appends a Tuple with the Element
namespace details {
  template<typename, typename>
  struct TuplePrepend;

  template<typename E, typename... T>
  struct TuplePrepend<E, std::tuple<T...>> {
    using type = std::tuple<E, T...>;
  };
} // namespace details

template<typename Element, typename Tuple>
using prepend_to_tuple_t = typename details::TuplePrepend<Element, Tuple>::type;

// Reverses a tuple
namespace details {
  template<typename T, typename I>
  struct ReverseTuple;

  template<typename T, auto... Is>
  struct ReverseTuple<T, std::index_sequence<Is...>> {
    using type = std::tuple<std::tuple_element_t<sizeof...(Is) - 1 - Is, T>...>;
  };
} // namespace details
template<typename Tuple>
using reverse_tuple_t = typename details::ReverseTuple<Tuple, std::make_index_sequence<std::tuple_size_v<Tuple>>>::type;

namespace details {
  template<typename...>
  struct ConcatTuple;

  template<typename... First, typename... Second>
  struct ConcatTuple<std::tuple<First...>, std::tuple<Second...>> {
    using type = std::tuple<First..., Second...>;
  };

  template<typename T1, typename T2, typename... Ts>
  struct ConcatTuple<T1, T2, Ts...> {
    using type = typename ConcatTuple<typename ConcatTuple<T1, T2>::type, Ts...>::type;
  };
} // namespace details

template<typename... Tuples>
using cat_tuples_t = typename details::ConcatTuple<Tuples...>::type;

namespace details {
  template<typename T>
  struct FlattenTuple;

  template<>
  struct FlattenTuple<std::tuple<>> {
    using type = std::tuple<>;
  };

  template<typename... InTuple, typename... Ts>
  struct FlattenTuple<std::tuple<std::tuple<InTuple...>, Ts...>> {
    using type = cat_tuples_t<std::tuple<InTuple...>, typename FlattenTuple<std::tuple<Ts...>>::type>;
  };

  template<typename T, typename... Ts>
  struct FlattenTuple<std::tuple<T, Ts...>> {
    using type = cat_tuples_t<std::tuple<T>, typename FlattenTuple<std::tuple<Ts...>>::type>;
  };
} // namespace details

template<typename Tuple>
using flatten_tuple_t = typename details::FlattenTuple<Tuple>::type;

// Access to tuple elements by checking whether they inherit from a Base type
template<typename Base, typename Tuple, std::size_t I = 0>
struct tuple_ref_index;

template<typename Base, typename Head, typename... Tail, std::size_t I>
struct tuple_ref_index<Base, std::tuple<Head, Tail...>, I>
  : std::conditional_t<
      std::is_base_of_v<std::decay_t<Base>, std::decay_t<Head>>,
      std::integral_constant<std::size_t, I>,
      tuple_ref_index<Base, std::tuple<Tail...>, I + 1>> {
};

template<typename Base, typename Tuple>
auto tuple_ref_by_inheritance(Tuple&& tuple)
  -> decltype(std::get<tuple_ref_index<Base, std::decay_t<Tuple>>::value>(std::forward<Tuple>(tuple)))
{
  return std::get<tuple_ref_index<Base, std::decay_t<Tuple>>::value>(std::forward<Tuple>(tuple));
}

namespace Allen {
  template<typename T, typename U>
  using forward_type_t = std::conditional_t<std::is_const_v<T>, std::add_const_t<U>, std::remove_const_t<U>>;

  template<typename T>
  using bool_as_char_t = std::conditional_t<std::is_same_v<std::decay_t<T>, bool>, char, std::decay_t<T>>;

  /**
   * @brief Checks whether class U is derived from class T,
   *        where T is a templated class.
   */
  template<template<class...> class T, class U>
  struct isDerivedFrom {
  private:
    template<class... V>
    static decltype(static_cast<const T<V...>&>(std::declval<U>()), std::true_type {}) test(const T<V...>&);

    static std::false_type test(...);

  public:
    static constexpr bool value = decltype(isDerivedFrom::test(std::declval<U>()))::value;
  };
} // namespace Allen
