/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

namespace Detail {

  // Specialization for std::tuple
  template<std::size_t I, class Archive, class... Args>
  struct tuple_serialize_helper {
    static int impl(Archive& ar, std::tuple<Args...>& t, const unsigned int version)
    {
      ar& std::get<I - 1>(t);
      tuple_serialize_helper<I - 1, Archive, Args...>::impl(ar, t, version);
      return 0;
    }
  };

  template<class Archive, class... Args>
  struct tuple_serialize_helper<0, Archive, Args...> {
    static int impl(Archive& ar, std::tuple<Args...>& t, const unsigned int)
    {
      ar& std::get<0>(t);
      return 0;
    }
  };

} // namespace Detail

namespace boost {
  namespace serialization {

    // Serialize RunInfo
    template<typename Archive, class... Args>
    auto serialize(Archive& archive, std::tuple<Args...>& t, const unsigned int v) -> void
    {
      Detail::tuple_serialize_helper<sizeof...(Args), Archive, Args...>::impl(archive, t, v);
    }

  } // namespace serialization
} // namespace boost
