/*****************************************************************************\
* (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#ifndef DUMPUTILS_H
#define DUMPUTILS_H

#include <boost/filesystem.hpp>
#include <boost/interprocess/streams/vectorstream.hpp>

#include <Detector/Muon/TileID.h>
#include <Kernel/STLExtensions.h>
#include <fstream>
#include <functional>

#include <string>
#include <type_traits>
#include <vector>

#include <Common.h>

template<class T>
void optional_resize(T&, size_t)
{}

template<class... Args>
void optional_resize(std::vector<Args...>& v, size_t s)
{
  v.resize(s);
}

namespace DumpUtils {

  bool createDirectory(boost::filesystem::path dir);

  namespace detail {

    template<typename T>
    std::ostream& write(std::ostream& os, const T& t)
    {
      // if you would like to know why there is a check for trivially copyable,
      // please read the 'notes' section of https://en.cppreference.com/w/cpp/types/is_trivially_copyable
      if constexpr (std::is_same_v<T, gsl::span<const std::byte>>) {
        return os.write(reinterpret_cast<const char*>(t.data()), t.size());
      }
      else if constexpr (std::is_trivially_copyable_v<T> && !gsl::details::is_span<T>::value) {
        return os.write(reinterpret_cast<const char*>(&t), sizeof(T));
      }
      else {
        static_assert(std::is_trivially_copyable_v<typename T::value_type>);
        return write(os, as_bytes(LHCb::make_span(t)));
      }
    }

  } // namespace detail

  class Writer {

    boost::interprocess::basic_vectorstream<std::vector<char>> m_buffer;

  public:
    template<typename... Args>
    Writer& write(Args&&... args)
    {
      (detail::write(m_buffer, std::forward<Args>(args)), ...);
      return *this;
    }

    std::vector<char> const& buffer() { return m_buffer.vector(); }
  };

  class FileWriter {
    std::ofstream m_f;

  public:
    FileWriter(const std::string& name) : m_f {name, std::ios::out | std::ios::binary} {}

    template<typename... Args>
    FileWriter& write(Args&&... args)
    {
      (detail::write(m_f, std::forward<Args>(args)), ...);
      return *this;
    }
  };

  using Dump = std::tuple<std::vector<char>, std::string, std::string>;
  using Dumps = std::vector<Dump>;

} // namespace DumpUtils

namespace MuonUtils {
  size_t size_index(
    std::array<unsigned int, 16> const& sizeXOffset,
    std::array<int, 16> const& gridX,
    std::array<int, 16> const& gridY,
    LHCb::Detector::Muon::TileID const& tile);
}

#endif
