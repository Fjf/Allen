/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <string>

#include <BackendCommon.h>
#include <Common.h>
#include <Consumers.h>

namespace {
  using std::string;
  using std::to_string;
} // namespace

Consumers::RawGeometry::RawGeometry(char*& dev_geometry) : m_dev_geometry {dev_geometry} {}

void Consumers::RawGeometry::consume(std::vector<char> const& data)
{
  if (!m_dev_geometry.get()) {
    // Allocate space
    Allen::malloc((void**) &m_dev_geometry.get(), data.size());
    m_size = data.size();
  }
  else if (m_size != data.size()) {
    throw StrException {string {"sizes don't match: "} + to_string(m_size) + " " + to_string(data.size())};
  }
  Allen::memcpy(m_dev_geometry.get(), data.data(), m_size, Allen::memcpyHostToDevice);
}

Consumers::BasicGeometry::BasicGeometry(gsl::span<char>& dev_geometry) : m_dev_geometry {dev_geometry} {}

void Consumers::BasicGeometry::consume(std::vector<char> const& data)
{
  auto& dev_geometry = m_dev_geometry.get();
  if (dev_geometry.empty()) {
    // Allocate space
    char* p = nullptr;
    Allen::malloc((void**) &p, data.size());
    dev_geometry = gsl::span {p, static_cast<span_size_t<char>>(data.size())};
  }
  else if ((size_t) dev_geometry.size() != data.size()) {
    throw StrException {string {"sizes don't match: "} + to_string(dev_geometry.size()) + " " + to_string(data.size())};
  }
  Allen::memcpy(dev_geometry.data(), data.data(), data.size(), Allen::memcpyHostToDevice);
}
