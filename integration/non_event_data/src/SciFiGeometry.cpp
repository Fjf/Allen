#include <string>

#include <cuda_runtime.h>
#include <Common.h>
#include <Consumers.h>

namespace {
  using std::string;
  using std::to_string;
}

Consumers::SciFiGeometry::SciFiGeometry(std::vector<char>& host_geometry, char*& dev_geometry)
  : m_host_geometry{host_geometry},
    m_dev_geometry{dev_geometry} {
}

void Consumers::SciFiGeometry::consume(std::vector<char> const& data) {
  auto& dev_geometry = m_dev_geometry.get();
  auto& host_geometry = m_host_geometry.get();
  if (host_geometry.empty()) {
    cudaCheck(cudaMalloc((void**) &dev_geometry, data.size()));
  } else if (host_geometry.size() != data.size()) {
    throw StrException{string{"sizes don't match: "} + to_string(host_geometry.size()) + " " + to_string(data.size())};
  }
  host_geometry = std::move(data);
  cudaCheck(cudaMemcpy(dev_geometry, host_geometry.data(), host_geometry.size(), cudaMemcpyHostToDevice));
}
