#include <string>

#include <cuda_runtime.h>
#include <Common.h>
#include <Consumers.h>

namespace {
  using std::string;
  using std::to_string;
}

Consumers::RawGeometry::RawGeometry(char*& dev_geometry)
  : m_dev_geometry{dev_geometry} {
}

void Consumers::RawGeometry::consume(std::vector<char> const& data) {
  if (!m_dev_geometry.get()) {
    // Allocate space
    char* p = nullptr;
    cudaCheck(cudaMalloc((void**) &m_dev_geometry.get(), data.size()));
    m_size = data.size();
  } else if (m_size != data.size()) {
    throw StrException{string{"sizes don't match: "} + to_string(m_size) + " " + to_string(data.size())};
  }
  cudaCheck(cudaMemcpy(m_dev_geometry.get(), data.data(), m_size, cudaMemcpyHostToDevice));
}

Consumers::BasicGeometry::BasicGeometry(gsl::span<char>& dev_geometry)
  : m_dev_geometry{dev_geometry} {
}

void Consumers::BasicGeometry::consume(std::vector<char> const& data) {
  auto& dev_geometry = m_dev_geometry.get();
  if (dev_geometry.empty()) {
    // Allocate space
    char* p = nullptr;
    cudaCheck(cudaMalloc((void**) &p, data.size()));
    dev_geometry = gsl::span{p, data.size()};
  } else if (dev_geometry.size() != data.size()) {
    throw StrException{string{"sizes don't match: "} + to_string(dev_geometry.size()) + " " + to_string(data.size())};
  }
  cudaCheck(cudaMemcpy(dev_geometry.data(), data.data(), data.size(), cudaMemcpyHostToDevice));
}
