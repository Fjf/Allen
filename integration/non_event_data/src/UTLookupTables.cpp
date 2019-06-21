#include <string>

#include <cuda_runtime.h>
#include <Common.h>
#include <Consumers.h>

namespace {
  using std::string;
  using std::to_string;
} // namespace

Consumers::UTLookupTables::UTLookupTables(PrUTMagnetTool*& tool) : m_tool {tool} {}

void Consumers::UTLookupTables::consume(std::vector<char> const& data)
{
  char const* p = data.data();
  int const* layout = reinterpret_cast<int const*>(p);
  p += sizeof(int);
  int nVar = layout[0];
  assert(nVar == 2);
  std::vector<int> nBins(nVar);
  std::copy_n(layout + 1, nVar, nBins.begin());
  assert(nBins[0] == 3);
  assert(nBins[1] == 30);
  p += nVar * sizeof(int);
  size_t const* table_size = reinterpret_cast<size_t const*>(p);
  assert(table_size[0] == 124);
  p += sizeof(size_t);
  float const* deflection = reinterpret_cast<float const*>(p);
  p += table_size[0] * sizeof(float);

  layout = reinterpret_cast<int const*>(p);
  p += sizeof(int);
  nVar = layout[0];
  assert(nVar == 3);
  nBins.resize(nVar);
  std::copy_n(layout + 1, nVar, nBins.begin());
  assert(nBins[0] == 30);
  assert(nBins[1] == 10);
  assert(nBins[2] == 10);
  p += nVar * sizeof(int);
  table_size = reinterpret_cast<size_t const*>(p);
  assert(table_size[0] == 3751);
  p += sizeof(size_t);
  float const* bdl = reinterpret_cast<float const*>(p);
  p += table_size[0] * sizeof(float);

  if (!m_tool) {
    cudaCheck(cudaMalloc((void**) &m_tool.get(), sizeof(PrUTMagnetTool)));
    m_size = sizeof(PrUTMagnetTool);
  }
  if (m_size != (data.size() - 7 * sizeof(int) - 2 * sizeof(size_t))) {
    throw StrException {string {"sizes don't match: "} + to_string(m_size) + " " + to_string(data.size())};
  }

  PrUTMagnetTool host_tool {deflection, bdl};

  // deflection table
  cudaCheck(cudaMemcpy(m_tool.get(), &host_tool, sizeof(PrUTMagnetTool), cudaMemcpyHostToDevice));
}
