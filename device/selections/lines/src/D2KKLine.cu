#include "D2KKLine.cuh"

INSTANTIATE_LINE(d2kk_line::d2kk_line_t, d2kk_line::Parameters)

__device__ bool d2kk_line::d2kk_line_t::select(
  const Parameters& parameters,
  std::tuple<const VertexFit::TrackMVAVertex&> input) const
{
  const auto& vertex = std::get<0>(input);
  if (vertex.chi2 < 0) {
    return false;
  }
  const bool decision = vertex.pt() > parameters.minComboPt && vertex.chi2 < parameters.maxVertexChi2 &&
                        vertex.eta > parameters.minEta && vertex.eta < parameters.maxEta &&
                        vertex.minpt > parameters.minTrackPt && vertex.minipchi2 > parameters.minTrackIPChi2 &&
                        fabsf(vertex.m(parameters.mK, parameters.mK) - parameters.mD) < parameters.massWindow;
  return decision;
}
