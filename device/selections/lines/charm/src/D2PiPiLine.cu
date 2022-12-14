/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "D2PiPiLine.cuh"

INSTANTIATE_LINE(d2pipi_line::d2pipi_line_t, d2pipi_line::Parameters)

__device__ bool d2pipi_line::d2pipi_line_t::select(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input)
{
  const auto particle = std::get<0>(input);
  const auto vertex = particle.vertex();
  if (vertex.chi2() < 0) {
    return false;
  }
  const bool decision = vertex.pt() > parameters.minComboPt && vertex.chi2() < parameters.maxVertexChi2 &&
                        particle.eta() > parameters.minEta && particle.eta() < parameters.maxEta &&
                        particle.doca12() < parameters.maxDOCA && particle.minpt() > parameters.minTrackPt &&
                        particle.minip() > parameters.minTrackIP &&
                        fabsf(particle.m12(Allen::mPi, Allen::mPi) - Allen::mDz) < parameters.massWindow &&
                        vertex.z() >= parameters.minZ && particle.pv().position.z >= parameters.minZ;

  return decision;
}
