/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "DiMuonSoftLine.cuh"

INSTANTIATE_LINE(di_muon_soft_line::di_muon_soft_line_t, di_muon_soft_line::Parameters)

__device__ bool di_muon_soft_line::di_muon_soft_line_t::select(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input)
{
  const auto vertex = std::get<0>(input);

  if (!vertex.is_dimuon()) return false;
  if (vertex.minipchi2() < parameters.DMSoftMinIPChi2) return false;

  // KS pipi misid veto
  const bool decision =
    vertex.vertex().chi2() > 0 && (vertex.mdimu() < parameters.DMSoftM0 || vertex.mdimu() > parameters.DMSoftM1) &&
    (vertex.mdimu() < parameters.DMSoftM2) && vertex.eta() > 0 &&
    (vertex.vertex().x() * vertex.vertex().x() + vertex.vertex().y() * vertex.vertex().y()) >
      parameters.DMSoftMinRho2 &&
    (vertex.vertex().z() > parameters.DMSoftMinZ) && (vertex.vertex().z() < parameters.DMSoftMaxZ) &&
    vertex.doca12() < parameters.DMSoftMaxDOCA && vertex.ip() / vertex.dz() < parameters.DMSoftMaxIPDZ &&
    vertex.clone_sin2() > parameters.DMSoftGhost;
  return decision;
}
