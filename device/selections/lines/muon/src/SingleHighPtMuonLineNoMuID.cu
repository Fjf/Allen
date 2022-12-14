/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "SingleHighPtMuonLineNoMuID.cuh"

// Explicit instantiation
INSTANTIATE_LINE(
  single_high_pt_muon_no_muid_line::single_high_pt_muon_no_muid_line_t,
  single_high_pt_muon_no_muid_line::Parameters)

__device__ bool single_high_pt_muon_no_muid_line::single_high_pt_muon_no_muid_line_t::select(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::BasicParticle> input)
{
  const auto& track = std::get<0>(input);
  const bool decision = track.state().chi2() / track.state().ndof() < parameters.maxChi2Ndof &&
                        track.state().pt() > parameters.singleMinPt && track.state().p() > parameters.singleMinP &&
                        track.state().z() > parameters.minZ;

  return decision;
}
