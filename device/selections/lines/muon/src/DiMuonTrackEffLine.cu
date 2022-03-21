/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "DiMuonTrackEffLine.cuh"

INSTANTIATE_LINE(di_muon_track_eff_line::di_muon_track_eff_line_t, di_muon_track_eff_line::Parameters)

__device__ bool di_muon_track_eff_line::di_muon_track_eff_line_t::select(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle&> input)
{
  const auto vertex = std::get<0>(input);
  if (!vertex.is_dimuon()) return false;
  const bool decision =
    vertex.vertex().chi2() > 0 && vertex.mdimu() > parameters.DMTrackEffM0 && vertex.mdimu() < parameters.DMTrackEffM1;
  return decision;
}
