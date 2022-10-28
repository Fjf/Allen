/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "DiMuonMassLine.cuh"

INSTANTIATE_LINE(di_muon_mass_line::di_muon_mass_line_t, di_muon_mass_line::Parameters)

void di_muon_mass_line::di_muon_mass_line_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions& ro,
  const Constants& c,
  const HostBuffers& h) const
{
  set_size<typename Parameters::host_histogram_Jpsi_mass_t>(arguments, 100u);
  set_size<typename Parameters::dev_histogram_Jpsi_mass_t>(arguments, 100u);
}

__device__ bool di_muon_mass_line::di_muon_mass_line_t::select(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input)
{
  const auto vertex = std::get<0>(input);
  return vertex.is_dimuon() && vertex.minipchi2() >= parameters.minIPChi2 && vertex.doca12() <= parameters.maxDoca &&
         vertex.mdimu() >= parameters.minMass && vertex.minpt() >= parameters.minHighMassTrackPt &&
         vertex.minp() >= parameters.minHighMassTrackP && vertex.vertex().chi2() > 0 &&
         vertex.vertex().chi2() < parameters.maxVertexChi2 && vertex.vertex().z() >= parameters.minZ &&
         vertex.pv().position.z >= parameters.minZ;
}

void di_muon_mass_line::di_muon_mass_line_t::init_monitor(
  const ArgumentReferences<Parameters>& arguments,
  const Allen::Context& context)
{
  Allen::memset_async<dev_histogram_Jpsi_mass_t>(arguments, 0, context);
}

__device__ void di_muon_mass_line::di_muon_mass_line_t::monitor(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input,
  unsigned,
  bool sel)
{
  const auto particle = std::get<0>(input);
  const auto m = particle.m();
  if (sel) {
    if (m > parameters.histogram_Jpsi_mass_min && m < parameters.histogram_Jpsi_mass_max) {
      const unsigned int bin = static_cast<unsigned int>(
        (m - parameters.histogram_Jpsi_mass_min) * parameters.histogram_Jpsi_mass_nbins /
        (parameters.histogram_Jpsi_mass_max - parameters.histogram_Jpsi_mass_min));
      ++parameters.dev_histogram_Jpsi_mass[bin];
    }
  }
}
