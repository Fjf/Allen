/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "D2KPiLine.cuh"

INSTANTIATE_LINE(d2kpi_line::d2kpi_line_t, d2kpi_line::Parameters)

void d2kpi_line::d2kpi_line_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions& ro,
  const Constants& c,
  const HostBuffers& h) const
{
  Line<d2kpi_line::d2kpi_line_t, d2kpi_line::Parameters>::set_arguments_size(arguments, ro, c, h);
  set_size<typename Parameters::host_histogram_d0_mass_t>(arguments, 100u);
  set_size<typename Parameters::host_histogram_d0_pt_t>(arguments, 100u);
  set_size<typename Parameters::dev_histogram_d0_mass_t>(arguments, 100u);
  set_size<typename Parameters::dev_histogram_d0_pt_t>(arguments, 100u);
}

__device__ bool d2kpi_line::d2kpi_line_t::select(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input)
{
  const auto particle = std::get<0>(input);
  const auto vertex = particle.vertex();
  if (vertex.chi2() < 0) {
    return false;
  }
  const float m1 = particle.m12(Allen::mK, Allen::mPi);
  const float m2 = particle.m12(Allen::mPi, Allen::mK);
  const bool decision = vertex.pt() > parameters.minComboPt && vertex.chi2() < parameters.maxVertexChi2 &&
                        particle.eta() > parameters.minEta && particle.eta() < parameters.maxEta &&
                        particle.doca12() < parameters.maxDOCA && particle.minpt() > parameters.minTrackPt &&
                        particle.minip() > parameters.minTrackIP &&
                        min(fabsf(m1 - Allen::mDz), fabsf(m2 - Allen::mDz)) < parameters.massWindow &&
                        vertex.z() >= parameters.minZ && particle.pv().position.z >= parameters.minZ;

  return decision;
}

void d2kpi_line::d2kpi_line_t::init_monitor(
  const ArgumentReferences<Parameters>& arguments,
  const Allen::Context& context)
{
  Allen::memset_async<dev_histogram_d0_mass_t>(arguments, 0, context);
  Allen::memset_async<dev_histogram_d0_pt_t>(arguments, 0, context);
}

__device__ void d2kpi_line::d2kpi_line_t::monitor(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input,
  unsigned,
  bool sel)
{
  const auto particle = std::get<0>(input);
  const auto vertex = particle.vertex();
  if (sel) {
    const float m1 = particle.m12(Allen::mK, Allen::mPi);
    const float m2 = particle.m12(Allen::mPi, Allen::mK);
    const float pt = vertex.pt();
    if (m1 > parameters.histogram_d0_mass_min && m1 < parameters.histogram_d0_mass_max) {
      const unsigned int bin = static_cast<unsigned int>(
        (m1 - parameters.histogram_d0_mass_min) * parameters.histogram_d0_mass_nbins /
        (parameters.histogram_d0_mass_max - parameters.histogram_d0_mass_min));
      ++parameters.dev_histogram_d0_mass[bin];
    }
    if (m2 > parameters.histogram_d0_mass_min && m2 < parameters.histogram_d0_mass_max) {
      const unsigned int bin = static_cast<unsigned int>(
        (m2 - parameters.histogram_d0_mass_min) * parameters.histogram_d0_mass_nbins /
        (parameters.histogram_d0_mass_max - parameters.histogram_d0_mass_min));
      ++parameters.dev_histogram_d0_mass[bin];
    }
    if (pt > parameters.histogram_d0_pt_min && pt < parameters.histogram_d0_pt_max) {
      const unsigned int bin = static_cast<unsigned int>(
        (pt - parameters.histogram_d0_pt_min) * parameters.histogram_d0_pt_nbins /
        (parameters.histogram_d0_pt_max - parameters.histogram_d0_pt_min));
      ++parameters.dev_histogram_d0_pt[bin];
    }
  }
}
