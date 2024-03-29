/*****************************************************************************\
* (c) Copyright 2023 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "SMOG2_KsToPiPi.cuh"
#include <ROOTHeaders.h>
#include "ROOTService.h"

INSTANTIATE_LINE(SMOG2_kstopipi_line::SMOG2_kstopipi_line_t, SMOG2_kstopipi_line::Parameters)

__device__ bool SMOG2_kstopipi_line::SMOG2_kstopipi_line_t::select(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input)
{
  const auto vertex = std::get<0>(input);
  return vertex.has_pv() && vertex.pv().position.z >= parameters.minPVZ && vertex.pv().position.z < parameters.maxPVZ &&
         vertex.minipchi2() > parameters.minIPChi2 && vertex.charge() == parameters.CombCharge &&
         vertex.vertex().chi2() < parameters.maxVertexChi2 && vertex.ip() < parameters.maxIP &&
         vertex.m12(Allen::mPi, Allen::mPi) >= parameters.minMass &&
         vertex.m12(Allen::mPi, Allen::mPi) < parameters.maxMass && vertex.vertex().z() >= parameters.minPVZ;
}

void SMOG2_kstopipi_line::SMOG2_kstopipi_line_t::init()
{
  Line<SMOG2_kstopipi_line::SMOG2_kstopipi_line_t, SMOG2_kstopipi_line::Parameters>::init();
#ifndef ALLEN_STANDALONE
  histogram_smogks_mass = new gaudi_monitoring::Lockable_Histogram<> {{this,
                                                                       "ks_mass",
                                                                       "M(Ks) [MeV]",
                                                                       {property<histogram_smogks_mass_nbins_t>(),
                                                                        property<histogram_smogks_mass_min_t>(),
                                                                        property<histogram_smogks_mass_max_t>()}},
                                                                      {}};

  histogram_smogks_svz = new gaudi_monitoring::Lockable_Histogram<> {{this,
                                                                      "smogks_svz",
                                                                      "SV_z (Ks)",
                                                                      {property<histogram_smogks_svz_nbins_t>(),
                                                                       property<histogram_smogks_svz_min_t>(),
                                                                       property<histogram_smogks_svz_max_t>()}},
                                                                     {}};
#endif
}

void SMOG2_kstopipi_line::SMOG2_kstopipi_line_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions& ro,
  const Constants& c) const
{
  static_cast<Line const*>(this)->set_arguments_size(arguments, ro, c);
  set_size<typename Parameters::dev_histogram_smogks_mass_t>(arguments, 100u);
  set_size<typename Parameters::dev_histogram_smogks_svz_t>(arguments, 100u);
}

void SMOG2_kstopipi_line::SMOG2_kstopipi_line_t::output_monitor(
  [[maybe_unused]] const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  [[maybe_unused]] const Allen::Context& context) const
{
#ifndef ALLEN_STANDALONE
  gaudi_monitoring::fill(
    arguments,
    context,
    std::tuple {std::tuple {get<dev_histogram_smogks_mass_t>(arguments),
                            histogram_smogks_mass,
                            property<histogram_smogks_mass_min_t>(),
                            property<histogram_smogks_mass_max_t>()},
                std::tuple {get<dev_histogram_smogks_svz_t>(arguments),
                            histogram_smogks_svz,
                            property<histogram_smogks_svz_min_t>(),
                            property<histogram_smogks_svz_max_t>()}});
#endif
}

void SMOG2_kstopipi_line::SMOG2_kstopipi_line_t::init_monitor(
  const ArgumentReferences<Parameters>& arguments,
  const Allen::Context& context)
{
  Allen::memset_async<dev_histogram_smogks_mass_t>(arguments, 0, context);
  Allen::memset_async<dev_histogram_smogks_svz_t>(arguments, 0, context);
}

__device__ void SMOG2_kstopipi_line::SMOG2_kstopipi_line_t::monitor(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input,
  unsigned index,
  bool sel)
{
  const auto smogks = std::get<0>(input);
  if (sel) {
    parameters.sv_masses[index] = smogks.m12(Allen::mPi, Allen::mPi);
    parameters.svz[index] = smogks.vertex().z();

    const float svz = smogks.vertex().z();
    const float m = smogks.m12(Allen::mPi, Allen::mPi);
    if (m > parameters.histogram_smogks_mass_min && m < parameters.histogram_smogks_mass_max) {
      const unsigned int bin = static_cast<unsigned int>(
        (m - parameters.histogram_smogks_mass_min) * parameters.histogram_smogks_mass_nbins /
        (parameters.histogram_smogks_mass_max - parameters.histogram_smogks_mass_min));
      ++parameters.dev_histogram_smogks_mass[bin];
    }
    if (svz > parameters.histogram_smogks_svz_min && svz < parameters.histogram_smogks_svz_max) {
      const unsigned int bin = static_cast<unsigned int>(
        (svz - parameters.histogram_smogks_svz_min) * parameters.histogram_smogks_svz_nbins /
        (parameters.histogram_smogks_svz_max - parameters.histogram_smogks_svz_min));
      ++parameters.dev_histogram_smogks_svz[bin];
    }
  }
}
