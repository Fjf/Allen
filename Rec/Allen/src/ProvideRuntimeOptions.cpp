/***************************************************************************** \
 * (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
 *                                                                             *
 * This software is distributed under the terms of the GNU General Public      *
 * Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
 *                                                                             *
 * In applying this licence, CERN does not waive the privileges and immunities *
 * granted to it by virtue of its status as an Intergovernmental Organization  *
 * or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include <array>

// Gaudi
#include <GaudiAlg/Transformer.h>
#include <GaudiAlg/FunctionalUtilities.h>
#include <Event/RawBank.h>
#include <RuntimeOptions.h>
#include "AllenROOTService.h"

// Allen
#include <TESProvider.h>
#include <Constants.cuh>
#include <Logger.h>

using Gaudi::Functional::Traits::useLegacyGaudiAlgorithm;

class ProvideRuntimeOptions final : public Gaudi::Functional::Transformer<
                                      RuntimeOptions(std::array<TransposedBanks, LHCb::RawBank::LastType> const&),
                                      useLegacyGaudiAlgorithm> {

public:
  /// Standard constructor
  ProvideRuntimeOptions(const std::string& name, ISvcLocator* pSvcLocator);

  StatusCode initialize() override;

  /// Algorithm execution
  RuntimeOptions operator()(std::array<TransposedBanks, LHCb::RawBank::LastType> const& allen_banks) const override;

private:
  SmartIF<AllenROOTService> m_rootService;
};

ProvideRuntimeOptions::ProvideRuntimeOptions(const std::string& name, ISvcLocator* pSvcLocator) :
  Transformer(
    name,
    pSvcLocator,
    // Inputs
    {KeyValue {"AllenBanksLocation", "Allen/Raw/Banks"}},
    // Output
    KeyValue {"RuntimeOptionsLocation", "Allen/Stream/RuntimeOptions"})
{}

StatusCode ProvideRuntimeOptions::initialize()
{
  return Transformer::initialize().andThen([&] {
    m_rootService = svc<AllenROOTService>("AllenROOTService", true);
    return m_rootService.isValid();
  });
}

RuntimeOptions ProvideRuntimeOptions::operator()(
  std::array<TransposedBanks, LHCb::RawBank::LastType> const& allen_banks) const
{
  const unsigned number_of_repetitions = 1;
  const bool param_inject_mem_fail = false;
  const size_t n_slices = 1;
  const size_t events_per_slice = 1;
  const unsigned event_start = 0;
  const unsigned event_end = 1;
  auto tes_provider = std::make_shared<TESProvider>(n_slices, events_per_slice, event_end - event_start);
  tes_provider->set_banks(allen_banks);

  // initialize RuntimeOptions
  const size_t slice_index = 0;
  const bool mep_layout = false;

  return {tes_provider,
          slice_index,
          {event_start, event_end},
          number_of_repetitions,
          mep_layout,
          param_inject_mem_fail,
          nullptr,
          m_rootService->rootService()};
}

DECLARE_COMPONENT(ProvideRuntimeOptions)
