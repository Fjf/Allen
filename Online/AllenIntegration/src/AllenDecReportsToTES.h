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
#ifndef ALLENDECREPORTSTOTES_H
#define ALLENDECREPORTSTOTES_H

// Gaudi
#include "GaudiAlg/Transformer.h"
#include "GaudiKernel/StdArrayAsProperty.h"

// LHCb

// Allen
#include "HostBuffers.cuh"
#include "Logger.h"
#include "RawBanksDefinitions.cuh"

class AllenDecReportsToTES final : public Gaudi::Functional::Transformer<std::vector<uint32_t>(const HostBuffers&)> {
public:
  /// Standard constructor
  AllenDecReportsToTES(const std::string& name, ISvcLocator* pSvcLocator);

  /// initialization
  StatusCode initialize() override;

  /// Algorithm execution
  std::vector<uint32_t> operator()(const HostBuffers&) const override;

private:
};

#endif