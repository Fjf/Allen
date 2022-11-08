/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "BackendCommon.h"
#include "AlgorithmTypes.cuh"
#include "GenericContainerContracts.h"

#include "Gaudi/Accumulators.h"
#include "Gaudi/Accumulators/Histogram.h"

#include <LumiDefinitions.cuh>

#include "CaloDigit.cuh"

namespace calo_lumi_counters {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_lumi_summaries_size_t, unsigned) host_lumi_summaries_size;
    DEVICE_INPUT(dev_lumi_summary_offsets_t, unsigned) dev_lumi_summary_offsets;
    DEVICE_INPUT(dev_ecal_digits_t, CaloDigit) dev_ecal_digits;
    DEVICE_INPUT(dev_ecal_digits_offsets_t, unsigned) dev_ecal_digits_offsets;
    DEVICE_OUTPUT(dev_energies_t, float) dev_energies;
    DEVICE_OUTPUT(dev_lumi_infos_t, Lumi::LumiInfo) dev_lumi_infos;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
    PROPERTY(monitoring_t, "enable_monitoring", "enable monitoring", bool) monitoring;
  }; // struct Parameters

  __global__ void calo_lumi_counters(Parameters, const unsigned number_of_events, const char* raw_ecal_geometry);

  struct calo_lumi_counters_t : public DeviceAlgorithm, Parameters {

    void init();

    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      const Allen::Context& context) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{64, 1, 1}}};
    Property<monitoring_t> m_monitoring {this, false};

    std::unique_ptr<Gaudi::Accumulators::Histogram<1, Gaudi::Accumulators::atomicity::full, float>> m_histos_sum_et;
    std::array<std::unique_ptr<Gaudi::Accumulators::Histogram<1, Gaudi::Accumulators::atomicity::full, float>>, 6>
      m_histos_energy;
    std::array<std::unique_ptr<Gaudi::Accumulators::Histogram<1, Gaudi::Accumulators::atomicity::full, float>>, 6>
      m_histos_et;
    std::array<std::unique_ptr<Gaudi::Accumulators::Histogram<1, Gaudi::Accumulators::atomicity::full, float>>, 3>
      m_histos_energy_diff;
    std::array<std::unique_ptr<Gaudi::Accumulators::Histogram<1, Gaudi::Accumulators::atomicity::full, float>>, 3>
      m_histos_et_diff;
  }; // struct calo_lumi_counters_t
} // namespace calo_lumi_counters
