/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "Common.h"
#include "AlgorithmTypes.cuh"
#include "InputProvider.h"

namespace data_provider {
  struct Parameters {
    DEVICE_OUTPUT(dev_raw_banks_t, char) dev_raw_banks;
    DEVICE_OUTPUT(dev_raw_offsets_t, unsigned) dev_raw_offsets;
    DEVICE_OUTPUT(dev_raw_sizes_t, unsigned) dev_raw_sizes;
    DEVICE_OUTPUT(dev_raw_types_t, unsigned) dev_raw_types;
    HOST_OUTPUT(host_raw_bank_version_t, int) host_raw_bank_version;
    PROPERTY(raw_bank_type_t, "bank_type", "type of raw bank to provide", BankTypes) prop_raw_bank_type;
    PROPERTY(empty_t, "empty", "will provide empty banks", bool) empty;
  };

  struct data_provider_t : public ProviderAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions& runtime_options,
      const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Property<raw_bank_type_t> m_bank_type {this, BankTypes::ODIN};
    Property<empty_t> m_empty {this, false};
  };
} // namespace data_provider
