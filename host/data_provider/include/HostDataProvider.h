/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "Common.h"
#include "AlgorithmTypes.cuh"
#include "InputProvider.h"
#include <gsl/span>

namespace host_data_provider {
  struct Parameters {
    HOST_OUTPUT(host_raw_banks_t, gsl::span<char const>) host_raw_banks;
    HOST_OUTPUT(host_raw_offsets_t, gsl::span<unsigned int const>) host_raw_offsets;
    HOST_OUTPUT(host_raw_sizes_t, gsl::span<unsigned int const>) host_raw_sizes;
    HOST_OUTPUT(host_raw_types_t, gsl::span<unsigned int const>) host_raw_types;
    HOST_OUTPUT(host_raw_bank_version_t, int) host_raw_bank_version;
    PROPERTY(raw_bank_type_t, "bank_type", "type of raw bank to provide", BankTypes) prop_raw_bank_type;
    PROPERTY(empty_t, "empty", "will provide empty banks", bool) empty;
  };

  struct host_data_provider_t : public ProviderAlgorithm, Parameters {
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
} // namespace host_data_provider
