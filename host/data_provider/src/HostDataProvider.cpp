/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "HostDataProvider.h"

void host_data_provider::host_data_provider_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const HostBuffers&) const
{
  auto bno = runtime_options.input_provider->banks(m_bank_type.get_value(), runtime_options.slice_index);
  // A number of spans for the blocks equal to the number of blocks
  set_size<host_raw_banks_t>(arguments, std::get<0>(bno).size());

  // A single span for the offsets
  set_size<host_raw_offsets_t>(arguments, 1);

  // A single number for the version
  set_size<host_raw_bank_version_t>(arguments, 1);
}

void host_data_provider::host_data_provider_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  HostBuffers&,
  const Allen::Context&) const
{
  auto bno = runtime_options.input_provider->banks(m_bank_type.get_value(), runtime_options.slice_index);

  // memcpy the offsets span directly
  auto const& offsets = std::get<2>(bno);
  ::memcpy(data<host_raw_offsets_t>(arguments), &offsets, sizeof(offsets));

  // Copy the spans for the blocks
  auto const& blocks = std::get<0>(bno);
  ::memcpy(
    data<host_raw_banks_t>(arguments),
    blocks.data(),
    blocks.size() * sizeof(typename std::remove_reference_t<decltype(blocks)>::value_type));

  // Copy the bank version
  auto version = std::get<3>(bno);
  ::memcpy(data<host_raw_bank_version_t>(arguments), &version, sizeof(version));
}
