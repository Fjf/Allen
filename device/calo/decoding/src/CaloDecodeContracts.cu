/*****************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include <BackendCommon.h>
#include <CaloDigit.cuh>
#include <CaloDecode.cuh>
#include <unordered_set>
#include <climits>

bool check_digits(CaloDigit const* digits, size_t n_digits)
{
  bool valid = true;
  for (size_t i = 0; i < n_digits; ++i) {
    valid &= digits[i].adc < SHRT_MAX;
  }
  return valid;
}

void calo_decode::check_digits::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  const auto event_list = make_host_buffer<Parameters::dev_event_list_t>(arguments, context);

  const auto ecal_digits = make_host_buffer<Parameters::dev_ecal_digits_t>(arguments, context);
  const auto ecal_offsets = make_host_buffer<Parameters::dev_ecal_digits_offsets_t>(arguments, context);

  bool ecal_valid = true;

  CaloGeometry ecal_geom {constants.host_ecal_geometry.data()};

  for (auto event : event_list) {
    ecal_valid &= ::check_digits(ecal_digits.data() + ecal_offsets[event], ecal_geom.max_index);
  }

  require(ecal_valid, "Require that all ECal digits are present with a reasonable ADC");
}
