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
#include "D2KPiLine.cuh"

#ifndef ALLEN_STANDALONE
#include "Gaudi/Accumulators/Histogram.h"

template<int I>
using gaudi_histo_t = Gaudi::Accumulators::Histogram<I, Gaudi::Accumulators::atomicity::full, double>;
#endif

void d2kpi_line::d2kpi_line_t::init()
{
  Line<d2kpi_line::d2kpi_line_t, d2kpi_line::Parameters>::init();
#ifndef ALLEN_STANDALONE
  histogram_d0_mass = (void*) new gaudi_histo_t<1>(
    this,
    "d0_mass",
    "m(D0)",
    Gaudi::Accumulators::Axis<double> {
      property<histogram_d0_mass_nbins_t>(), property<histogram_d0_mass_min_t>(), property<histogram_d0_mass_max_t>()});
  histogram_d0_pt = (void*) new gaudi_histo_t<1>(
    this,
    "d0_pt",
    "pT(D0)",
    Gaudi::Accumulators::Axis<double> {
      property<histogram_d0_pt_nbins_t>(), property<histogram_d0_pt_min_t>(), property<histogram_d0_pt_max_t>()});
#endif
}

void d2kpi_line::d2kpi_line_t::output_monitor(
  [[maybe_unused]] const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  [[maybe_unused]] const Allen::Context& context) const
{
#ifndef ALLEN_STANDALONE
  Allen::copy_async<host_histogram_d0_mass_t, dev_histogram_d0_mass_t>(arguments, context);
  Allen::copy_async<host_histogram_d0_pt_t, dev_histogram_d0_pt_t>(arguments, context);
  Allen::synchronize(context);

  float binWidth =
    (property<histogram_d0_mass_max_t>() - property<histogram_d0_mass_min_t>()) / property<histogram_d0_mass_nbins_t>();
  auto* histogram_d0_mass_p = reinterpret_cast<gaudi_histo_t<1>*>(histogram_d0_mass);
  auto mass_buffer = histogram_d0_mass_p->buffer();
  for (auto i = 0u; i < property<histogram_d0_mass_nbins_t>(); ++i) {
    mass_buffer[property<histogram_d0_mass_min_t>() + i * binWidth] += *(data<host_histogram_d0_mass_t>(arguments) + i);
  }
  binWidth =
    (property<histogram_d0_pt_max_t>() - property<histogram_d0_pt_min_t>()) / property<histogram_d0_pt_nbins_t>();
  auto* histogram_d0_pt_p = reinterpret_cast<gaudi_histo_t<1>*>(histogram_d0_pt);
  auto pt_buffer = histogram_d0_pt_p->buffer();
  for (auto i = 0u; i < property<histogram_d0_pt_nbins_t>(); ++i) {
    pt_buffer[property<histogram_d0_pt_min_t>() + i * binWidth] += *(data<host_histogram_d0_pt_t>(arguments) + i);
  }
#endif
}
