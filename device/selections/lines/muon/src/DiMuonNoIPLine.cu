/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "DiMuonNoIPLine.cuh"
#include <ROOTHeaders.h>
#include "ROOTService.h"
#include <array>
#include "BinarySearch.cuh"

namespace {
  const unsigned n_bins = 10389u;
}

INSTANTIATE_LINE(di_muon_no_ip_line::di_muon_no_ip_line_t, di_muon_no_ip_line::Parameters)

void di_muon_no_ip_line::di_muon_no_ip_line_t::init()
{
#ifndef ALLEN_STANDALONE
  float start_q = 0;
  float stop_q = 110e3;

  histogram_prompt_q =
    new gaudi_monitoring::Lockable_Histogram<> {{this, "dimuon_q", "dimuon q", {10390, start_q, stop_q}}, {}};
#endif
}

float bin_size(float q)
{
  float a = 9.164e-9;
  float b = 1.488e-4;
  float c = 0.1208;
  return a * q * q + b * q + c;
}

void di_muon_no_ip_line::di_muon_no_ip_line_t::init_monitor(
  [[maybe_unused]] const ArgumentReferences<Parameters>& arguments,
  [[maybe_unused]] const Allen::Context& context)
{
#ifndef ALLEN_STANDALONE
  Allen::memset_async<dev_array_prompt_q_t>(arguments, 0, context);
  auto boundaries = make_host_buffer<float>(arguments, n_bins + 1);
  boundaries[0] = 0.f;

  for (unsigned i = 0; i < n_bins; i++) {
    float last_bound = boundaries[i];
    float increment = bin_size(last_bound);
    boundaries[i + 1] = (last_bound + increment * 2);
  }

  Allen::copy(arguments.template get<dev_q_bin_boundaries_t>(), boundaries.get(), context, Allen::memcpyHostToDevice);
#endif
}

void di_muon_no_ip_line::di_muon_no_ip_line_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions& ro,
  const Constants& c) const
{
  static_cast<Line const*>(this)->set_arguments_size(arguments, ro, c);
  set_size<dev_q_bin_boundaries_t>(arguments, n_bins + 1);
  set_size<dev_array_prompt_q_t>(arguments, n_bins);
}

__device__ bool di_muon_no_ip_line::di_muon_no_ip_line_t::select(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input)
{
  const auto vertex = std::get<0>(input);
  const auto track1 = static_cast<const Allen::Views::Physics::BasicParticle*>(vertex.child(0));
  const auto track2 = static_cast<const Allen::Views::Physics::BasicParticle*>(vertex.child(1));

  const bool same_sign = vertex.charge() != 0;

  return vertex.is_dimuon() && (same_sign == parameters.ss_on) &&
         track1->state().chi2() / track1->state().ndof() <= parameters.maxTrChi2 &&
         track2->state().chi2() / track2->state().ndof() <= parameters.maxTrChi2 && track1->state().chi2() > 0 &&
         track2->state().chi2() > 0 && vertex.doca12() <= parameters.maxDoca &&
         track1->state().pt() * track2->state().pt() >= parameters.minTrackPtPROD &&
         track1->state().p() >= parameters.minTrackP && track2->state().p() >= parameters.minTrackP &&
         vertex.vertex().chi2() > 0 && vertex.vertex().chi2() <= parameters.maxVertexChi2 &&
         vertex.vertex().pt() > parameters.minPt && vertex.vertex().z() >= parameters.minZ;
}

__device__ void di_muon_no_ip_line::di_muon_no_ip_line_t::monitor(
  [[maybe_unused]] const Parameters& parameters,
  [[maybe_unused]] std::tuple<const Allen::Views::Physics::CompositeParticle> input,
  unsigned,
  bool sel)
{
  if (sel) {
#ifndef ALLEN_STANDALONE
    const auto vertex = std::get<0>(input);
    const auto track1 = static_cast<const Allen::Views::Physics::BasicParticle*>(vertex.child(0));
    const auto track2 = static_cast<const Allen::Views::Physics::BasicParticle*>(vertex.child(1));
    if (track1->ip_chi2() < 6 && track2->ip_chi2() < 6) {
      float q = sqrtf(vertex.m() * vertex.m() - 4 * Allen::mMu * Allen::mMu);
      if (q < parameters.dev_q_bin_boundaries[n_bins]) {
        unsigned bin = binary_search_rightmost(&parameters.dev_q_bin_boundaries[0], n_bins, q);
        parameters.dev_array_prompt_q[bin]++;
      }
    }
#endif
  }
}

void di_muon_no_ip_line::di_muon_no_ip_line_t::output_monitor(
  [[maybe_unused]] const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  [[maybe_unused]] const Allen::Context& context) const
{
#ifndef ALLEN_STANDALONE
  gaudi_monitoring::fill(
    arguments, context, std::tuple {get<dev_array_prompt_q_t>(arguments), histogram_prompt_q, 0, 110e3});
#endif
}
