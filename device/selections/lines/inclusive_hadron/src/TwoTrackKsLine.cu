/*****************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "TwoTrackKsLine.cuh"

INSTANTIATE_LINE(two_track_line_ks::two_track_line_ks_t, two_track_line_ks::Parameters)

__device__ bool two_track_line_ks::two_track_line_ks_t::select(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input)
{
  const auto& particle = std::get<0>(input);
  const bool opposite_sign = particle.charge() == 0;

  if (opposite_sign != parameters.OppositeSign) return false;
  const auto& vertex = particle.vertex();
  if (vertex.chi2() < 0) {
    return false;
  }

  const auto trk1 = static_cast<const Allen::Views::Physics::BasicParticle*>(particle.child(0));
  const auto trk2 = static_cast<const Allen::Views::Physics::BasicParticle*>(particle.child(1));
  const auto& state1 = trk1->state();
  const auto& state2 = trk2->state();
  const float cos =
    (state1.px() * state2.px() + state1.py() * state2.py() + state1.pz() * state2.pz()) / (state1.p() * state2.p());
  const bool decision =
    vertex.chi2() < parameters.maxVertexChi2 && particle.eta() > parameters.minEta_Ks &&
    particle.eta() < parameters.maxEta_Ks && particle.has_pv() && particle.minipchi2() > parameters.minTrackIPChi2_Ks &&
    particle.m12(Allen::mPi, Allen::mPi) > parameters.minM_Ks &&
    particle.m12(Allen::mPi, Allen::mPi) < parameters.maxM_Ks && vertex.pt() > parameters.minComboPt_Ks &&
    cos > parameters.minCosOpening && particle.dira() > parameters.minCosDira &&
    particle.minp() > parameters.minTrackP_piKs && trk1->has_pv() && trk2->has_pv() &&
    trk1->ip() * trk2->ip() / particle.ip() > parameters.min_combip && particle.minpt() > parameters.minTrackPt_piKs &&
    vertex.z() >= parameters.minZ && particle.pv().position.z >= parameters.minZ;

  return decision;
}

__device__ void two_track_line_ks::two_track_line_ks_t::fill_tuples(
  const Parameters& parameters,
  std::tuple<const Allen::Views::Physics::CompositeParticle> input,
  unsigned index,
  bool sel)
{
  if (sel) {
    const auto& particle = std::get<0>(input);
    const auto trk1 = static_cast<const Allen::Views::Physics::BasicParticle*>(particle.child(0));
    const auto trk2 = static_cast<const Allen::Views::Physics::BasicParticle*>(particle.child(1));
    const auto& vertex = particle.vertex();
    parameters.pt_ks[index] = vertex.pt();
    parameters.eta_ks[index] = particle.eta();
    parameters.comb_ip[index] = trk1->ip() * trk2->ip() / particle.ip();
    parameters.min_pt[index] = particle.minpt();
    parameters.min_ipchi2[index] = particle.minipchi2();
    parameters.min_p[index] = particle.minp();
    parameters.mass[index] = particle.m12(Allen::mPi, Allen::mPi);
  }
}
