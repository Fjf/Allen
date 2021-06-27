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
#include "TwoTrackCatBoostLine.cuh"

INSTANTIATE_LINE(two_track_catboost_line::two_track_catboost_line_t, two_track_catboost_line::Parameters)

__device__ unsigned two_track_catboost_line::two_track_catboost_line_t::offset(
  const Parameters& parameters,
  const unsigned event_number)
{
  return parameters.dev_sv_offsets[event_number];
}

unsigned two_track_catboost_line::two_track_catboost_line_t::get_decisions_size(
  ArgumentReferences<Parameters>& arguments)
{
  return first<typename Parameters::host_number_of_svs_t>(arguments);
}

__device__ std::tuple<const VertexFit::TrackMVAVertex&, const float>
two_track_catboost_line::two_track_catboost_line_t::get_input(
  const Parameters& parameters,
  const unsigned event_number,
  const unsigned i)
{
  const unsigned sv_index = i + parameters.dev_sv_offsets[event_number];
  return std::forward_as_tuple(parameters.dev_svs[sv_index], parameters.dev_two_track_evaluation[sv_index]);
}

__device__ bool two_track_catboost_line::two_track_catboost_line_t::select(
  const Parameters& parameters,
  std::tuple<const VertexFit::TrackMVAVertex&, const float> input)
{
  const auto& vertex = std::get<0>(input);
  const auto& response = std::get<1>(input);
  const bool decision =
    (vertex.minpt > parameters.minPt && vertex.eta > parameters.minEta && vertex.eta < parameters.maxEta &&
     vertex.mcor > parameters.minMcor && response > parameters.minMVA);
  return decision;
}