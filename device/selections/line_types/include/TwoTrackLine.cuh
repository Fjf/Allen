#pragma once

#include "Line.cuh"
#include "VertexDefinitions.cuh"

/**
 * A OneTrackLine.
 *
 * It assumes an inheriting class will have the following inputs:
 *  (HOST_INPUT(host_number_of_events_t, unsigned), host_number_of_events),
 *  (HOST_INPUT(host_number_of_svs_t, unsigned), host_number_of_svs),
 *  (DEVICE_INPUT(dev_svs_t, VertexFit::TrackMVAVertex), dev_svs),
 *  (DEVICE_INPUT(dev_sv_offsets_t, unsigned), dev_sv_offsets),
 *  (DEVICE_OUTPUT(decisions_t, bool), decisions),
 *
 * It also assumes the OneTrackLine will be defined as:
 *  __device__ bool select(const Parameters& parameters, std::tuple<const VertexFit::TrackMVAVertex&> input) const;
 */
template<typename Derived, typename Parameters>
struct TwoTrackLine : public Line<Derived, Parameters> {
  inline void set_decisions_size(ArgumentReferences<Parameters>& arguments) const
  {
    set_size<typename Parameters::dev_decisions_t>(
      arguments, first<typename Parameters::host_number_of_svs_t>(arguments));
  }

  __device__ inline unsigned get_input_size(const Parameters& parameters, const unsigned event_number) const
  {
    const auto number_of_vertices_event =
      parameters.dev_sv_offsets[event_number + 1] - parameters.dev_sv_offsets[event_number];
    return number_of_vertices_event;
  }

  __device__ inline bool* get_decision(const Parameters& parameters, const unsigned event_number) const
  {
    return parameters.dev_decisions + parameters.dev_sv_offsets[event_number];
  }

  __device__ inline std::tuple<const VertexFit::TrackMVAVertex&>
  get_input(const Parameters& parameters, const unsigned event_number, const unsigned i) const
  {
    const VertexFit::TrackMVAVertex* event_vertices = parameters.dev_svs + parameters.dev_sv_offsets[event_number];
    const auto& vertex = event_vertices[i];
    return std::forward_as_tuple(vertex);
  }
};
