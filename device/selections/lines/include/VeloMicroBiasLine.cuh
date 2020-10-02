#pragma once

#include "SelectionAlgorithm.cuh"
#include "EventLine.cuh"
#include "VeloConsolidated.cuh"

namespace velo_micro_bias_line {
  DEFINE_PARAMETERS(
    Parameters,
    (HOST_INPUT(host_number_of_events_t, unsigned), host_number_of_events),
    (DEVICE_INPUT(dev_number_of_events_t, unsigned), dev_number_of_events),
    (DEVICE_INPUT(dev_event_list_t, unsigned), dev_event_list),
    (DEVICE_INPUT(dev_offsets_velo_tracks_t, unsigned), dev_offsets_velo_tracks),
    (DEVICE_INPUT(dev_offsets_velo_track_hit_number_t, unsigned), dev_offsets_velo_track_hit_number),
    (DEVICE_OUTPUT(dev_decisions_t, bool), dev_decisions),
    (DEVICE_OUTPUT(dev_decisions_offsets_t, unsigned), dev_decisions_offsets),
    (DEVICE_OUTPUT(dev_post_scaler_t, float), dev_post_scaler),
    (PROPERTY(pre_scaler_t, "pre_scaler", "Selection pre-scaling factor", float), pre_scaler),
    (PROPERTY(post_scaler_t, "post_scaler", "Selection pre-scaling factor", float), post_scaler),
    (PROPERTY(min_velo_tracks_t, "min_velo_tracks", "Minimum number of VELO tracks", unsigned), min_velo_tracks))

  struct velo_micro_bias_line_t : public SelectionAlgorithm, Parameters, EventLine<velo_micro_bias_line_t, Parameters> {
    __device__ std::tuple<const unsigned>
    get_input(const Parameters& parameters, const unsigned event_number) const;

    __device__ bool select(const Parameters& parameters, std::tuple<const unsigned> input) const;

  private:
    Property<pre_scaler_t> m_pre_scaler {this, 1.f};
    Property<post_scaler_t> m_post_scaler {this, 1.f};
    Property<min_velo_tracks_t> m_min_velo_tracks {this, 1};
  };
} // namespace velo_micro_bias_line