/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "SelectionAlgorithm.cuh"
#include "OneTrackLine.cuh"

namespace track_mva_line {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_scifi_tracks_t, unsigned) host_number_of_reconstructed_scifi_tracks;
    DEVICE_INPUT(dev_tracks_t, ParKalmanFilter::FittedTrack) dev_tracks;
    DEVICE_INPUT(dev_track_offsets_t, unsigned) dev_track_offsets;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_odin_raw_input_t, char) dev_odin_raw_input;
    DEVICE_INPUT(dev_odin_raw_input_offsets_t, unsigned) dev_odin_raw_input_offsets;
    DEVICE_INPUT(dev_mep_layout_t, unsigned) dev_mep_layout;
    DEVICE_OUTPUT(dev_decisions_t, bool) dev_decisions;
    DEVICE_OUTPUT(dev_decisions_offsets_t, unsigned) dev_decisions_offsets;
    HOST_OUTPUT(host_post_scaler_t, float) host_post_scaler;
    HOST_OUTPUT(host_post_scaler_hash_t, uint32_t) host_post_scaler_hash;
    PROPERTY(pre_scaler_t, "pre_scaler", "Pre-scaling factor", float) pre_scaler;
    PROPERTY(post_scaler_t, "post_scaler", "Post-scaling factor", float) post_scaler;
    PROPERTY(pre_scaler_hash_string_t, "pre_scaler_hash_string", "Pre-scaling hash string", std::string);
    PROPERTY(post_scaler_hash_string_t, "post_scaler_hash_string", "Post-scaling hash string", std::string);
    PROPERTY(maxChi2Ndof_t, "maxChi2Ndof", "maxChi2Ndof description", float) maxChi2Ndof;
    PROPERTY(minPt_t, "minPt", "minPt description", float) minPt;
    PROPERTY(maxPt_t, "maxPt", "maxPt description", float) maxPt;
    PROPERTY(minIPChi2_t, "minIPChi2", "minIPChi2 description", float) minIPChi2;
    PROPERTY(param1_t, "param1", "param1 description", float) param1;
    PROPERTY(param2_t, "param2", "param2 description", float) param2;
    PROPERTY(param3_t, "param3", "param3 description", float) param3;
    PROPERTY(alpha_t, "alpha", "alpha description", float) alpha;
  };

  struct track_mva_line_t : public SelectionAlgorithm, Parameters, OneTrackLine<track_mva_line_t, Parameters> {
    __device__ static bool select(const Parameters& ps, std::tuple<const ParKalmanFilter::FittedTrack&> input);

  private:
    Property<pre_scaler_t> m_pre_scaler {this, 1.f};
    Property<post_scaler_t> m_post_scaler {this, 1.f};
    Property<pre_scaler_hash_string_t> m_pre_scaler_hash_string {this, ""};
    Property<post_scaler_hash_string_t> m_post_scaler_hash_string {this, ""};
    Property<maxChi2Ndof_t> m_maxChi2Ndof {this, 2.5f};
    Property<minPt_t> m_minPt {this, 2000.0f / Gaudi::Units::GeV};
    Property<maxPt_t> m_maxPt {this, 26000.0f / Gaudi::Units::GeV};
    Property<minIPChi2_t> m_minIPChi2 {this, 7.4f};
    Property<param1_t> m_param1 {this, 1.0f};
    Property<param2_t> m_param2 {this, 2.0f};
    Property<param3_t> m_param3 {this, 1.248f};
    Property<alpha_t> m_alpha {this, 0.f};
  };
} // namespace track_mva_line