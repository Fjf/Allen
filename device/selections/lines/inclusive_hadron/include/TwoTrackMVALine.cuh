/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "SelectionAlgorithm.cuh"
#include "TwoTrackLine.cuh"

namespace two_track_mva_line {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_svs_t, unsigned) host_number_of_svs;
    DEVICE_INPUT(dev_svs_t, VertexFit::TrackMVAVertex) dev_svs;
    DEVICE_INPUT(dev_sv_offsets_t, unsigned) dev_sv_offsets;
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
    PROPERTY(minComboPt_t, "minComboPt", "minComboPt description", float) minComboPt;
    PROPERTY(maxVertexChi2_t, "maxVertexChi2", "maxVertexChi2 description", float) maxVertexChi2;
    PROPERTY(minMCor_t, "minMCor", "minMCor description", float) minMCor;
    PROPERTY(minEta_t, "minEta", "minEta description", float) minEta;
    PROPERTY(maxEta_t, "maxEta", "maxEta description", float) maxEta;
    PROPERTY(minTrackPt_t, "minTrackPt", "minTrackPt description", float) minTrackPt;
    PROPERTY(maxNTrksAssoc_t, "maxNTrksAssoc", "maxNTrksAssoc description", int)
    maxNTrksAssoc; // Placeholder. To be replaced with MVA selection.
    PROPERTY(minFDChi2_t, "minFDChi2", "minFDChi2 description", float)
    minFDChi2; // Placeholder. To be replaced with MVA selection.
    PROPERTY(minTrackIPChi2_t, "minTrackIPChi2", "minTrackIPChi2 description", float) minTrackIPChi2;
  };

  struct two_track_mva_line_t : public SelectionAlgorithm, Parameters, TwoTrackLine<two_track_mva_line_t, Parameters> {
    __device__ static bool select(const Parameters&, std::tuple<const VertexFit::TrackMVAVertex&>);

  private:
    Property<pre_scaler_t> m_pre_scaler {this, 1.f};
    Property<post_scaler_t> m_post_scaler {this, 1.f};
    Property<pre_scaler_hash_string_t> m_pre_scaler_hash_string {this, ""};
    Property<post_scaler_hash_string_t> m_post_scaler_hash_string {this, ""};
    Property<minComboPt_t> m_minComboPt {this, 2000.0f / Gaudi::Units::MeV};
    Property<maxVertexChi2_t> m_maxVertexChi2 {this, 25.0f};
    Property<minMCor_t> m_minMCor {this, 1000.0f / Gaudi::Units::MeV};
    Property<minEta_t> m_minEta {this, 2.0f};
    Property<maxEta_t> m_maxEta {this, 5.0f};
    Property<minTrackPt_t> m_minTrackPt {this, 700.f / Gaudi::Units::MeV};
    Property<maxNTrksAssoc_t> m_maxNTrksAssoc {this, 1};
    Property<minFDChi2_t> m_minFDChi2 {this, 0.0f};
    Property<minTrackIPChi2_t> m_minTrackIPChi2 {this, 12.f};
  };
} // namespace two_track_mva_line