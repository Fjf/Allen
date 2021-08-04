/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#pragma once

#include "SelectionAlgorithm.cuh"
#include "TwoTrackLine.cuh"
#include "ROOTService.h"

namespace kstopipi_line {
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

    DEVICE_OUTPUT(dev_sv_masses_t, float) dev_sv_masses;
    HOST_OUTPUT(host_sv_masses_t, float) host_sv_masses;

    DEVICE_OUTPUT(dev_pt_t, float) dev_pt;
    HOST_OUTPUT(host_pt_t, float) host_pt;

    PROPERTY(pre_scaler_t, "pre_scaler", "Pre-scaling factor", float) pre_scaler;
    PROPERTY(post_scaler_t, "post_scaler", "Post-scaling factor", float) post_scaler;
    PROPERTY(pre_scaler_hash_string_t, "pre_scaler_hash_string", "Pre-scaling hash string", std::string);
    PROPERTY(post_scaler_hash_string_t, "post_scaler_hash_string", "Post-scaling hash string", std::string);
    PROPERTY(minHighMassTrackPt_t, "minHighMassTrackPt", "minHighMassTrackPt description", float) minHighMassTrackPt;
    PROPERTY(minHighMassTrackP_t, "minHighMassTrackP", "minHighMassTrackP description", float) minHighMassTrackP;
    PROPERTY(minMass_t, "minMass", "minMass description", float) minMass;
    PROPERTY(maxDoca_t, "maxDoca", "maxDoca description", float) maxDoca;
    PROPERTY(maxVertexChi2_t, "maxVertexChi2", "maxVertexChi2 description", float) maxVertexChi2;
    PROPERTY(minIPChi2_t, "minIPChi2", "minIPChi2 description", float) minIPChi2;
  };

  struct kstopipi_line_t : public SelectionAlgorithm, Parameters, TwoTrackLine<kstopipi_line_t, Parameters> {
    __device__ static bool select(const Parameters&, std::tuple<const VertexFit::TrackMVAVertex&>);
#ifdef WITH_ROOT
    static void init_monitor(const ArgumentReferences<Parameters>& arguments, const Allen::Context& context);
    __device__ static void
    monitor(const Parameters& parameters, std::tuple<const VertexFit::TrackMVAVertex&> input, unsigned index, bool sel);
    __host__ void
    output_monitor(const ArgumentReferences<Parameters>& arguments, const RuntimeOptions&, const Allen::Context&) const;
#endif

    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const;

  private:
    Property<pre_scaler_t> m_pre_scaler {this, 1.f};
    Property<post_scaler_t> m_post_scaler {this, 1.f};
    Property<pre_scaler_hash_string_t> m_pre_scaler_hash_string {this, ""};
    Property<post_scaler_hash_string_t> m_post_scaler_hash_string {this, ""};
    Property<minHighMassTrackPt_t> m_minHighMassTrackPt {this, 300.f / Gaudi::Units::MeV};
    Property<minHighMassTrackP_t> m_minHighMassTrackP {this, 6000.f / Gaudi::Units::MeV};
    Property<minMass_t> m_minMass {this, 2700.f / Gaudi::Units::MeV};
    Property<maxDoca_t> m_maxDoca {this, 0.2f};
    Property<maxVertexChi2_t> m_maxVertexChi2 {this, 25.0f};
    Property<minIPChi2_t> m_minIPChi2 {this, 0.f};
  };
} // namespace kstopipi_line