/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"
#include "OneTrackLine.cuh"

namespace track_electron_mva_line {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_scifi_tracks_t, unsigned) host_number_of_reconstructed_scifi_tracks;
    DEVICE_INPUT(dev_particle_container_t, Allen::Views::Physics::MultiEventBasicParticles) dev_particle_container;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_track_isElectron_t, bool) dev_track_isElectron;
    DEVICE_INPUT(dev_brem_corrected_pt_t, float) dev_brem_corrected_pt;
    HOST_OUTPUT(host_decisions_size_t, unsigned) host_decisions_size;
    HOST_OUTPUT(host_post_scaler_t, float) host_post_scaler;
    HOST_OUTPUT(host_post_scaler_hash_t, uint32_t) host_post_scaler_hash;

    HOST_OUTPUT_WITH_DEPENDENCIES(host_fn_parameters_t, DEPENDENCIES(dev_particle_container_t), char)
    host_fn_parameters;
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
    PROPERTY(minBPVz_t, "min_BPVz", "Minimum z for the associated best primary vertex", float) minBPVz;
    DEVICE_OUTPUT(pt_t, float) pt;
    DEVICE_OUTPUT(pt_corrected_t, float) pt_corrected;
    DEVICE_OUTPUT(ipchi2_t, float) ipchi2;
    DEVICE_OUTPUT(evtNo_t, uint64_t) evtNo;
    DEVICE_OUTPUT(runNo_t, unsigned) runNo;

    PROPERTY(enable_tupling_t, "enable_tupling", "Enables monitoring ntuple", bool) enable_tupling;
  };

  struct track_electron_mva_line_t : public SelectionAlgorithm,
                                     Parameters,
                                     OneTrackLine<track_electron_mva_line_t, Parameters> {
    __device__ static bool select(
      const Parameters& ps,
      std::tuple<const Allen::Views::Physics::BasicParticle, const bool, const float> input);

    __device__ static void fill_tuples(
      const Parameters& parameters,
      std::tuple<const Allen::Views::Physics::BasicParticle, const bool, const float> input,
      unsigned index,
      bool sel);

    using monitoring_types = std::tuple<pt_t, pt_corrected_t, ipchi2_t, evtNo_t, runNo_t>;

    __device__ static std::tuple<const Allen::Views::Physics::BasicParticle, const bool, const float>
    get_input(const Parameters& parameters, const unsigned event_number, const unsigned i);

  private:
    Property<pre_scaler_t> m_pre_scaler {this, 1.f};
    Property<post_scaler_t> m_post_scaler {this, 1.f};
    Property<pre_scaler_hash_string_t> m_pre_scaler_hash_string {this, ""};
    Property<post_scaler_hash_string_t> m_post_scaler_hash_string {this, ""};
    Property<maxChi2Ndof_t> m_maxChi2Ndof {this, 2.5f};
    Property<minPt_t> m_minPt {this, 1.f * Gaudi::Units::GeV};
    Property<maxPt_t> m_maxPt {this, 26.f * Gaudi::Units::GeV};
    Property<minIPChi2_t> m_minIPChi2 {this, 3.f};
    Property<param1_t> m_param1 {this, 1.f * Gaudi::Units::GeV* Gaudi::Units::GeV};
    Property<param2_t> m_param2 {this, 1.f * Gaudi::Units::GeV};
    Property<param3_t> m_param3 {this, 1.248f};
    Property<alpha_t> m_alpha {this, 0.f};
    Property<minBPVz_t> m_minBPVz {this, -341.f * Gaudi::Units::mm};

    Property<enable_tupling_t> m_enable_tupling {this, false};
  };
} // namespace track_electron_mva_line
