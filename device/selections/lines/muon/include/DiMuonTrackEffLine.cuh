/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"
#include "TwoTrackLine.cuh"

namespace di_muon_track_eff_line {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_svs_t, unsigned) host_number_of_svs;
    DEVICE_INPUT(dev_particle_container_t, Allen::Views::Physics::MultiEventCompositeParticles) dev_particle_container;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    HOST_OUTPUT(host_decisions_size_t, unsigned) host_decisions_size;
    HOST_OUTPUT(host_post_scaler_t, float) host_post_scaler;
    HOST_OUTPUT(host_post_scaler_hash_t, uint32_t) host_post_scaler_hash;

    HOST_OUTPUT_WITH_DEPENDENCIES(host_fn_parameters_t, DEPENDENCIES(dev_particle_container_t), char)
    host_fn_parameters;
    PROPERTY(pre_scaler_t, "pre_scaler", "Pre-scaling factor", float) pre_scaler;
    PROPERTY(post_scaler_t, "post_scaler", "Post-scaling factor", float) post_scaler;
    PROPERTY(pre_scaler_hash_string_t, "pre_scaler_hash_string", "Pre-scaling hash string", std::string);
    PROPERTY(post_scaler_hash_string_t, "post_scaler_hash_string", "Post-scaling hash string", std::string);
    PROPERTY(DMTrackEffM0_t, "DMTrackEffM0", "DMTrackEffM0 description", float) DMTrackEffM0;
    PROPERTY(DMTrackEffM1_t, "DMTrackEffM1", "DMTrackEffM1 description", float) DMTrackEffM1;
    PROPERTY(DMTrackEffMinZ_t, "DMTrackEffMinZ", "MinZ for DMTrackEff", float) DMTrackEffMinZ;
    PROPERTY(OppositeSign_t, "OppositeSign", "Selects opposite sign dimuon combinations", bool) OppositeSign;
  };

  struct di_muon_track_eff_line_t : public SelectionAlgorithm,
                                    Parameters,
                                    TwoTrackLine<di_muon_track_eff_line_t, Parameters> {
    __device__ static bool select(const Parameters&, std::tuple<const Allen::Views::Physics::CompositeParticle>);

  private:
    Property<pre_scaler_t> m_pre_scaler {this, 1.f};
    Property<post_scaler_t> m_post_scaler {this, 1.f};
    Property<pre_scaler_hash_string_t> m_pre_scaler_hash_string {this, ""};
    Property<post_scaler_hash_string_t> m_post_scaler_hash_string {this, ""};
    // Mass window around J/psi meson.
    Property<DMTrackEffM0_t> m_DMTrackEffM0 {this, 2900.f};
    Property<DMTrackEffM1_t> m_DMTrackEffM1 {this, 3100.f};
    Property<DMTrackEffMinZ_t> m_DMTrackEffMinZ {this, -341.f};
    Property<OppositeSign_t> m_opposite_sign {this, true};
  };
} // namespace di_muon_track_eff_line
