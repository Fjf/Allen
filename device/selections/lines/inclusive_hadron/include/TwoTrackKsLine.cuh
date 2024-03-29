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
#pragma once

#include "AlgorithmTypes.cuh"
#include "TwoTrackLine.cuh"
#include "MassDefinitions.h"

namespace two_track_line_ks {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    HOST_OUTPUT(host_decisions_size_t, unsigned) host_decisions_size;
    HOST_OUTPUT(host_post_scaler_t, float) host_post_scaler;
    HOST_OUTPUT(host_post_scaler_hash_t, uint32_t) host_post_scaler_hash;

    PROPERTY(pre_scaler_t, "pre_scaler", "Pre-scaling factor", float) pre_scaler;
    PROPERTY(post_scaler_t, "post_scaler", "Post-scaling factor", float) post_scaler;
    PROPERTY(pre_scaler_hash_string_t, "pre_scaler_hash_string", "Pre-scaling hash string", std::string);
    PROPERTY(post_scaler_hash_string_t, "post_scaler_hash_string", "Post-scaling hash string", std::string);
    // Line-specific inputs and properties
    HOST_INPUT(host_number_of_svs_t, unsigned) host_number_of_svs;
    DEVICE_INPUT(dev_particle_container_t, Allen::Views::Physics::MultiEventCompositeParticles) dev_particle_container;
    HOST_OUTPUT_WITH_DEPENDENCIES(host_fn_parameters_t, DEPENDENCIES(dev_particle_container_t), char)
    host_fn_parameters;
    PROPERTY(maxVertexChi2_t, "maxVertexChi2", "maxVertexChi2 description", float) maxVertexChi2;
    PROPERTY(minComboPt_Ks_t, "minComboPt_Ks", "minComboPt Ks description", float) minComboPt_Ks;
    PROPERTY(minCosDira_t, "minCosDira", "minCosDira description", float) minCosDira;
    PROPERTY(minEta_Ks_t, "minEta_Ks", "minEta_Ks description", float) minEta_Ks;
    PROPERTY(maxEta_Ks_t, "maxEta_Ks", "maxEta_Ks description", float) maxEta_Ks;
    PROPERTY(minTrackPt_piKs_t, "minTrackPt_piKs", "minTrackPt_piKs description", float) minTrackPt_piKs;
    PROPERTY(minTrackP_piKs_t, "minTrackP_piKs", "minTrackP_piKs description", float) minTrackP_piKs;
    PROPERTY(minTrackIPChi2_Ks_t, "minTrackIPChi2_Ks", "minTrackIPChi2_Ks description", float) minTrackIPChi2_Ks;
    PROPERTY(minM_Ks_t, "minM_Ks", "minM_Ks description", float) minM_Ks;
    PROPERTY(maxM_Ks_t, "maxM_Ks", "maxM_Ks description", float) maxM_Ks;
    PROPERTY(minCosOpening_t, "minCosOpening", "minCosOpening description", float) minCosOpening;
    PROPERTY(min_combip_t, "min_combip", "min_combip description", float) min_combip;
    PROPERTY(OppositeSign_t, "OppositeSign", "Selects opposite sign dimuon combinations", bool) OppositeSign;
    PROPERTY(minZ_t, "minZ", "minimum vertex z coordinate", float) minZ;

    DEVICE_OUTPUT(eta_ks_t, float) eta_ks;
    DEVICE_OUTPUT(pt_ks_t, float) pt_ks;
    DEVICE_OUTPUT(min_pt_t, float) min_pt;
    DEVICE_OUTPUT(min_ipchi2_t, float) min_ipchi2;
    DEVICE_OUTPUT(min_p_t, float) min_p;
    DEVICE_OUTPUT(comb_ip_t, float) comb_ip;
    DEVICE_OUTPUT(mass_t, float) mass;
    DEVICE_OUTPUT(evtNo_t, uint64_t) evtNo;
    DEVICE_OUTPUT(runNo_t, unsigned) runNo;
    PROPERTY(enable_tupling_t, "enable_tupling", "Enable line monitoring", bool) enable_tupling;
  };

  struct two_track_line_ks_t : public SelectionAlgorithm, Parameters, TwoTrackLine<two_track_line_ks_t, Parameters> {
    __device__ static bool select(const Parameters&, std::tuple<const Allen::Views::Physics::CompositeParticle>);
    __device__ static void fill_tuples(
      const Parameters& parameters,
      std::tuple<const Allen::Views::Physics::CompositeParticle> input,
      unsigned index,
      bool sel);

    using monitoring_types =
      std::tuple<eta_ks_t, pt_ks_t, min_pt_t, min_ipchi2_t, min_p_t, comb_ip_t, mass_t, evtNo_t, runNo_t>;

  private:
    Property<pre_scaler_t> m_pre_scaler {this, 1.f};
    Property<post_scaler_t> m_post_scaler {this, 1.f};
    Property<pre_scaler_hash_string_t> m_pre_scaler_hash_string {this, ""};
    Property<post_scaler_hash_string_t> m_post_scaler_hash_string {this, ""};
    Property<maxVertexChi2_t> m_maxVertexChi2 {this, 20.f};
    Property<minComboPt_Ks_t> m_minComboPt_Ks {this, 2500.f / Gaudi::Units::MeV};
    Property<minCosDira_t> m_minCosDira {this, 0.99f};
    Property<minEta_Ks_t> m_minEta_Ks {this, 2.f};
    Property<maxEta_Ks_t> m_maxEta_Ks {this, 4.2f};
    Property<minTrackPt_piKs_t> m_minTrackPt_piKs {this, 470.f / Gaudi::Units::MeV};
    Property<minTrackP_piKs_t> m_minTrackP_piKs {this, 5000.f / Gaudi::Units::MeV};
    Property<minTrackIPChi2_Ks_t> m_minTrackIPChi2_Ks {this, 50.f};
    Property<minM_Ks_t> m_minM_Ks {this, 455.0f / Gaudi::Units::MeV};
    Property<maxM_Ks_t> m_maxM_Ks {this, 545.0f / Gaudi::Units::MeV};
    Property<minCosOpening_t> m_minCosOpening {this, 0.99f};
    Property<min_combip_t> m_min_combip {this, 0.72f / Gaudi::Units::mm};
    Property<minZ_t> m_minZ {this, -341.f * Gaudi::Units::mm};
    Property<OppositeSign_t> m_opposite_sign {this, true};

    Property<enable_tupling_t> m_enable_tupling {this, false};
  };
} // namespace two_track_line_ks
