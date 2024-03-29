/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"
#include "TwoTrackLine.cuh"
#include "ROOTService.h"

#ifndef ALLEN_STANDALONE
#include "GaudiMonitoring.h"
#include <Gaudi/Accumulators.h>
#endif

namespace di_muon_drell_yan_line {
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

    PROPERTY(minTrackP_t, "minTrackP", "Minimal momentum for both daughters ", float) minTrackP;
    PROPERTY(minTrackPt_t, "minTrackPt", "Minimal pT for both daughters", float) minTrackPt;
    PROPERTY(maxTrackEta_t, "maxTrackEta", "Maximal ETA for both daughters", float) maxTrackEta;

    PROPERTY(maxDoca_t, "maxDoca", "maxDoca description", float) maxDoca;
    PROPERTY(maxVertexChi2_t, "maxVertexChi2", "maxVertexChi2 description", float) maxVertexChi2;

    PROPERTY(minMass_t, "minMass", "Min mass of the composite", float) minMass;
    PROPERTY(maxMass_t, "maxMass", "Max mass of the composite", float) maxMass;

    PROPERTY(OppositeSign_t, "OppositeSign", "Selects opposite sign dimuon combinations", bool) OppositeSign;
    PROPERTY(minZ_t, "minZ", "minimum vertex z coordinate", float) minZ;

    PROPERTY(enable_monitoring_t, "enable_monitoring", "Enable line monitoring", bool) enable_monitoring;
    PROPERTY(enable_tupling_t, "enable_tupling", "Enable line tupling", bool) enable_tupling;
    DEVICE_OUTPUT(dev_histogram_Z_mass_t, unsigned) dev_histogram_Z_mass;
    DEVICE_OUTPUT(dev_histogram_Z_mass_ss_t, unsigned) dev_histogram_Z_mass_ss;
    PROPERTY(histogram_Z_mass_min_t, "histogram_Z_mass_min", "histogram_Z_mass_min description", float)
    histogram_Z_mass_min;
    PROPERTY(histogram_Z_mass_max_t, "histogram_Z_mass_max", "histogram_Z_mass_max description", float)
    histogram_Z_mass_max;
    PROPERTY(histogram_Z_mass_nbins_t, "histogram_Z_mass_nbins", "histogram_Z_mass_nbins description", unsigned int)
    histogram_Z_mass_nbins;

    DEVICE_OUTPUT(mass_t, float) mass;
    DEVICE_OUTPUT(transverse_momentum_t, float) transverse_momentum;

    DEVICE_OUTPUT(evtNo_t, uint64_t) evtNo;
    DEVICE_OUTPUT(runNo_t, unsigned) runNo;
  };

  struct di_muon_drell_yan_line_t : public SelectionAlgorithm,
                                    Parameters,
                                    TwoTrackLine<di_muon_drell_yan_line_t, Parameters> {
    __device__ static bool select(const Parameters&, std::tuple<const Allen::Views::Physics::CompositeParticle>);
    void init();
    static void init_monitor(const ArgumentReferences<Parameters>& arguments, const Allen::Context& context);

    __device__ static void
    monitor(const Parameters&, std::tuple<const Allen::Views::Physics::CompositeParticle>, unsigned, bool);
    __device__ static void
    fill_tuples(const Parameters&, std::tuple<const Allen::Views::Physics::CompositeParticle>, unsigned, bool);
    __host__ void
    output_monitor(const ArgumentReferences<Parameters>& arguments, const RuntimeOptions&, const Allen::Context&) const;
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

  private:
    Property<pre_scaler_t> m_pre_scaler {this, 1.f};
    Property<post_scaler_t> m_post_scaler {this, 1.f};
    Property<pre_scaler_hash_string_t> m_pre_scaler_hash_string {this, ""};
    Property<post_scaler_hash_string_t> m_post_scaler_hash_string {this, ""};

    Property<minTrackP_t> m_minTrackP {this, 15.f * Gaudi::Units::GeV};
    Property<minTrackPt_t> m_minTrackPt {this, 1.2f * Gaudi::Units::GeV};
    Property<maxTrackEta_t> m_maxTrackEta {this, 4.9};

    Property<minMass_t> m_minMass {this, 5.f * Gaudi::Units::GeV};
    Property<maxMass_t> m_maxMass {this, 400.f * Gaudi::Units::GeV};

    Property<maxDoca_t> m_maxDoca {this, .15f * Gaudi::Units::mm};
    Property<maxVertexChi2_t> m_maxVertexChi2 {this, 20.f};

    Property<OppositeSign_t> m_only_select_opposite_sign {this, true};
    Property<minZ_t> m_minZ {this, -341.f * Gaudi::Units::mm};

    Property<histogram_Z_mass_min_t> m_histogramZMassMin {this, 60000.f};
    Property<histogram_Z_mass_max_t> m_histogramZMassMax {this, 120000.f};
    Property<histogram_Z_mass_nbins_t> m_histogramZMassNBins {this, 100u};
    Property<enable_monitoring_t> m_enable_monitoring {this, false};
    Property<enable_tupling_t> m_enable_tupling {this, false};

#ifndef ALLEN_STANDALONE
    gaudi_monitoring::Lockable_Histogram<>* histogram_Z_mass;
    gaudi_monitoring::Lockable_Histogram<>* histogram_Z_mass_ss;
#endif

    using monitoring_types = std::tuple<transverse_momentum_t, mass_t, evtNo_t, runNo_t>;
  };
} // namespace di_muon_drell_yan_line
