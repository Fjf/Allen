/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"
#include "TwoTrackLine.cuh"
#include "MassDefinitions.h"

namespace d2kpi_line {
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
    PROPERTY(minComboPt_t, "minComboPt", "minComboPt description", float) minComboPt;
    PROPERTY(maxVertexChi2_t, "maxVertexChi2", "maxVertexChi2 description", float) maxVertexChi2;
    PROPERTY(maxDOCA_t, "maxDOCA", "maxDOCA description", float) maxDOCA;
    PROPERTY(minEta_t, "minEta", "minEta description", float) minEta;
    PROPERTY(maxEta_t, "maxEta", "maxEta description", float) maxEta;
    PROPERTY(minTrackPt_t, "minTrackPt", "minTrackPt description", float) minTrackPt;
    PROPERTY(massWindow_t, "massWindow", "massWindow description", float) massWindow;
    PROPERTY(minTrackIP_t, "minTrackIP", "minTrackIP description", float) minTrackIP;
    PROPERTY(minZ_t, "minZ", "minimum vertex z coordinate", float) minZ;

    HOST_OUTPUT(host_histogram_d0_mass_t, float) host_histogram_d0_mass;
    DEVICE_OUTPUT(dev_histogram_d0_mass_t, float) dev_histogram_d0_mass;
    PROPERTY(histogram_d0_mass_min_t, "histogram_d0_mass_min", "histogram_d0_mass_min description", float)
    histogram_d0_mass_min;
    PROPERTY(histogram_d0_mass_max_t, "histogram_d0_mass_max", "histogram_d0_mass_max description", float)
    histogram_d0_mass_max;
    PROPERTY(histogram_d0_mass_nbins_t, "histogram_d0_mass_nbins", "histogram_d0_mass_nbins description", unsigned int)
    histogram_d0_mass_nbins;

    HOST_OUTPUT(host_histogram_d0_pt_t, float) host_histogram_d0_pt;
    DEVICE_OUTPUT(dev_histogram_d0_pt_t, float) dev_histogram_d0_pt;
    PROPERTY(histogram_d0_pt_min_t, "histogram_d0_pt_min", "histogram_d0_pt_min description", float)
    histogram_d0_pt_min;
    PROPERTY(histogram_d0_pt_max_t, "histogram_d0_pt_max", "histogram_d0_pt_max description", float)
    histogram_d0_pt_max;
    PROPERTY(histogram_d0_pt_nbins_t, "histogram_d0_pt_nbins", "histogram_d0_pt_nbins description", unsigned int)
    histogram_d0_pt_nbins;
  };

  struct d2kpi_line_t : public SelectionAlgorithm, Parameters, TwoTrackLine<d2kpi_line_t, Parameters> {
    __device__ static bool select(const Parameters&, std::tuple<const Allen::Views::Physics::CompositeParticle>);
    void init();

    static void init_monitor(const ArgumentReferences<Parameters>& arguments, const Allen::Context& context);
    __device__ static void monitor(
      const Parameters& parameters,
      std::tuple<const Allen::Views::Physics::CompositeParticle> input,
      unsigned index,
      bool sel);
    __host__ void
    output_monitor(const ArgumentReferences<Parameters>& arguments, const RuntimeOptions&, const Allen::Context&) const;

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
    Property<minComboPt_t> m_minComboPt {this, 2000.0f * Gaudi::Units::MeV};
    Property<maxVertexChi2_t> m_maxVertexChi2 {this, 10.f};
    Property<maxDOCA_t> m_maxDOCA {this, 0.15f * Gaudi::Units::mm};
    Property<minEta_t> m_minEta {this, 2.0f};
    Property<maxEta_t> m_maxEta {this, 5.0f};
    Property<minTrackPt_t> m_minTrackPt {this, 800.f * Gaudi::Units::MeV};
    Property<massWindow_t> m_massWindow {this, 100.f * Gaudi::Units::MeV};
    Property<minTrackIP_t> m_minTrackIP {this, 0.06f * Gaudi::Units::mm};
    Property<minZ_t> m_minZ {this, -300.f * Gaudi::Units::mm};
    Property<minTrackIPChi2_t> m_minTrackIPChi2 {this, 9.f};
    Property<histogram_d0_mass_min_t> m_histogramD0MassMin {this, 1765.f};
    Property<histogram_d0_mass_max_t> m_histogramD0MassMax {this, 1965.f};
    Property<histogram_d0_mass_nbins_t> m_histogramD0MassNBins {this, 100u};
    Property<histogram_d0_pt_min_t> m_histogramD0PtMin {this, 0.f};
    Property<histogram_d0_pt_max_t> m_histogramD0PtMax {this, 1e4f};
    Property<histogram_d0_pt_nbins_t> m_histogramD0PtNBins {this, 100u};

#ifndef ALLEN_STANDALONE
    void* histogram_d0_mass;
    void* histogram_d0_pt;
#endif
  };
} // namespace d2kpi_line
