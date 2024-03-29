/*****************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/

#pragma once

#include "CaloGeometry.cuh"
#include "CaloDigit.cuh"
#include "CaloCluster.cuh"
#include "AlgorithmTypes.cuh"
#include <cfloat>

namespace calo_find_clusters {
  struct Parameters {
    HOST_INPUT(host_ecal_number_of_clusters_t, unsigned) host_ecal_number_of_clusters;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_ecal_digits_t, CaloDigit) dev_ecal_digits;
    DEVICE_INPUT(dev_ecal_digits_offsets_t, unsigned) dev_ecal_digits_offsets;
    DEVICE_INPUT(dev_ecal_seed_clusters_t, CaloSeedCluster) dev_ecal_seed_clusters;
    DEVICE_INPUT(dev_ecal_cluster_offsets_t, unsigned) dev_ecal_cluster_offsets;
    DEVICE_OUTPUT(dev_ecal_clusters_t, CaloCluster) dev_ecal_clusters;
    PROPERTY(block_dim_x_t, "block_dim_x", "block dimension X", unsigned) block_dim_x;
    PROPERTY(block_dim_y_t, "block_dim_y", "block dimension Y", unsigned) block_dim_y;
    PROPERTY(ecal_min_adc_t, "ecal_min_adc", "cluster neighbors' minimum ADC", int16_t) ecal_min_adc;
  };

  // Global function
  __global__ void calo_find_clusters(Parameters parameters, const char* raw_ecal_geometry, const int16_t min_adc);

  // Algorithm
  struct calo_find_clusters_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    __host__ void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      Allen::Context const&) const;

  private:
    Property<block_dim_x_t> m_block_dim_x {this, 64};
    Property<block_dim_y_t> m_block_dim_y {this, 16};
    Property<ecal_min_adc_t> m_ecal_min_adc {this, 10};
  };
} // namespace calo_find_clusters
