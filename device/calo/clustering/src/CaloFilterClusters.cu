/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "CaloFilterClusters.cuh"

INSTANTIATE_ALGORITHM(calo_filter_clusters::calo_filter_clusters_t)

void calo_filter_clusters::calo_filter_clusters_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_cluster1_idx_t>(arguments, first<host_ecal_number_of_twoclusters_t>(arguments));
  set_size<dev_cluster2_idx_t>(arguments, first<host_ecal_number_of_twoclusters_t>(arguments));
}
void calo_filter_clusters::calo_filter_clusters_t::init()
{
#ifndef ALLEN_STANDALONE
  m_calo_clusters = std::make_unique<Gaudi::Accumulators::Counter<>>(this, "n_calo_clusters");
#endif
}

void calo_filter_clusters::calo_filter_clusters_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  global_function(calo_filter_clusters)(
    dim3(size<dev_event_list_t>(arguments)), property<block_dim_filter_t>(), context)(arguments);

#ifndef ALLEN_STANDALONE
  // Monitoring
  auto host_ecal_cluster_offsets = make_host_buffer<dev_ecal_cluster_offsets_t>(arguments, context);
  for (auto i = 0u; i < first<host_number_of_events_t>(arguments); ++i) {
    auto n_clusters_event = host_ecal_cluster_offsets[i + 1] - host_ecal_cluster_offsets[i];
    (*m_calo_clusters) += n_clusters_event;
  }
#endif
}

__global__ void calo_filter_clusters::calo_filter_clusters(calo_filter_clusters::Parameters parameters)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  const unsigned ecal_twoclusters_offsets = parameters.dev_ecal_twocluster_offsets[event_number];
  unsigned* event_cluster1_idx = parameters.dev_cluster1_idx + ecal_twoclusters_offsets;
  unsigned* event_cluster2_idx = parameters.dev_cluster2_idx + ecal_twoclusters_offsets;

  const unsigned ecal_cluster_offsets = parameters.dev_ecal_cluster_offsets[event_number];
  const unsigned* prefiltered_clusters_idx = parameters.dev_prefiltered_clusters_idx + ecal_cluster_offsets;
  const unsigned n_prefltred_clusters = parameters.dev_num_prefiltered_clusters[event_number];

  // Loop over pre-filtered clusters.
  for (unsigned i_cluster = threadIdx.x; i_cluster < n_prefltred_clusters; i_cluster += blockDim.x) {
    const unsigned dicluster_idx_offset = i_cluster * n_prefltred_clusters - ((i_cluster + 1) * i_cluster) / 2;
    for (unsigned j_cluster = threadIdx.y + i_cluster + 1; j_cluster < n_prefltred_clusters; j_cluster += blockDim.y) {
      const unsigned dicluster_idx = dicluster_idx_offset + j_cluster - i_cluster - 1;
      event_cluster1_idx[dicluster_idx] = prefiltered_clusters_idx[i_cluster];
      event_cluster2_idx[dicluster_idx] = prefiltered_clusters_idx[j_cluster];
    }
  }
}
