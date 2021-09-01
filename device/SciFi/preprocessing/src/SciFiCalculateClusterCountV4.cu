/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <MEPTools.h>
#include <SciFiCalculateClusterCountV4.cuh>

/**
 * @brief This function calculates the amount of clusters in all mats.
 * @details More details about the SciFi format:
 *          https://cds.cern.ch/record/2630154/files/LHCb-INT-2018-024.pdf
 *
 * Kernel for decoding from MEP layout
 */
template<bool mep_layout>
__global__ void scifi_calculate_cluster_count_v4_kernel(
  scifi_calculate_cluster_count_v4::Parameters parameters,
  const char* scifi_geometry)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  const auto scifi_raw_event =
    SciFi::RawEvent<mep_layout>(parameters.dev_scifi_raw_input, parameters.dev_scifi_raw_input_offsets, event_number);
  const SciFi::SciFiGeometry geom(scifi_geometry);
  SciFi::HitCount hit_count {parameters.dev_scifi_hit_count, event_number};

  for (unsigned i = threadIdx.x; i < SciFi::Constants::n_consecutive_raw_banks; i += blockDim.x) {
    const unsigned current_raw_bank = SciFi::getRawBankIndexOrderedByX(i);
    const auto rawbank = scifi_raw_event.raw_bank(current_raw_bank);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;

    if (*(last - 1) == 0) --last; // Remove padding at the end
    const unsigned number_of_clusters = last - it;

    if (last > it) {
      hit_count.set_mat_offsets(i, number_of_clusters);
    }
  }

  const unsigned mats_difference = 3 * SciFi::Constants::n_consecutive_raw_banks;
  for (unsigned i = SciFi::Constants::n_consecutive_raw_banks + threadIdx.x; i < scifi_raw_event.number_of_raw_banks();
       i += blockDim.x) {
    uint32_t* hits_mat;
    const auto rawbank = scifi_raw_event.raw_bank(i);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;

    // For details see RawBankDecoder
    if (*(last - 1) == 0) --last; // Remove padding at the end
    for (; it < last; ++it) {     // loop over the clusters
      uint16_t c = *it;
      uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + SciFi::channelInBank(c);
      hits_mat = hit_count.mat_offsets_p(SciFi::SciFiChannelID(ch).correctedUniqueMat() - mats_difference);
      atomicAdd(hits_mat, 1);
    }
  }
}

void scifi_calculate_cluster_count_v4::scifi_calculate_cluster_count_v4_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_scifi_hit_count_t>(
    arguments, first<host_number_of_events_t>(arguments) * SciFi::Constants::n_mat_groups_and_mats);
}

void scifi_calculate_cluster_count_v4::scifi_calculate_cluster_count_v4_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers&,
  const Allen::Context& context) const
{
  initialize<dev_scifi_hit_count_t>(arguments, 0, context);

  global_function(
    runtime_options.mep_layout ? scifi_calculate_cluster_count_v4_kernel<true> :
                                 scifi_calculate_cluster_count_v4_kernel<false>)(
    dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(arguments, constants.dev_scifi_geometry);
}
