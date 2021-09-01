/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "MEPTools.h"
#include "SciFiPreDecodeV4.cuh"
#include <cassert>

__device__ void store_sorted_cluster_reference_v4(
  SciFi::ConstHitCount& hit_count,
  const uint32_t uniqueMat,
  const uint32_t chan,
  const uint32_t& shared_mat_offsets,
  uint32_t& shared_mat_count,
  const int raw_bank,
  const int it,
  uint32_t* cluster_references)
{
  uint32_t hitIndex = shared_mat_count++;

  const SciFi::SciFiChannelID id {chan};
  if (id.reversedZone()) {
    hitIndex = hit_count.mat_number_of_hits(uniqueMat) - 1 - hitIndex;
  }

  assert(hitIndex < hit_count.mat_number_of_hits(uniqueMat));
  hitIndex += shared_mat_offsets;

  // Cluster reference:
  //   raw bank: 8 bits
  //   element (it): 8 bits
  cluster_references[hitIndex] = (raw_bank & 0xFF) << 8 | (it & 0xFF);
}

__device__ void pre_decode_raw_bank_v4(
  SciFi::ConstHitCount& hit_count,
  SciFi::SciFiGeometry const& geom,
  SciFi::SciFiRawBank const& rawbank,
  const unsigned bank_index,
  uint32_t const* shared_mat_offsets,
  uint32_t* shared_mat_count,
  uint32_t* cluster_references)
{
  const uint16_t* starting_it = rawbank.data + 2;
  uint16_t* last = rawbank.last;
  if (*(last - 1) == 0) --last; // Remove padding at the end

  if (starting_it < last) {
    const unsigned number_of_iterations = last - starting_it;
    for (unsigned it_number = 0; it_number < number_of_iterations; ++it_number) {
      auto it = starting_it + it_number;
      const uint16_t c = *it;
      const uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + SciFi::channelInBank(c);
      const auto chid = SciFi::SciFiChannelID(ch);
      const uint32_t correctedMat = chid.correctedUniqueMat();

      store_sorted_cluster_reference_v4(
        hit_count,
        correctedMat,
        ch,
        shared_mat_offsets
          [correctedMat - SciFi::Constants::n_consecutive_raw_banks * SciFi::Constants::n_mats_per_consec_raw_bank],
        shared_mat_count
          [correctedMat - SciFi::Constants::n_consecutive_raw_banks * SciFi::Constants::n_mats_per_consec_raw_bank],
        bank_index,
        it_number,
        cluster_references);
    }
  }
}

template<bool mep_layout>
__global__ void scifi_pre_decode_v4_kernel(scifi_pre_decode_v4::Parameters parameters, const char* scifi_geometry)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  SciFi::SciFiGeometry geom(scifi_geometry);
  SciFi::ConstHitCount hit_count {parameters.dev_scifi_hit_count, event_number};

  __shared__ uint32_t shared_mat_offsets[SciFi::Constants::n_mats_without_group];
  __shared__ uint32_t shared_mat_count[SciFi::Constants::n_mats_without_group];
  for (unsigned i = threadIdx.x; i < SciFi::Constants::n_mats_without_group; i += blockDim.x) {
    shared_mat_offsets[i] = hit_count.mat_offsets(
      i + SciFi::Constants::n_consecutive_raw_banks * SciFi::Constants::n_mats_per_consec_raw_bank);
    shared_mat_count[i] = 0;
  }

  __syncthreads();

  const auto scifi_raw_event =
    SciFi::RawEvent<mep_layout>(parameters.dev_scifi_raw_input, parameters.dev_scifi_raw_input_offsets, event_number);

  // Main execution loop
  for (unsigned i = SciFi::Constants::n_consecutive_raw_banks + threadIdx.x; i < scifi_raw_event.number_of_raw_banks();
       i += blockDim.x) {
    pre_decode_raw_bank_v4(
      hit_count,
      geom,
      scifi_raw_event.raw_bank(i),
      i,
      (const uint32_t*) &shared_mat_offsets,
      (uint32_t*) &shared_mat_count,
      parameters.dev_cluster_references);
  }
}

void scifi_pre_decode_v4::scifi_pre_decode_v4_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_cluster_references_t>(
    arguments, first<host_accumulated_number_of_scifi_hits_t>(arguments) * SciFi::Hits::number_of_arrays);
}

void scifi_pre_decode_v4::scifi_pre_decode_v4_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers&,
  const Allen::Context& context) const
{
  global_function(runtime_options.mep_layout ? scifi_pre_decode_v4_kernel<true> : scifi_pre_decode_v4_kernel<false>)(
    dim3(size<dev_event_list_t>(arguments)), dim3(SciFi::SciFiRawBankParams::NbBanks), context)(
    arguments, constants.dev_scifi_geometry);
}
