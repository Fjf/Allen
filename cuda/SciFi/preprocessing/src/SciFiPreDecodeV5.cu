#include "SciFiPreDecodeV5.cuh"
#include "assert.h"

using namespace SciFi;

__device__ void store_sorted_cluster_reference_v5(
  const SciFi::HitCount& hit_count,
  const uint32_t uniqueMat,
  const uint32_t chan,
  const uint32_t* shared_mat_offsets,
  uint32_t* shared_mat_count,
  const int raw_bank,
  const int it,
  const int condition_1,
  const int condition_2,
  const int delta,
  SciFi::Hits& hits)
{
  uint32_t uniqueGroupOrMat;
  // adaptation to hybrid decoding
  if (uniqueMat < SciFi::Constants::n_consecutive_raw_banks * SciFi::Constants::n_mats_per_consec_raw_bank)
    uniqueGroupOrMat = uniqueMat / SciFi::Constants::n_mats_per_consec_raw_bank;
  else
    uniqueGroupOrMat = uniqueMat - SciFi::Constants::mat_index_substract;

  uint32_t hitIndex = shared_mat_count[uniqueGroupOrMat]++;

  const SciFi::SciFiChannelID id {chan};
  if (id.reversedZone()) {
    hitIndex = hit_count.mat_group_or_mat_number_of_hits(uniqueGroupOrMat) - 1 - hitIndex;
  }
  assert(hitIndex < hit_count.mat_group_or_mat_number_of_hits(uniqueGroupOrMat));
  assert(uniqueGroupOrMat < SciFi::Constants::n_mat_groups_and_mats);
  hitIndex += shared_mat_offsets[uniqueGroupOrMat];
  // Cluster reference:
  //   raw bank: 8 bits
  //   element (it): 8 bits
  //   Condition 1-2-3: 2 bits
  //   Condition 2.1-2.2: 1 bit
  //   Condition 2.1: log2(n+1) - 8 bits
  hits.cluster_reference[hitIndex] = (raw_bank & 0xFF) << 24 | (it & 0xFF) << 16 | (condition_1 & 0x03) << 14 |
                                     (condition_2 & 0x01) << 13 | (delta & 0xFF);
}

__global__ void scifi_pre_decode_v5(
  char* scifi_events,
  uint* scifi_event_offsets,
  const uint* event_list,
  uint* scifi_hit_count,
  uint* scifi_hits,
  char* scifi_geometry,
  const float* dev_inv_clus_res)
{
  const int number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint selected_event_number = event_list[event_number];

  SciFiGeometry geom(scifi_geometry);
  const auto event = SciFiRawEvent(scifi_events + scifi_event_offsets[selected_event_number]);

  Hits hits {
    scifi_hits, scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats], &geom, dev_inv_clus_res};
  HitCount hit_count {scifi_hit_count, event_number};

  __shared__ uint32_t shared_mat_offsets[SciFi::Constants::n_mat_groups_and_mats];
  __shared__ uint32_t shared_mat_count[SciFi::Constants::n_mat_groups_and_mats];

  for (uint i = threadIdx.x; i < SciFi::Constants::n_mat_groups_and_mats; i += blockDim.x) {
    shared_mat_offsets[i] = hit_count.mat_offsets[i];
    shared_mat_count[i] = 0;
  }

  __syncthreads();

  // Main execution loop
  for (uint i = threadIdx.x; i < event.number_of_raw_banks; i += blockDim.x) {
    const uint current_raw_bank = getRawBankIndexOrderedByX(i);
    auto rawbank = event.getSciFiRawBank(current_raw_bank);
    const uint16_t* starting_it = rawbank.data + 2;
    uint16_t* last = rawbank.last;
    if (*(last - 1) == 0) --last; // Remove padding at the end

    if (starting_it >= last) continue;

    // loop over hits in a raw bank
    for (uint it_number = 0; it_number < last - starting_it; ++it_number) {
      auto it = starting_it + it_number;
      const uint16_t c = *it;
      const uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
      const auto chid = SciFiChannelID(ch);
      const uint32_t correctedMat = chid.correctedUniqueMat();

      // Condition 1: "00"
      // Reconstructs a single cluster
      if (!cSize(c) || it + 1 == last) {
        const int condition_1 = 0x00;

        store_sorted_cluster_reference_v5(
          hit_count,
          correctedMat,
          ch,
          shared_mat_offsets,
          shared_mat_count,
          current_raw_bank,
          it_number,
          condition_1, // Condition 1
          0x00,        // Condition 2
          0x00,        // Delta
          hits);
      }
      else {
        const unsigned c2 = *(it + 1);
        if (cSize(c2) && getLinkInBank(c) == getLinkInBank(c2)) {
          // Condition 1: "01"
          // Reconstructs a big cluster, composed of two fragments
          const int condition_1 = 0x01;

          const auto delta = (cell(c2) - cell(c));
          if (delta > SciFiRawBankParams::clusterMaxWidth) {
            // Condition 2: "0"
            const int condition_2 = 0x00;

            for (auto j = SciFiRawBankParams::clusterMaxWidth; j < delta; j += SciFiRawBankParams::clusterMaxWidth) {
              // Delta equals j / SciFiRawBankParams::clusterMaxWidth
              const int delta_parameter = j / SciFiRawBankParams::clusterMaxWidth;
              store_sorted_cluster_reference_v5(
                hit_count,
                correctedMat,
                ch,
                shared_mat_offsets,
                shared_mat_count,
                current_raw_bank,
                it_number,
                condition_1,
                condition_2,
                delta_parameter,
                hits);
            }

            // Delta equals 0
            const int delta_parameter = 0;
            store_sorted_cluster_reference_v5(
              hit_count,
              correctedMat,
              ch,
              shared_mat_offsets,
              shared_mat_count,
              current_raw_bank,
              it_number,
              condition_1,
              condition_2,
              delta_parameter,
              hits);
          }
          else {
            // Condition 2: "1"
            const int condition_2 = 0x01;
            store_sorted_cluster_reference_v5(
              hit_count,
              correctedMat,
              ch,
              shared_mat_offsets,
              shared_mat_count,
              current_raw_bank,
              it_number,
              condition_1,
              condition_2,
              0x00,
              hits);
          }

          // Due to v5
          ++it_number;
        }
        else {
          // Condition 1: "10"
          // Reconstructs a single cluster
          const int condition_1 = 0x02;

          store_sorted_cluster_reference_v5(
            hit_count,
            correctedMat,
            ch,
            shared_mat_offsets,
            shared_mat_count,
            current_raw_bank,
            it_number,
            condition_1,
            0x00,
            0x00,
            hits);
        }
      }
    }
  }
}
