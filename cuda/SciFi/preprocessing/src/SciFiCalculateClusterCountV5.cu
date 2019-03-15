#include "SciFiCalculateClusterCountV5.cuh"

using namespace SciFi;

__global__ void scifi_calculate_cluster_count_v5(
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets,
  const uint* event_list,
  uint* scifi_hit_count,
  char* scifi_geometry)
{
  const uint event_number = blockIdx.x;
  const uint selected_event_number = event_list[event_number];

  const SciFiRawEvent event(scifi_raw_input + scifi_raw_input_offsets[selected_event_number]);
  const SciFiGeometry geom(scifi_geometry);
  SciFi::HitCount hit_count {scifi_hit_count, event_number};

  // NO version checking. Be careful, as v5 is assumed.

  for(uint i = threadIdx.x; i < event.number_of_raw_banks; i += blockDim.x) {
    const uint k = i % 10;
    const bool reverse_raw_bank_order = k < 5;
    const uint current_raw_bank = reverse_raw_bank_order ?
      5 * (i / 5) + (4 - i % 5) :
      i;    

    uint32_t* hits_module;
    const auto rawbank = event.getSciFiRawBank(current_raw_bank);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;

    // For details see RawBankDecoder
    if (*(last - 1) == 0) --last; // Remove phadding at the end
    for (; it < last; ++it) {     // loop over the clusters
      uint16_t c = *it;
      uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
      if(current_raw_bank < SciFi::Constants::n_consecutive_raw_banks)
        hits_module = hit_count.mat_offsets + i;
      else
        hits_module = hit_count.mat_offsets + SciFiChannelID(ch).correctedUniqueMat() - SciFi::Constants::mat_index_substract;
      if( !cSize(c) || it+1 == last ) { //No size flag or last cluster
        atomicAdd(hits_module, 1);
      } else { //Flagged or not the last one.
        unsigned c2 = *(it+1);
        if( cSize(c2) && getLinkInBank(c) == getLinkInBank(c2) ) {
          unsigned int delta = (cell(c2) - cell(c));
          atomicAdd(hits_module, 1 + (delta - 1) / SciFiRawBankParams::clusterMaxWidth);
          ++it;
        }
        else {
          atomicAdd(hits_module, 1);
        }
      }
    }
  }
}
