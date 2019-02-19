#include "SciFiCalculateClusterCountV6.cuh"

using namespace SciFi;

__global__ void scifi_calculate_cluster_count_v6(
  char* scifi_raw_input,
  uint* scifi_raw_input_offsets,
  const uint* event_list,
  uint* scifi_hit_count,
  char* scifi_geometry
) {
  const uint event_number = blockIdx.x;
  const uint selected_event_number = event_list[event_number];

  const SciFiRawEvent event(scifi_raw_input + scifi_raw_input_offsets[selected_event_number]); 
  const SciFiGeometry geom(scifi_geometry);
  SciFi::HitCount hit_count {scifi_hit_count, event_number};

  // NO version checking. Be careful, as v6 is assumed.

  for(uint i = threadIdx.x; i < event.number_of_raw_banks; i += blockDim.x)
  {
    uint32_t* hits_mat;
    const auto rawbank = event.getSciFiRawBank(i); 
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;

    //For details see RawBankDecoder
    if (*(last-1) == 0) --last; //Remove padding at the end
    for( ;  it < last; ++it ){ // loop over the clusters
      uint16_t c = *it;
      uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
      if(i < SciFi::Constants::n_consecutive_raw_banks)
        hits_mat = hit_count.mat_offsets + i;
      else
        hits_mat = hit_count.mat_offsets + SciFiChannelID(ch).correctedUniqueMat() - SciFi::Constants::mat_index_substract;
      if( !cSize(c) ) {  //Not flagged as large
        atomicAdd(hits_mat, 1);
      } else if( fraction(c) ) { // flagged as first edge of large cluster
        // last cluster in bank or in sipm
        if(  it+1 == last || getLinkInBank(c) != getLinkInBank( *(it+1)) ) 
          atomicAdd(hits_mat, 1);
        else {
          unsigned c2 = *(it+1);
          assert( cSize(c2) && !fraction(c2) );
          unsigned int widthClus = (cell(c2) - cell(c) + 2);
          if ( widthClus  > 8 )
            // number of for loop passes + one additional
            atomicAdd(hits_mat, widthClus / 4 + 1);
          else
            atomicAdd(hits_mat, 1);
          ++it;
        }
      }
    }
  }
}
