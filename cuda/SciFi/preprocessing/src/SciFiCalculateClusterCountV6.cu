#include "SciFiCalculateClusterCountV6.cuh"

void scifi_calculate_cluster_count_v6_t::set_arguments_size(
  ArgumentRefManager<Arguments> arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers) const
{
  arguments.set_size<dev_scifi_hit_count>(
    2 * host_buffers.host_number_of_selected_events[0] * SciFi::Constants::n_mats + 1);
}

void scifi_calculate_cluster_count_v6_t::operator()(
  const ArgumentRefManager<Arguments>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event) const
{
  cudaCheck(
    cudaMemsetAsync(arguments.offset<dev_scifi_hit_count>(), 0, arguments.size<dev_scifi_hit_count>(), cuda_stream));

  function.invoke(dim3(host_buffers.host_number_of_selected_events[0]), dim3(SciFi::SciFiRawBankParams::NbBanks), cuda_stream)(
    arguments.offset<dev_scifi_raw_input>(),
    arguments.offset<dev_scifi_raw_input_offsets>(),
    arguments.offset<dev_event_list>(),
    arguments.offset<dev_scifi_hit_count>(),
    constants.dev_scifi_geometry);
}

using namespace SciFi;

__global__ void scifi_calculate_cluster_count_v6(
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

  // NO version checking. Be careful, as v6 is assumed.

  for (uint i = threadIdx.x; i < event.number_of_raw_banks; i += blockDim.x) {
    const uint current_raw_bank = getRawBankIndexOrderedByX(i);
    uint32_t* hits_module;
    const auto rawbank = event.getSciFiRawBank(current_raw_bank);
    uint16_t* it = rawbank.data + 2;
    uint16_t* last = rawbank.last;

    // For details see RawBankDecoder
    if (*(last - 1) == 0) --last; // Remove padding at the end
    for (; it < last; ++it) {     // loop over the clusters
      uint16_t c = *it;
      uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + channelInBank(c);
      if (current_raw_bank < SciFi::Constants::n_consecutive_raw_banks)
        hits_module = hit_count.mat_offsets + i;
      else
        hits_module =
          hit_count.mat_offsets + SciFiChannelID(ch).correctedUniqueMat() - SciFi::Constants::mat_index_substract;
      if (!cSize(c)) { // Not flagged as large
        atomicAdd(hits_module, 1);
      }
      else if (fraction(c)) { // flagged as first edge of large cluster
        // last cluster in bank or in sipm
        if (it + 1 == last || getLinkInBank(c) != getLinkInBank(*(it + 1)))
          atomicAdd(hits_module, 1);
        else {
          unsigned c2 = *(it + 1);
          assert(cSize(c2) && !fraction(c2));
          unsigned int widthClus = (cell(c2) - cell(c) + 2);
          if (widthClus > 8)
            // number of for loop passes in decoder + one additional
            atomicAdd(hits_module, (widthClus - 1) / 4 + 1);
          else
            atomicAdd(hits_module, 1);
          ++it;
        }
      }
    }
  }
}
