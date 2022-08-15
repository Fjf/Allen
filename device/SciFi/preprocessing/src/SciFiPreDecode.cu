/*****************************************************************************\
 * (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "SciFiPreDecode.cuh"
#include <MEPTools.h>
#include "assert.h"

INSTANTIATE_ALGORITHM(scifi_pre_decode::scifi_pre_decode_t)

__device__ void store_sorted_cluster_reference(
  SciFi::ConstHitCount& hit_count,
  const uint32_t uniqueMat,
  const uint32_t chan,
  const uint32_t mat_offset,
  uint32_t& mat_count,
  const int raw_bank,
  const int it,
  uint32_t* cluster_references,
  const int condition,
  const int delta)
{
  uint32_t hitIndex = mat_count++;

  const SciFi::SciFiChannelID id {chan};
  if (id.reversedZone()) {
    hitIndex = hit_count.mat_group_or_mat_number_of_hits(uniqueMat) - 1 - hitIndex;
  }
  assert(hitIndex < hit_count.mat_group_or_mat_number_of_hits(uniqueMat));
  assert(uniqueMat < SciFi::Constants::n_mat_groups_and_mats);
  hitIndex += mat_offset;

  cluster_references[hitIndex] =
    (raw_bank & 0xFF) << 24 | (it & 0xFF) << 16 | (condition & 0x07) << 13 | (delta & 0xFF);
}

template<typename F>
__device__ void insertionSort(uint32_t* arr, int n, F&& f)
{
  for (int i = 1; i < n; i++) {
    uint32_t key = arr[i];
    int j = i - 1;
    while (j >= 0 && f(arr[j]) > f(key)) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = key;
  }
}

template<int decoding_version, bool mep_layout>
__global__ void scifi_pre_decode_kernel(scifi_pre_decode::Parameters parameters, const char* scifi_geometry)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  SciFi::SciFiGeometry geom(scifi_geometry);
  const auto scifi_raw_event = SciFi::RawEvent<mep_layout>(
    parameters.dev_scifi_raw_input,
    parameters.dev_scifi_raw_input_offsets,
    parameters.dev_scifi_raw_input_sizes,
    event_number);

  SciFi::ConstHitCount hit_count {parameters.dev_scifi_hit_offsets, event_number};

  // Main execution loop
  for (unsigned iRawBank = threadIdx.x; iRawBank < scifi_raw_event.number_of_raw_banks(); iRawBank += blockDim.x) {
    // In decoding < 7, (continuous index) -> (x ordered continuous index)
    // Could be a lookup table where index is i (x ordered index) and value is the source ID. lookup[i] is the sourceID
    // of the i-th bank, x ordered
    auto rawbank = scifi_raw_event.raw_bank(iRawBank);
    const auto sourceID = rawbank.sourceID;
    const auto iRowInMap = iSource(geom, sourceID);
    if (iRowInMap == geom.number_of_banks)
      continue; // FIXME: means the source ID is unknown. This should have been caught by the PreDecode error
    uint16_t const* starting_it = rawbank.data;
    uint16_t const* last = rawbank.last;
    // Skip empty raw banks: very unlikely, as there should always be a header. But it has been seen in early data
    // taking.
    if (starting_it == last) continue;
    starting_it += 2;                                    // skip header
    if (starting_it != last && *(last - 1) == 0) --last; // Remove padding at the end
    if (starting_it >= last || starting_it >= rawbank.last) continue;

    const unsigned number_of_iterations = last - starting_it;
    int last_uniqueMat = -1;
    unsigned mat_offset = 0;
    unsigned mat_count = 0;
    [[maybe_unused]] bool reversedZone = false;
    for (unsigned it_number = 0; it_number < number_of_iterations; ++it_number) {
      auto it = starting_it + it_number;
      const uint16_t c = *it;
      uint32_t ch;
      if constexpr (decoding_version == 4 || decoding_version == 6) {
        ch = geom.bank_first_channel[iRawBank] + SciFi::channelInBank(c); //---LoH: only works for versions 4-6
      }
      else { // Decoding v7
        ch = SciFi::getGlobalSiPMFromIndex(geom, iRowInMap, c) + SciFi::channelInLink(c);
      }
      const auto chid = SciFi::SciFiChannelID(ch);
      uint32_t correctedMat = chid.globalMatIdx_Xorder();

      // adaptation to hybrid decoding
      if (correctedMat < SciFi::Constants::n_consecutive_raw_banks * SciFi::Constants::n_mats_per_consec_raw_bank)
        correctedMat = correctedMat / SciFi::Constants::n_mats_per_consec_raw_bank;
      else
        correctedMat = correctedMat - SciFi::Constants::mat_index_substract;

      if ((int) correctedMat != last_uniqueMat) {
        if constexpr (decoding_version == 7) {
          if (last_uniqueMat != -1) {
            // array is small and already almost sorted, using an insertion sort is enough
            if (reversedZone) {
              insertionSort(parameters.dev_cluster_references + mat_offset, mat_count, [&](uint32_t val) {
                const uint16_t c = starting_it[((val >> 16) & 0xff)];
                return -(SciFi::getGlobalSiPMFromIndex(geom, iRowInMap, c) + SciFi::channelInLink(c));
              });
            }
            else {
              insertionSort(parameters.dev_cluster_references + mat_offset, mat_count, [&](uint32_t val) {
                const uint16_t c = starting_it[((val >> 16) & 0xff)];
                return (SciFi::getGlobalSiPMFromIndex(geom, iRowInMap, c) + SciFi::channelInLink(c));
              });
            }
          }
        }
        last_uniqueMat = correctedMat;
        mat_offset = *hit_count.mat_offsets_p(correctedMat);
        mat_count = 0;
        reversedZone = chid.reversedZone();
      }
      const auto store_sorted_fn = [&](const int condition, const int delta) {
        store_sorted_cluster_reference(
          hit_count,
          correctedMat,
          ch,
          mat_offset,
          mat_count,
          iRawBank,
          it_number,
          parameters.dev_cluster_references,
          condition,
          delta);
      };

      if constexpr (decoding_version == 4) {
        store_sorted_fn(0x01, 0x00);
      }
      else if constexpr (decoding_version == 6 || decoding_version == 7) {
        if (!SciFi::cSize(c)) {
          // Single cluster
          store_sorted_fn(0x01, 0x00);
        }
        else {
          const unsigned c2 = *(it + 1);
          if (it + 1 == last || SciFi::getLinkInBank(c) != SciFi::getLinkInBank(c2))
            // last cluster in bank or in sipm
            store_sorted_fn(0x02, 0x00);
          else if (SciFi::fraction(c) && SciFi::cSize(c2) && !SciFi::fraction(c2)) {
            const unsigned int widthClus = (SciFi::cell(c2) - SciFi::cell(c) + 2);
            if (widthClus > 8) {
              uint16_t j = 0;
              for (; j < widthClus - 4; j += 4)
                // big cluster(s)
                store_sorted_fn(0x03, j);
              // add the last edge
              store_sorted_fn(0x04, j);
            }
            else
              store_sorted_fn(0x05, 0x00);
            ++it_number;
          }
          else { /* FIXME: Corrupt cluster */
          }
        }
      }
    }
    if constexpr (decoding_version == 7) {
      if (last_uniqueMat != -1) {
        // array is small and already almost sorted, using an insertion sort is enough
        if (reversedZone) {
          insertionSort(parameters.dev_cluster_references + mat_offset, mat_count, [&](uint32_t val) {
            const uint16_t c = starting_it[((val >> 16) & 0xff)];
            return -(SciFi::getGlobalSiPMFromIndex(geom, iRowInMap, c) + SciFi::channelInLink(c));
          });
        }
        else {
          insertionSort(parameters.dev_cluster_references + mat_offset, mat_count, [&](uint32_t val) {
            const uint16_t c = starting_it[((val >> 16) & 0xff)];
            return (SciFi::getGlobalSiPMFromIndex(geom, iRowInMap, c) + SciFi::channelInLink(c));
          });
        }
      }
    }
  }
}

void scifi_pre_decode::scifi_pre_decode_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  // Ensure the bank version is supported
  unsigned int const bank_version = first<host_raw_bank_version_t>(arguments);
  if (bank_version < 4 || bank_version > 7) {
    throw StrException("SciFi bank version not supported (" + std::to_string(bank_version) + ")");
  }

  set_size<dev_cluster_references_t>(
    arguments, first<host_accumulated_number_of_scifi_hits_t>(arguments) * SciFi::Hits::number_of_arrays);
}

void scifi_pre_decode::scifi_pre_decode_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers&,
  const Allen::Context& context) const
{
  unsigned int const bank_version = first<host_raw_bank_version_t>(arguments);

  // Mapping is:
  // * Version 4, version 5: Use v4 decoding
  // * Version 6: Use v6 decoding
  auto kernel_fn = (bank_version == 4 || bank_version == 5) ?
                     (runtime_options.mep_layout ? global_function(scifi_pre_decode_kernel<4, true>) :
                                                   global_function(scifi_pre_decode_kernel<4, false>)) :
                     (bank_version == 6) ?
                     (runtime_options.mep_layout ? global_function(scifi_pre_decode_kernel<6, true>) :
                                                   global_function(scifi_pre_decode_kernel<6, false>)) :
                     (runtime_options.mep_layout ? global_function(scifi_pre_decode_kernel<7, true>) :
                                                   global_function(scifi_pre_decode_kernel<7, false>));

  kernel_fn(dim3(size<dev_event_list_t>(arguments)), dim3(SciFi::SciFiRawBankParams::NbBanks), context)(
    arguments, constants.dev_scifi_geometry);
}
