#include <iostream>
#include <iomanip>

#include "CpuGlobalEventCut.cuh"
#include "SciFiRaw.cuh"
#include "UTRaw.cuh"
#include "ArgumentsCommon.cuh"
#include "GlobalEventCutConfiguration.cuh"

void cpu_global_event_cut(
  char const* ut_raw_input,
  uint const* ut_raw_input_offsets,
  char const* scifi_raw_input,
  uint const* scifi_raw_input_offsets,
  uint* number_of_selected_events,
  uint* event_list,
  uint number_of_events)
{
  uint insert_index = 0;
  for (uint event_number = 0; event_number < number_of_events; ++event_number) {
    // Check SciFi clusters
    const SciFi::SciFiRawEvent scifi_event(scifi_raw_input + scifi_raw_input_offsets[event_number]);
    uint n_SciFi_clusters = 0;

    for (uint i = 0; i < scifi_event.number_of_raw_banks; ++i) {
      // get bank size in bytes, subtract four bytes for header word
      uint bank_size = scifi_event.raw_bank_offset[i + 1] - scifi_event.raw_bank_offset[i] - 4;
      n_SciFi_clusters += bank_size;
    }

    // Bank size is given in bytes. There are 2 bytes per cluster.
    // 4 bytes are removed for the header.
    // Note that this overestimates slightly the number of clusters
    // due to bank padding in 32b. For v5, it further overestimates the
    // number of clusters due to the merging of clusters.
    n_SciFi_clusters = (n_SciFi_clusters >> 1) - 2;

    // Check UT clusters
    const uint32_t ut_event_offset = ut_raw_input_offsets[event_number];
    const UTRawEvent ut_event(ut_raw_input + ut_event_offset);
    uint n_UT_clusters = 0;

    for (uint i = 0; i < ut_event.number_of_raw_banks; ++i) {
      const UTRawBank ut_bank = ut_event.getUTRawBank(i);
      n_UT_clusters += ut_bank.number_of_hits;
    }

    const auto num_combined_clusters = n_UT_clusters + n_SciFi_clusters;
    if (num_combined_clusters < max_scifi_ut_clusters && num_combined_clusters > min_scifi_ut_clusters) {
      event_list[insert_index++] = event_number;
    }
  }

  number_of_selected_events[0] = insert_index;
}

void cpu_global_event_cut_mep(
  BanksAndOffsets const& ut_raw,
  BanksAndOffsets const& scifi_raw,
  uint* number_of_selected_events,
  uint* event_list,
  uint number_of_events)
{
  uint insert_index = 0;
  for (uint event_number = 0; event_number < number_of_events; ++event_number) {
    // Check SciFi clusters

    auto const& scifi_offsets = std::get<2>(scifi_raw);
    auto const number_of_scifi_raw_banks = scifi_offsets[0];
    uint n_SciFi_clusters = 0;

    for (uint i = 0; i < number_of_scifi_raw_banks; ++i) {
      uint const offset_index = 2 + number_of_scifi_raw_banks * (1 + event_number);
      uint bank_size = scifi_offsets[offset_index + i + number_of_scifi_raw_banks] - scifi_offsets[offset_index + i];
      // std::cout << "scifi " << std::setw(4) << event_number << " " << std::setw(3) << i << " " << bank_size << "\n";
      n_SciFi_clusters += bank_size;
    }

    // Bank size is given in bytes. There are 2 bytes per cluster.
    // 4 bytes are removed for the header.
    // Note that this overestimates slightly the number of clusters
    // due to bank padding in 32b. For v5, it further overestimates the
    // number of clusters due to the merging of clusters.
    n_SciFi_clusters = (n_SciFi_clusters >> 1) - 2;

    // Check UT clusters
    auto const& ut_data = std::get<0>(ut_raw);
    auto const& ut_offsets = std::get<2>(ut_raw);
    auto const number_of_ut_raw_banks = ut_offsets[0];
    uint n_UT_clusters = 0;

    for (uint i = 0; i < number_of_ut_raw_banks; ++i) {
      auto sourceID = ut_offsets[2 + i];
      // We're on the host, so use the blocks directly
      auto block_offset = ut_offsets[2 + number_of_ut_raw_banks + i];
      auto const fragment_offset = ut_offsets[2 + number_of_ut_raw_banks * (1 + event_number) + i] - block_offset;
      const UTRawBank ut_bank{sourceID, ut_data[i].begin() + fragment_offset};
      n_UT_clusters += ut_bank.number_of_hits;
    }

    const auto num_combined_clusters = n_UT_clusters + n_SciFi_clusters;
    if (num_combined_clusters < max_scifi_ut_clusters && num_combined_clusters > min_scifi_ut_clusters) {
      event_list[insert_index++] = event_number;
    }
  }

  number_of_selected_events[0] = insert_index;
}
