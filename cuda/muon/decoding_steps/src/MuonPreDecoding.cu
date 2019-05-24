#include "MuonPreDecoding.cuh"

__global__ void muon_pre_decoding(
  const uint* event_list,
  const char* events,
  const unsigned int* offsets,
  const Muon::MuonRawToHits* muon_raw_to_hits,
  uint* dev_storage_station_region_quarter_offsets,
  uint* dev_storage_tile_id,
  uint* dev_storage_tdc_value,
  uint* dev_atomics_muon)
{
  const size_t event_number = blockIdx.x;
  const size_t event_id = event_list[blockIdx.x];
  const auto raw_event = Muon::MuonRawEvent(events + offsets[event_id]);
  auto storage_station_region_quarter_offsets = dev_storage_station_region_quarter_offsets +
    event_number * Muon::Constants::n_stations * Muon::Constants::n_regions * Muon::Constants::n_quarters;
  auto storage_tile_id = dev_storage_tile_id + event_number * Muon::Constants::max_numhits_per_event;
  auto storage_tdc_value = dev_storage_tdc_value + event_number * Muon::Constants::max_numhits_per_event;

  // number_of_raw_banks = 10
  // batches_per_bank = 4
  constexpr uint32_t batches_per_bank_mask = 0x3;
  constexpr uint32_t batches_per_bank_shift = 2;
  for (int i = threadIdx.x; i < Muon::MuonRawEvent::number_of_raw_banks * Muon::MuonRawEvent::batches_per_bank; i += blockDim.x) {
    const auto bank_index = i >> batches_per_bank_shift;
    const auto batch_index = i & batches_per_bank_mask;

    const auto raw_bank = raw_event.getMuonBank(bank_index);
    const auto tell_number = raw_bank.sourceID;

    uint16_t* p = raw_bank.data;

    // Note: Review this logic
    p += (*p + 3) & 0xFFFE;
    for (int j = 0; j < batch_index; ++j) {
      p += 1 + *p;
    }

    const auto batch_size = *p;
    for (int j = 1; j < batch_size + 1; ++j) {
      const auto pp = *(p + j);
      const auto add = (pp & 0x0FFF);
      const auto tdc_value = ((pp & 0xF000) >> 12);
      const auto tileId = muon_raw_to_hits->muonGeometry->getADDInTell1(tell_number, add);

      if (tileId != 0) {
        const auto insert_index = atomicAdd(dev_atomics_muon + event_number, 1);
        storage_tile_id[insert_index] = tileId;
        storage_tdc_value[insert_index] = tdc_value;

        // Also add to storageStationRegionQuarterOccurrencesOffset
        const auto stationRegionQuarter = Muon::MuonTileID::stationRegionQuarter(tileId);
        atomicAdd(storage_station_region_quarter_offsets + stationRegionQuarter, 1);
      }
    }
  }
}
