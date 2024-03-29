/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <MEPTools.h>
#include <MuonPopulateTileAndTDC.cuh>
#include <BankTypes.h>

INSTANTIATE_ALGORITHM(muon_populate_tile_and_tdc::muon_populate_tile_and_tdc_t)

template<int decoding_version>
__global__ void muon_calculate_station_ocurrences_sizes(muon_populate_tile_and_tdc::Parameters parameters)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  const auto storage_station_region_quarter_offsets =
    parameters.dev_storage_station_region_quarter_offsets + event_number * Muon::Constants::n_layouts *
                                                              Muon::Constants::n_stations * Muon::Constants::n_regions *
                                                              Muon::Constants::n_quarters;
  const auto event_offset = storage_station_region_quarter_offsets[0];
  auto used = parameters.dev_muon_tile_used + event_offset;
  auto storage_tile_id = parameters.dev_storage_tile_id + event_offset;
  auto station_ocurrences_sizes = parameters.dev_station_ocurrences_sizes + event_number * Muon::Constants::n_stations;

  for (unsigned i = 0; i < Muon::Constants::n_stations * Muon::Constants::n_regions * Muon::Constants::n_quarters;
       i++) {

    // Note: The location of the indices depends on n_layouts.
    const auto start_index = storage_station_region_quarter_offsets[2 * i] - event_offset;
    const auto mid_index = storage_station_region_quarter_offsets[2 * i + 1] - event_offset;
    const auto end_index = storage_station_region_quarter_offsets[2 * i + 2] - event_offset;

    if (start_index == end_index) continue;
    const auto tile = Muon::MuonTileID(storage_tile_id[start_index]);
    const auto layout1 = getLayout(parameters.dev_muon_raw_to_hits->muonTables, tile)[0];
    const auto layout2 = getLayout(parameters.dev_muon_raw_to_hits->muonTables, tile)[1];
    bool pad = false;

    if constexpr (decoding_version == 3) {
      const auto station = tile.station();
      const auto region = tile.region();
      pad = (station == 0 && region > 1) || (station == 2 && region == 0) || (station == 3 && region == 0) ||
            (station == 3 && region == 3);

      if (pad && threadIdx.x == 0) {
        atomicAdd(station_ocurrences_sizes + station, 1);
        used[start_index] = true;
      }
    }

    if (!pad) {
      for (unsigned digitsOneIndex = start_index + threadIdx.x; digitsOneIndex < mid_index;
           digitsOneIndex += blockDim.x) {
        const unsigned int keyX =
          Muon::MuonTileID::nX(storage_tile_id[digitsOneIndex]) * layout2.xGrid() / layout1.xGrid();
        const unsigned int keyY = Muon::MuonTileID::nY(storage_tile_id[digitsOneIndex]);

        for (unsigned digitsTwoIndex = mid_index; digitsTwoIndex < end_index; digitsTwoIndex++) {
          const unsigned int candidateX = Muon::MuonTileID::nX(storage_tile_id[digitsTwoIndex]);
          const unsigned int candidateY =
            Muon::MuonTileID::nY(storage_tile_id[digitsTwoIndex]) * layout1.yGrid() / layout2.yGrid();

          if (keyX == candidateX && keyY == candidateY) {
            atomicAdd(station_ocurrences_sizes + tile.station(), 1);
            used[digitsOneIndex] = used[digitsTwoIndex] = true;
          }
        }
      }
    }

    __syncthreads(); // for used

    for (auto index = start_index + threadIdx.x; index < end_index; index += blockDim.x) {
      if (!used[index]) {
        atomicAdd(station_ocurrences_sizes + tile.station(), 1);
      }
    }
  }
}

template<int decoding_version>
__device__ void decode_muon_bank(
  Muon::MuonRawToHits const* muon_raw_to_hits,
  Muon::MuonRawBank<decoding_version> const& raw_bank,
  const unsigned* storage_station_region_quarter_offsets,
  unsigned* atomics_muon,
  unsigned* dev_storage_tile_id,
  unsigned* dev_storage_tdc_value)
{
  if constexpr (decoding_version == 2) {
    for (unsigned batch_index = threadIdx.y; batch_index < Muon::batches_per_bank; batch_index += blockDim.y) {
      const auto tell_number = raw_bank.sourceID;
      const uint16_t* p = raw_bank.data;

      p += (*p + 3) & 0xFFFE;
      for (unsigned j = 0; j < batch_index; ++j) {
        p += 1 + *p;
      }

      const auto batch_size = *p;
      for (int j = 1; j < batch_size + 1; ++j) {
        const auto pp = *(p + j);
        const auto add = (pp & 0x0FFF);
        const auto tdc_value = ((pp & 0xF000) >> 12);
        const auto tileId = muon_raw_to_hits->muonGeometry->getADDInTell1(tell_number, add);

        if (tileId != 0) {
          const auto tile = Muon::MuonTileID(tileId);
          const auto layout1 = getLayout(muon_raw_to_hits->muonTables, tile)[0];

          // Store tiles according to their station, region, quarter and layout,
          // to prepare data for easy process in muonaddcoordscrossingmaps.
          const auto storage_srq_layout =
            Muon::Constants::n_layouts * tile.stationRegionQuarter() + (tile.layout() != layout1);

          const auto insert_index = atomicAdd(atomics_muon + storage_srq_layout, 1);
          dev_storage_tile_id[storage_station_region_quarter_offsets[storage_srq_layout] + insert_index] = tileId;
          dev_storage_tdc_value[storage_station_region_quarter_offsets[storage_srq_layout] + insert_index] = tdc_value;
        }
      }
    }
  }
  else {
    const auto* p = raw_bank.data;
    unsigned synch_evt = (*p & 0x10) >> 4;
    if (synch_evt) return;
    const unsigned align_info = (*p & 0x20) >> 5;
    const unsigned link_start_pointer = align_info ? 3 : 0;

    const auto tell_pci = raw_bank.sourceID & 0x00FF;
    const auto tell_number = tell_pci / 2 + 1;
    const auto pci_number = tell_pci % 2;
    const auto tell_station = muon_raw_to_hits->muonGeometry->whichStationIsTell40(tell_number - 1);
    const auto active_links = muon_raw_to_hits->muonGeometry->NumberOfActiveLink(tell_number, pci_number);

    const Allen::device::span<const uint8_t> range8 {raw_bank.data, (raw_bank.last - raw_bank.data) / sizeof(uint8_t)};
    if (range8.empty()) return;

    const auto range_data = range8.subspan(1);
    if (range8.size() < 1 + 3 * align_info or range8.size() < active_links + 1) return;

    unsigned map_connected_fibers[24] = {};
    unsigned number_of_readout_fibers = muon_raw_to_hits->muonGeometry->get_number_of_readout_fibers(
      range8, active_links, map_connected_fibers, align_info);
    if (range8.size() < number_of_readout_fibers + 1 + 3 * align_info) return;

    bool corrupted = false;
    for (unsigned link = threadIdx.y; link < number_of_readout_fibers; link += blockDim.y) {
      unsigned current_pointer =
        raw_bank.type == LHCb::RawBank::BankType::MuonError ? link_start_pointer + 3 : link_start_pointer;

      auto size_of_link = (static_cast<unsigned>(range_data[current_pointer] & 0xF0) >> 4) + 1;
      for (unsigned j = 0; j < link; ++j) {
        current_pointer += size_of_link;
        if (current_pointer >= range_data.size()) {
          size_of_link = 0;
          corrupted = true;
          break;
        }
        else {
          size_of_link = (static_cast<unsigned>(range_data[current_pointer] & 0xF0) >> 4) + 1;
        }
      }
      if (
        size_of_link < 1 or (size_of_link > 1 and size_of_link <= 6) or
        (current_pointer + size_of_link) > range_data.size())
        corrupted = true;
    }
    if (corrupted) return;

    __syncthreads();
    for (unsigned link = threadIdx.y; link < number_of_readout_fibers; link += blockDim.y) {
      unsigned reroutered_link = map_connected_fibers[link];
      auto regionOfLink = muon_raw_to_hits->muonGeometry->RegionOfLink(tell_number, pci_number, reroutered_link);
      auto quarterOfLink = muon_raw_to_hits->muonGeometry->QuarterOfLink(tell_number, pci_number, reroutered_link);

      unsigned current_pointer =
        raw_bank.type == LHCb::RawBank::BankType::MuonError ? link_start_pointer + 3 : link_start_pointer;
      auto size_of_link = (static_cast<unsigned>(range_data[current_pointer] & 0xF0) >> 4) + 1;

      for (unsigned j = 0; j < link; ++j) {
        current_pointer += size_of_link;
        size_of_link = (static_cast<unsigned>(range_data[current_pointer] & 0xF0) >> 4) + 1;
      }

      if (size_of_link > 1) {
        auto range_link_HitsMap = range_data.subspan(current_pointer, 7);
        auto range_link_TDC = range_data.subspan(current_pointer + 6, size_of_link - 6);

        bool first_hitmap_byte = false;
        bool last_hitmap_byte = true;
        unsigned count_byte = 0;
        unsigned pos_in_link = 0;
        unsigned nSynch_hits_number = 0;
        unsigned TDC_counter = range_link_TDC.size() * 2 - 1;

        for (auto r = range_link_HitsMap.rbegin(); r < range_link_HitsMap.rend(); r++) {
          // loop in reverse mode hits map is 47->0
          count_byte++;
          if (count_byte == 7) first_hitmap_byte = true;
          if (count_byte > 7) break;
          for (unsigned bit_pos_1 = 8; bit_pos_1 > 0; --bit_pos_1) {
            unsigned bit_pos = bit_pos_1 - 1;

            if (first_hitmap_byte && bit_pos < 4) continue;
            if (last_hitmap_byte && bit_pos > 3) continue;
            if (*r & Muon::Constants::single_bit_position()[bit_pos]) {
              auto tileId =
                muon_raw_to_hits->muonGeometry->TileInTell40(tell_number, pci_number, reroutered_link, pos_in_link);

              if (tileId != 0) {
                const auto tile = Muon::MuonTileID(tileId);
                const auto layout1 = getLayout(muon_raw_to_hits->muonTables, tile)[0];

                unsigned tdc_value = 0;
                if (nSynch_hits_number < TDC_counter) {
                  if (nSynch_hits_number == 0)
                    tdc_value = range_link_TDC[0] & 0x0F;
                  else {
                    auto mask = nSynch_hits_number % 2 == 0 ? 0x0F : 0xF0;
                    auto shift = nSynch_hits_number % 2 == 0 ? 0 : 4;
                    tdc_value = range_link_TDC[1 + (nSynch_hits_number - 1) / 2] & mask >> shift;
                  }
                }
                nSynch_hits_number++;

                // Store tiles according to their station, region, quarter and layout,
                // to prepare data for easy process in muonaddcoordscrossingmaps.
                if (tell_station * 16 + regionOfLink * 4 + quarterOfLink < 64) {
                  const auto storage_srq_layout =
                    Muon::Constants::n_layouts * tile.stationRegionQuarter() + (tile.layout() != layout1);
                  const auto insert_index = atomicAdd(atomics_muon + storage_srq_layout, 1);
                  dev_storage_tile_id[storage_station_region_quarter_offsets[storage_srq_layout] + insert_index] =
                    tileId;
                  dev_storage_tdc_value[storage_station_region_quarter_offsets[storage_srq_layout] + insert_index] =
                    tdc_value;
                }
              }
            }
            pos_in_link++;
          }
          last_hitmap_byte = false;
        }
      }
    }
  }
}

template<int decoding_version, bool mep_layout>
__global__ void muon_populate_tile_and_tdc_kernel(
  muon_populate_tile_and_tdc::Parameters parameters,
  const unsigned event_start)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const auto storage_station_region_quarter_offsets =
    parameters.dev_storage_station_region_quarter_offsets +
    event_number * 2 * Muon::Constants::n_stations * Muon::Constants::n_regions * Muon::Constants::n_quarters;
  unsigned* atomics_muon = parameters.dev_atomics_muon + event_number * 2 * Muon::Constants::n_stations *
                                                           Muon::Constants::n_regions * Muon::Constants::n_quarters;

  const auto raw_event = Muon::RawEvent<mep_layout, decoding_version> {parameters.dev_muon_raw,
                                                                       parameters.dev_muon_raw_offsets,
                                                                       parameters.dev_muon_raw_sizes,
                                                                       parameters.dev_muon_raw_types,
                                                                       event_number + event_start};

  for (unsigned bank_index = threadIdx.x; bank_index < raw_event.number_of_raw_banks(); bank_index += blockDim.x) {
    const auto raw_bank = raw_event.raw_bank(bank_index);

    if (raw_bank.type != LHCb::RawBank::BankType::Muon && raw_bank.type != LHCb::RawBank::BankType::MuonError)
      continue; // skip invalid raw banks

    decode_muon_bank<decoding_version>(
      parameters.dev_muon_raw_to_hits,
      raw_bank,
      storage_station_region_quarter_offsets,
      atomics_muon,
      parameters.dev_storage_tile_id,
      parameters.dev_storage_tdc_value);
  }
}

void muon_populate_tile_and_tdc::muon_populate_tile_and_tdc_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_storage_tile_id_t>(arguments, first<host_muon_total_number_of_tiles_t>(arguments));
  set_size<dev_storage_tdc_value_t>(arguments, first<host_muon_total_number_of_tiles_t>(arguments));
  set_size<dev_atomics_muon_t>(
    arguments,
    first<host_number_of_events_t>(arguments) * 2 * Muon::Constants::n_stations * Muon::Constants::n_regions *
      Muon::Constants::n_quarters);
  set_size<dev_muon_tile_used_t>(arguments, first<host_muon_total_number_of_tiles_t>(arguments));
  set_size<dev_station_ocurrences_sizes_t>(
    arguments, first<host_number_of_events_t>(arguments) * Muon::Constants::n_stations);
}

void muon_populate_tile_and_tdc::muon_populate_tile_and_tdc_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_atomics_muon_t>(arguments, 0, context);
  Allen::memset_async<dev_storage_tile_id_t>(arguments, 0, context);
  Allen::memset_async<dev_storage_tdc_value_t>(arguments, 0, context);
  Allen::memset_async<dev_muon_tile_used_t>(arguments, 0, context);
  Allen::memset_async<dev_station_ocurrences_sizes_t>(arguments, 0, context);

  const auto bank_version = first<host_raw_bank_version_t>(arguments);
  if (bank_version < 0) return; // no Muon banks present in data
  auto populate_tile_and_tdc_kernel = bank_version == 2 ?
                                        (runtime_options.mep_layout ? muon_populate_tile_and_tdc_kernel<2, true> :
                                                                      muon_populate_tile_and_tdc_kernel<2, false>) :
                                        (runtime_options.mep_layout ? muon_populate_tile_and_tdc_kernel<3, true> :
                                                                      muon_populate_tile_and_tdc_kernel<3, false>);

  global_function(populate_tile_and_tdc_kernel)(size<dev_event_list_t>(arguments), dim3(64, 4), context)(
    arguments, std::get<0>(runtime_options.event_interval));

  auto calculate_station_ocurrences_kernel =
    bank_version == 2 ? muon_calculate_station_ocurrences_sizes<2> : muon_calculate_station_ocurrences_sizes<3>;

  global_function(calculate_station_ocurrences_kernel)(dim3(size<dev_event_list_t>(arguments)), dim3(64), context)(
    arguments);
}
