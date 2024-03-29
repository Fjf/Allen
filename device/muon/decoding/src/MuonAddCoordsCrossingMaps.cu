/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "MuonAddCoordsCrossingMaps.cuh"

INSTANTIATE_ALGORITHM(muon_add_coords_crossing_maps::muon_add_coords_crossing_maps_t)

template<int decoding_version>
__global__ void muon_add_coords_crossing_maps_kernel(muon_add_coords_crossing_maps::Parameters parameters)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  const auto storage_station_region_quarter_offsets =
    parameters.dev_storage_station_region_quarter_offsets + event_number * Muon::Constants::n_layouts *
                                                              Muon::Constants::n_stations * Muon::Constants::n_regions *
                                                              Muon::Constants::n_quarters;
  const auto event_offset = storage_station_region_quarter_offsets[0];

  auto current_hit_index = parameters.dev_atomics_index_insert + event_number;
  auto used = parameters.dev_muon_tile_used + event_offset;
  auto storage_tile_id = parameters.dev_storage_tile_id + event_offset;
  auto station_ocurrences_offsets =
    parameters.dev_station_ocurrences_offset + event_number * Muon::Constants::n_stations;
  auto muon_compact_hit = parameters.dev_muon_compact_hit + station_ocurrences_offsets[0];

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
        const int localCurrentHitIndex = atomicAdd(current_hit_index, 1);

        const uint64_t compact_hit =
          (((uint64_t)(start_index & 0x7FFF)) << 48) | (((uint64_t)(start_index & 0xFFFF)) << 32) |
          ((layout1.xGrid() & 0x3FFF) << 18) | ((layout2.yGrid() & 0x3FFF) << 4) |
          (((tile.id() & Muon::MuonBase::MaskStation) >> Muon::MuonBase::ShiftStation) & 0xF);

        muon_compact_hit[localCurrentHitIndex] = compact_hit;
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
            Muon::MuonTileID padTile(storage_tile_id[digitsOneIndex]);
            const int localCurrentHitIndex = atomicAdd(current_hit_index, 1);

            const uint64_t compact_hit =
              (((uint64_t)(digitsOneIndex & 0x7FFF)) << 48) | (((uint64_t)(digitsTwoIndex & 0xFFFF)) << 32) |
              ((layout1.xGrid() & 0x3FFF) << 18) | ((layout2.yGrid() & 0x3FFF) << 4) |
              (((padTile.id() & Muon::MuonBase::MaskStation) >> Muon::MuonBase::ShiftStation) & 0xF);

            muon_compact_hit[localCurrentHitIndex] = compact_hit;
          }
        }
      }
    }

    for (auto index = start_index + threadIdx.x; index < end_index; index += blockDim.x) {
      if (!used[index]) {
        const auto tile = Muon::MuonTileID(storage_tile_id[index]);

        int condition;
        if (tile.station() > (Muon::Constants::n_stations - 3) && tile.region() == 0) {
          condition = 0;
        }
        else {
          const auto hit_layout = Muon::MuonTileID::layout(storage_tile_id[index]);
          condition = (hit_layout == layout1 ? 1 : 2);
          // assert(hit_layout == layout1 || hit_layout == layout2);
        }

        const int localCurrentHitIndex = atomicAdd(current_hit_index, 1);

        const unsigned int uncrossed = 1;
        const uint64_t compact_hit =
          (((uint64_t)(uncrossed & 0x1)) << 63) | (((uint64_t)(index & 0x7FFF)) << 48) | (condition << 4) |
          (((tile.id() & Muon::MuonBase::MaskStation) >> Muon::MuonBase::ShiftStation) & 0xF);

        muon_compact_hit[localCurrentHitIndex] = compact_hit;
      }
    }
  }
}

void muon_add_coords_crossing_maps::muon_add_coords_crossing_maps_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_muon_compact_hit_t>(arguments, first<host_muon_total_number_of_hits_t>(arguments));
  set_size<dev_atomics_index_insert_t>(arguments, first<host_number_of_events_t>(arguments));
}

void muon_add_coords_crossing_maps::muon_add_coords_crossing_maps_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_muon_compact_hit_t>(arguments, 0, context);
  Allen::memset_async<dev_atomics_index_insert_t>(arguments, 0, context);

  const auto bank_version = first<host_raw_bank_version_t>(arguments);
  if (bank_version < 0) return; // no Muon banks present in data
  auto kernel_fn =
    bank_version == 2 ? muon_add_coords_crossing_maps_kernel<2> : muon_add_coords_crossing_maps_kernel<3>;

  global_function(kernel_fn)(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(arguments);
}
