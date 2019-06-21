#include "MuonAddCoordsCrossingMaps.cuh"

__global__ void muon_add_coords_crossing_maps(
  uint* dev_storage_station_region_quarter_offsets,
  uint* dev_storage_tile_id,
  uint* dev_storage_tdc_value,
  uint* dev_atomics_muon,
  uint* dev_permutation_srq,
  Muon::MuonRawToHits* muon_raw_to_hits,
  uint64_t* dev_muon_compact_hit,
  uint* dev_station_ocurrences_offset)
{
  const auto number_of_events = gridDim.x;
  const auto event_number = blockIdx.x;

  __shared__ bool used[Muon::Constants::max_numhits_per_event];
  for (int i = threadIdx.x; i < Muon::Constants::max_numhits_per_event; i += blockDim.x) {
    used[i] = false;
  }

  __syncthreads();

  auto muon_compact_hit = dev_muon_compact_hit + event_number * Muon::Constants::max_numhits_per_event;
  auto storage_tile_id = dev_storage_tile_id + event_number * Muon::Constants::max_numhits_per_event;
  auto storage_tdc_value = dev_storage_tdc_value + event_number * Muon::Constants::max_numhits_per_event;
  auto current_hit_index = dev_atomics_muon + number_of_events + event_number;
  auto storage_station_region_quarter_offsets =
    dev_storage_station_region_quarter_offsets +
    event_number * Muon::Constants::n_stations * Muon::Constants::n_regions * Muon::Constants::n_quarters;
  auto station_ocurrences_offset = dev_station_ocurrences_offset + event_number * Muon::Constants::n_stations;
  const auto base_offset = storage_station_region_quarter_offsets[0];

  for (int i = threadIdx.x; i < Muon::Constants::n_stations * Muon::Constants::n_regions * Muon::Constants::n_quarters;
       i += blockDim.x) {

    const auto start_index = storage_station_region_quarter_offsets[i] - base_offset;
    const auto end_index = storage_station_region_quarter_offsets[i + 1] - base_offset;

    if (start_index != end_index) {

      // muon_raw_to_hits->addCoordsCrossingMap(storage_tile_id, storage_tdc_value, used,
      //   start_index, end_index, event_muon_hits, *current_hit_index);

      // TODO: We are fetching the first tile ID
      //       We should verify this logic holds (it does not atm)
      const auto tile = Muon::MuonTileID(storage_tile_id[start_index]);
      const auto station = tile.station();
      const auto region = tile.region();

      const auto x1 = getLayoutX(muon_raw_to_hits->muonTables, Muon::MuonTables::stripXTableNumber, station, region);
      const auto y1 = getLayoutY(muon_raw_to_hits->muonTables, Muon::MuonTables::stripXTableNumber, station, region);
      const auto x2 = getLayoutX(muon_raw_to_hits->muonTables, Muon::MuonTables::stripYTableNumber, station, region);
      const auto y2 = getLayoutY(muon_raw_to_hits->muonTables, Muon::MuonTables::stripYTableNumber, station, region);

      Muon::MuonLayout layout_one;
      Muon::MuonLayout layout_two;
      if (x1 > x2) {
        layout_one = Muon::MuonLayout {x1, y1};
        layout_two = Muon::MuonLayout {x2, y2};
      }
      else {
        layout_one = Muon::MuonLayout {x2, y2};
        layout_two = Muon::MuonLayout {x1, y1};
      }

      uint mid_index = start_index;
      unsigned int tmp;
      for (uint j = start_index; j < end_index; ++j) {
        if (Muon::MuonTileID::layout(storage_tile_id[j]) == layout_one) {
          if (mid_index != j) {
            tmp = storage_tile_id[j];
            storage_tile_id[j] = storage_tile_id[mid_index];
            storage_tile_id[mid_index] = tmp;

            tmp = storage_tdc_value[j];
            storage_tdc_value[j] = storage_tdc_value[mid_index];
            storage_tdc_value[mid_index] = tmp;
          }
          mid_index++;
        }
      }

      const int thisGridX = layout_one.xGrid();
      const int thisGridY = layout_one.yGrid();
      const int otherGridX = layout_two.xGrid();
      const int otherGridY = layout_two.yGrid();
      for (uint digitsOneIndex = start_index; digitsOneIndex < mid_index; digitsOneIndex++) {
        const unsigned int keyX = Muon::MuonTileID::nX(storage_tile_id[digitsOneIndex]) * otherGridX / thisGridX;
        const unsigned int keyY = Muon::MuonTileID::nY(storage_tile_id[digitsOneIndex]);

        for (uint digitsTwoIndex = mid_index; digitsTwoIndex < end_index; digitsTwoIndex++) {
          const unsigned int candidateX = Muon::MuonTileID::nX(storage_tile_id[digitsTwoIndex]);
          const unsigned int candidateY =
            Muon::MuonTileID::nY(storage_tile_id[digitsTwoIndex]) * thisGridY / otherGridY;

          if (keyX == candidateX && keyY == candidateY) {
            Muon::MuonTileID padTile(storage_tile_id[digitsOneIndex]);
            // padTile.setY(Muon::MuonTileID::nY(storage_tile_id[digitsTwoIndex]));
            // padTile.setLayout(Muon::MuonLayout(thisGridX, otherGridY));
            // float x = 0., dx = 0., y = 0., dy = 0., z = 0., dz = 0.;
            // Muon::calcTilePos(muon_raw_to_hits->muonTables, padTile, x, dx, y, dy, z);
            // const int clusterSize = 0;
            // const int region = padTile.region();
            // setAtIndex(
            //   event_muon_hits,
            //   localCurrentHitIndex,
            //   padTile.id(),
            //   x,
            //   dx,
            //   y,
            //   dy,
            //   z,
            //   dz,
            //   uncrossed,
            //   storage_tdc_value[digitsOneIndex],
            //   storage_tdc_value[digitsOneIndex] - storage_tdc_value[digitsTwoIndex],
            //   clusterSize,
            //   region);
            const int localCurrentHitIndex = atomicAdd(current_hit_index, 1);

            uint64_t compact_hit =
              (((uint64_t)(digitsOneIndex & 0x7FFF)) << 48) | (((uint64_t)(digitsTwoIndex & 0xFFFF)) << 32) |
              ((thisGridX & 0x3FFF) << 18) | ((otherGridY & 0x3FFF) << 4) |
              (((padTile.id() & Muon::MuonBase::MaskStation) >> Muon::MuonBase::ShiftStation) & 0xF);

            muon_compact_hit[localCurrentHitIndex] = compact_hit;

            atomicAdd(station_ocurrences_offset + station, 1);

            used[digitsOneIndex] = used[digitsTwoIndex] = true;
          }
        }
      }

      for (int index = start_index; index < end_index; ++index) {
        if (!used[index]) {
          // float x = 0., dx = 0., y = 0., dy = 0., z = 0., dz = 0.;
          const auto tile = Muon::MuonTileID(storage_tile_id[index]);
          const int region = tile.region();

          int condition;
          if (tile.station() > (Muon::Constants::n_stations - 3) && region == 0) {
            condition = 0;
            // calcTilePos(muon_raw_to_hits->muonTables, tile, x, dx, y, dy, z);
          }
          else {
            if (index < mid_index) {
              condition = 1;
              // calcStripXPos(muon_raw_to_hits->muonTables, tile, x, dx, y, dy, z);
            }
            else {
              condition = 2;
              // calcStripYPos(muon_raw_to_hits->muonTables, tile, x, dx, y, dy, z);
            }
          }
          // const int clusterSize = 0;
          // setAtIndex(
          //   event_muon_hits,
          //   localCurrentHitIndex,
          //   tile.id(),
          //   x,
          //   dx,
          //   y,
          //   dy,
          //   z,
          //   dz,
          //   uncrossed,
          //   storage_tdc_value[index],
          //   storage_tdc_value[index],
          //   clusterSize,
          //   region);

          const int localCurrentHitIndex = atomicAdd(current_hit_index, 1);
          const unsigned int uncrossed = 1;

          uint64_t compact_hit = (((uint64_t)(uncrossed & 0x1)) << 63) | (((uint64_t)(index & 0x7FFF)) << 48) |
                                 (condition << 4) |
                                 (((tile.id() & Muon::MuonBase::MaskStation) >> Muon::MuonBase::ShiftStation) & 0xF);
          muon_compact_hit[localCurrentHitIndex] = compact_hit;

          atomicAdd(station_ocurrences_offset + station, 1);
        }
      }
    }
  }
}
