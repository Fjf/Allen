/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <MEPTools.h>
#include <MuonPopulateTileAndTDC.cuh>

INSTANTIATE_ALGORITHM(muon_populate_tile_and_tdc::muon_populate_tile_and_tdc_t)

__device__ void decode_muon_bank(
  Muon::MuonRawToHits const* muon_raw_to_hits,
  int const batch_index,
  Muon::MuonRawBank<2> const& raw_bank,
  const unsigned* storage_station_region_quarter_offsets,
  unsigned* atomics_muon,
  unsigned* dev_storage_tile_id,
  unsigned* dev_storage_tdc_value)
{
  const auto tell_number = raw_bank.sourceID;
  const uint16_t* p = raw_bank.data;

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
      const auto tile = Muon::MuonTileID(tileId);

      const auto x1 =
        getLayoutX(muon_raw_to_hits->muonTables, Muon::MuonTables::stripXTableNumber, tile.station(), tile.region());
      const auto y1 =
        getLayoutY(muon_raw_to_hits->muonTables, Muon::MuonTables::stripXTableNumber, tile.station(), tile.region());
      const auto x2 =
        getLayoutX(muon_raw_to_hits->muonTables, Muon::MuonTables::stripYTableNumber, tile.station(), tile.region());
      const auto y2 =
        getLayoutY(muon_raw_to_hits->muonTables, Muon::MuonTables::stripYTableNumber, tile.station(), tile.region());
      const auto layout1 = (x1 > x2 ? Muon::MuonLayout {x1, y1} : Muon::MuonLayout {x2, y2});

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

template<bool mep_layout>
__global__ void muon_populate_tile_and_tdc_kernel(muon_populate_tile_and_tdc::Parameters parameters)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const auto raw_event = Muon::RawEvent<mep_layout> {
    parameters.dev_muon_raw, parameters.dev_muon_raw_offsets, parameters.dev_muon_raw_sizes, event_number};
  const auto storage_station_region_quarter_offsets =
    parameters.dev_storage_station_region_quarter_offsets +
    event_number * 2 * Muon::Constants::n_stations * Muon::Constants::n_regions * Muon::Constants::n_quarters;
  unsigned* atomics_muon = parameters.dev_atomics_muon + event_number * 2 * Muon::Constants::n_stations *
                                                           Muon::Constants::n_regions * Muon::Constants::n_quarters;

  // number_of_raw_banks = 10
  // batches_per_bank = 4
  constexpr uint32_t batches_per_bank_mask = 0x3;
  constexpr uint32_t batches_per_bank_shift = 2;
  for (unsigned i = threadIdx.x; i < raw_event.number_of_raw_banks() * Muon::batches_per_bank; i += blockDim.x) {
    const auto bank_index = i >> batches_per_bank_shift;
    const auto batch_index = i & batches_per_bank_mask;

    const auto raw_bank = raw_event.raw_bank(bank_index);

    decode_muon_bank(
      parameters.dev_muon_raw_to_hits,
      batch_index,
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
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_storage_tile_id_t>(arguments, first<host_muon_total_number_of_tiles_t>(arguments));
  set_size<dev_storage_tdc_value_t>(arguments, first<host_muon_total_number_of_tiles_t>(arguments));
  set_size<dev_atomics_muon_t>(
    arguments,
    first<host_number_of_events_t>(arguments) * 2 * Muon::Constants::n_stations * Muon::Constants::n_regions *
      Muon::Constants::n_quarters);
}

void muon_populate_tile_and_tdc::muon_populate_tile_and_tdc_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  HostBuffers&,
  const Allen::Context& context) const
{
  initialize<dev_atomics_muon_t>(arguments, 0, context);
  initialize<dev_storage_tile_id_t>(arguments, 0, context);
  initialize<dev_storage_tdc_value_t>(arguments, 0, context);

  global_function(
    runtime_options.mep_layout ? muon_populate_tile_and_tdc_kernel<true> : muon_populate_tile_and_tdc_kernel<false>)(
    size<dev_event_list_t>(arguments),
    // FIXME:
    10 * Muon::batches_per_bank,
    context)(arguments);
}
