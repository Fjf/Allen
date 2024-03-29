/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <MEPTools.h>
#include <UTDecodeRawBanksInOrder.cuh>
#include "LHCbID.cuh"
#include <UTUniqueID.cuh>

INSTANTIATE_ALGORITHM(ut_decode_raw_banks_in_order::ut_decode_raw_banks_in_order_t)

void ut_decode_raw_banks_in_order::ut_decode_raw_banks_in_order_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_ut_hits_t>(arguments, first<host_accumulated_number_of_ut_hits_t>(arguments) * UT::Hits::element_size);
}

void ut_decode_raw_banks_in_order::ut_decode_raw_banks_in_order_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const Allen::Context& context) const
{
  auto const bank_version = first<host_raw_bank_version_t>(arguments);
  if (bank_version < 0) return; // no UT banks present in data

  auto fun = bank_version == 4 ?
               (runtime_options.mep_layout ? global_function(ut_decode_raw_banks_in_order<4, true>) :
                                             global_function(ut_decode_raw_banks_in_order<4, false>)) :
               (runtime_options.mep_layout ? global_function(ut_decode_raw_banks_in_order<3, true>) :
                                             global_function(ut_decode_raw_banks_in_order<3, false>));

  fun(dim3(size<dev_event_list_t>(arguments), UT::Constants::n_layers), property<block_dim_t>(), context)(
    arguments,
    std::get<0>(runtime_options.event_interval),
    constants.dev_ut_boards,
    constants.dev_ut_geometry.data(),
    constants.dev_unique_x_sector_layer_offsets.data());

  if (property<verbosity_t>() >= logger::debug) {
    auto host_ut_hits = make_host_buffer<dev_ut_hits_t>(arguments, context);
    auto host_ut_hit_offsets = make_host_buffer<dev_ut_hit_offsets_t>(arguments, context);
    auto hits_view = UT::Hits {host_ut_hits.data(), first<host_accumulated_number_of_ut_hits_t>(arguments)};

    for (unsigned i = 0; i < first<host_accumulated_number_of_ut_hits_t>(arguments); ++i) {
      debug_cout << hits_view.id(i) << ", ";
    }
    debug_cout << "\n";
  }
}

/**
 * @brief Given a RawBank and indices from sorted hits, this function fully decodes UTHits for use in the tracking.
 */
template<int decoding_version>
__device__ void decode_raw_bank(
  UTGeometry const&,
  UTBoards const&,
  UTRawBank<decoding_version> const&,
  unsigned const,
  unsigned const,
  UT::Hits&)
{}

template<>
__device__ void decode_raw_bank<3>(
  UTGeometry const& geometry,
  UTBoards const& boards,
  UTRawBank<3> const& raw_bank,
  unsigned const hit_index,
  unsigned const raw_bank_hit_index,
  UT::Hits& ut_hits)
{
  // decode the hit's index within the RawBank, get and decode it (see UTPredecode for more details)
  const unsigned hit_index_inside_raw_bank = raw_bank_hit_index & 0xFFFFFF;
  const auto nStripsPerHybrid = boards.stripsPerHybrids[raw_bank.sourceID];
  const uint16_t word = raw_bank.data[hit_index_inside_raw_bank];

  // Extract values from raw_data
  const uint32_t fracStrip = (word & UT::Decoding::v4::frac_mask) >> UT::Decoding::v4::frac_offset;
  const uint32_t channelID = (word & UT::Decoding::v4::chan_mask) >> UT::Decoding::v4::chan_offset;
  // Calculate the relative index of the corresponding board
  const uint32_t index = channelID / nStripsPerHybrid;
  const uint32_t stripID = channelID - (index * nStripsPerHybrid) + 1;
  const uint32_t fullChanIndex = raw_bank.sourceID * UT::Decoding::ut_number_of_sectors_per_board + index;

  const uint32_t side = boards.sides[fullChanIndex];
  const uint32_t layer = boards.layers[fullChanIndex];
  const uint32_t stave = boards.staves[fullChanIndex];
  const uint32_t face = boards.faces[fullChanIndex];
  const uint32_t module = boards.modules[fullChanIndex];
  const uint32_t sector = boards.sectors[fullChanIndex];
  const uint32_t chanID = boards.chanIDs[fullChanIndex];
  int sec = sector_unique_id(side, layer, stave, face, module, sector);

  const float pitch = geometry.pitch[sec];
  const uint32_t firstStrip = geometry.firstStrip[sec];
  const float dy = geometry.dy[sec];
  const float dp0diX = geometry.dp0diX[sec];
  const float dp0diY = geometry.dp0diY[sec];
  const float dp0diZ = geometry.dp0diZ[sec];
  const float p0X = geometry.p0X[sec];
  const float p0Y = geometry.p0Y[sec];
  const float p0Z = geometry.p0Z[sec];
  const float numstrips = 0.25f * fracStrip + stripID - firstStrip;

  const float yBegin = p0Y + numstrips * dp0diY;
  const float yEnd = dy + yBegin;
  const float zAtYEq0 = fabsf(p0Z) + numstrips * dp0diZ;
  const float xAtYEq0 = p0X + numstrips * dp0diX;
  const float weight = 12.f / (pitch * pitch);
  const uint32_t LHCbID = lhcb_id::set_detector_type_id(lhcb_id::LHCbIDType::UT, (chanID + stripID - 1));

  ut_hits.yBegin(hit_index) = yBegin;
  ut_hits.yEnd(hit_index) = yEnd;
  ut_hits.zAtYEq0(hit_index) = zAtYEq0;
  ut_hits.xAtYEq0(hit_index) = xAtYEq0;
  ut_hits.weight(hit_index) = weight;
  ut_hits.id(hit_index) = LHCbID;
}

template<>
__device__ void decode_raw_bank<4>(
  UTGeometry const& geometry,
  UTBoards const& boards,
  UTRawBank<4> const& raw_bank,
  unsigned const hit_index,
  unsigned const raw_bank_hit_index,
  UT::Hits& ut_hits)
{
  // decode the hit's index within the RawBank, get and decode it (see UTPredecode for more details)
  const unsigned hit_index_inside_raw_bank = raw_bank_hit_index & 0xFFFFFF;
  const auto nStripsPerHybrid = boards.stripsPerHybrids[raw_bank.sourceID];

  const uint16_t word = raw_bank.data[hit_index_inside_raw_bank];
  const uint32_t stripID = (word & UT::Decoding::v5::strip_mask) >> UT::Decoding::v5::strip_offset;
  // decode lane number from hit_index_inside_raw_bank, which is 16*(ihit/2) + 2*(5-lane) + ihit%2
  const uint32_t lane = 5 - (hit_index_inside_raw_bank % 16) / 2;
  assert(lane <= 6);
  const uint32_t fullChanIndex = raw_bank.sourceID * UT::Decoding::ut_number_of_sectors_per_board + lane;

  const uint32_t side = boards.sides[fullChanIndex];
  const uint32_t layer = boards.layers[fullChanIndex];
  const uint32_t stave = boards.staves[fullChanIndex];
  const uint32_t face = boards.faces[fullChanIndex];
  const uint32_t module = boards.modules[fullChanIndex];
  const uint32_t sector = boards.sectors[fullChanIndex];
  const uint32_t chanID = boards.chanIDs[fullChanIndex];
  int sec = sector_unique_id(side, layer, stave, face, module, sector);

  const float pitch = geometry.pitch[sec];
  const uint32_t firstStrip = geometry.firstStrip[sec];
  const float dy = geometry.dy[sec];
  const float dp0diX = geometry.dp0diX[sec];
  const float dp0diY = geometry.dp0diY[sec];
  const float dp0diZ = geometry.dp0diZ[sec];
  const float p0X = geometry.p0X[sec];
  const float p0Y = geometry.p0Y[sec];
  const float p0Z = geometry.p0Z[sec];
  const float numstrips = p0Z < 0 ? nStripsPerHybrid - stripID - firstStrip : stripID;

  const float yBegin = p0Y + numstrips * dp0diY;
  const float yEnd = dy + yBegin;
  const float zAtYEq0 = fabsf(p0Z) + numstrips * dp0diZ;
  const float xAtYEq0 = p0X + numstrips * dp0diX;
  const float weight = 12.f / (pitch * pitch);
  const uint32_t LHCbID = lhcb_id::set_detector_type_id(lhcb_id::LHCbIDType::UT, (chanID + stripID));

  ut_hits.yBegin(hit_index) = yBegin;
  ut_hits.yEnd(hit_index) = yEnd;
  ut_hits.zAtYEq0(hit_index) = zAtYEq0;
  ut_hits.xAtYEq0(hit_index) = xAtYEq0;
  ut_hits.weight(hit_index) = weight;
  ut_hits.id(hit_index) = LHCbID;
}

template<int decoding_version, bool mep>
__global__ void ut_decode_raw_banks_in_order::ut_decode_raw_banks_in_order(
  ut_decode_raw_banks_in_order::Parameters parameters,
  const unsigned event_start,
  const char* ut_boards,
  const char* ut_geometry,
  const unsigned* dev_unique_x_sector_layer_offsets)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const unsigned number_of_events = parameters.dev_number_of_events[0];
  const unsigned layer_number = blockIdx.y;
  const unsigned number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[UT::Constants::n_layers];

  const UT::HitOffsets ut_hit_offsets {
    parameters.dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};

  UT::Hits ut_hits {parameters.dev_ut_hits,
                    parameters.dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]};

  UT::ConstPreDecodedHits ut_pre_decoded_hits {
    parameters.dev_ut_pre_decoded_hits, parameters.dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]};

  const UTBoards boards(ut_boards);
  const UTGeometry geometry(ut_geometry);

  const UTRawEvent<mep> raw_event {parameters.dev_ut_raw_input,
                                   parameters.dev_ut_raw_input_offsets,
                                   parameters.dev_ut_raw_input_sizes,
                                   event_number + event_start};

  const unsigned layer_offset = ut_hit_offsets.layer_offset(layer_number);
  const unsigned layer_number_of_hits = ut_hit_offsets.layer_number_of_hits(layer_number);

  for (unsigned i = threadIdx.x; i < layer_number_of_hits; i += blockDim.x) {
    const unsigned hit_index = layer_offset + i;
    const unsigned raw_bank_hit_index = ut_pre_decoded_hits.index(parameters.dev_ut_hit_permutations[hit_index]);
    const unsigned raw_bank_index = raw_bank_hit_index >> 24;
    const auto raw_bank = raw_event.template raw_bank<decoding_version>(raw_bank_index);
    decode_raw_bank(geometry, boards, raw_bank, hit_index, raw_bank_hit_index, ut_hits);
  }
}
