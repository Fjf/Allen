/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <MEPTools.h>
#include <UTPreDecode.cuh>
#include <UTUniqueID.cuh>

INSTANTIATE_ALGORITHM(ut_pre_decode::ut_pre_decode_t)

void ut_pre_decode::ut_pre_decode_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants& constants) const
{
  set_size<dev_ut_pre_decoded_hits_t>(
    arguments, first<host_accumulated_number_of_ut_hits_t>(arguments) * UT::PreDecodedHits::element_size);
  set_size<dev_ut_hit_count_t>(
    arguments,
    first<host_number_of_events_t>(arguments) * constants.host_unique_x_sector_layer_offsets[UT::Constants::n_layers]);
}

void ut_pre_decode::ut_pre_decode_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_ut_hit_count_t>(arguments, 0, context);

  auto const bank_version = first<host_raw_bank_version_t>(arguments);
  if (bank_version < 0) return; // no UT banks present in data

  auto fun = bank_version == 4 ? (runtime_options.mep_layout ? global_function(ut_pre_decode<4, true>) :
                                                               global_function(ut_pre_decode<4, false>)) :
                                 (runtime_options.mep_layout ? global_function(ut_pre_decode<3, true>) :
                                                               global_function(ut_pre_decode<3, false>));

  fun(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(
    arguments,
    std::get<0>(runtime_options.event_interval),
    constants.dev_ut_boards,
    constants.dev_ut_geometry.data(),
    constants.dev_unique_x_sector_layer_offsets.data(),
    constants.dev_unique_x_sector_offsets.data());
}

/**
 * @brief Makes an unsigned key out of a float, where
 *        float order is preserved. Unsigned order can be composed
 *        to one another as opposed to float order (sign-magnitude).
 */
__device__ uint32_t generate_sort_key(const float a)
{
  int32_t i = Allen::device::bit_cast<int32_t>(a);
  return i < 0 ? -i & 0x7FFFFFFF : (1u << 31) | (i & 0x7FFFFFFF);
}

/**
 * @brief Makes a composed key value made out of:
 *        (u32 representation of yBegin) (u32 representation of xAtYEq0)
 */
__device__ uint64_t generate_sort_key(const float xAtYEq0, const float yBegin)
{
  auto yBegin_u = generate_sort_key(yBegin);
  auto xAtYEq0_u = generate_sort_key(xAtYEq0);
  return static_cast<uint64_t>(yBegin_u) << 32 | xAtYEq0_u;
}

/**
 * @details Given a RawBank, this function partly decodes the hits to sort them by yBegin.
 *          In case hits have the same yBegin, they are sorted by x (xAtYEq0).
 *          Hit indices in the RawBank are persisted along with the variable for sorting
 *          to enable a loop over hits later on.
 */
template<int decoding_version>
__device__ void pre_decode_raw_bank(
  unsigned const*,
  uint32_t const*,
  UTGeometry const&,
  UTBoards const&,
  UTRawBank<decoding_version> const&,
  unsigned const,
  UT::PreDecodedHits&,
  uint32_t*)
{}

template<>
__device__ void pre_decode_raw_bank<3>(
  unsigned const* dev_unique_x_sector_offsets,
  uint32_t const* hit_offsets,
  UTGeometry const& geometry,
  UTBoards const& boards,
  UTRawBank<3> const& raw_bank,
  unsigned const raw_bank_index,
  UT::PreDecodedHits& ut_pre_decoded_hits,
  uint32_t* hit_count)
{
  const uint32_t m_nStripsPerHybrid = boards.stripsPerHybrids[raw_bank.sourceID];
  for (unsigned i = threadIdx.y; i < raw_bank.number_of_hits[0]; i += blockDim.y) {
    // Extract values from raw_data
    const uint16_t value = raw_bank.data[i];
    const uint32_t fracStrip = (value & UT::Decoding::v4::frac_mask) >> UT::Decoding::v4::frac_offset;
    const uint32_t channelID = (value & UT::Decoding::v4::chan_mask) >> UT::Decoding::v4::chan_offset;

    // Calculate the relative index of the corresponding board
    const uint32_t index = channelID / m_nStripsPerHybrid;
    const uint32_t strip = channelID - (index * m_nStripsPerHybrid) + 1;

    const uint32_t fullChanIndex = raw_bank.sourceID * UT::Decoding::ut_number_of_sectors_per_board + index;
    if (fullChanIndex >= boards.number_of_channels) continue;
    const uint32_t side = boards.sides[fullChanIndex];
    const uint32_t layer = boards.layers[fullChanIndex];
    const uint32_t stave = boards.staves[fullChanIndex];
    const uint32_t face = boards.faces[fullChanIndex];
    const uint32_t module = boards.modules[fullChanIndex];
    const uint32_t sector = boards.sectors[fullChanIndex];

    // Calculate the index to get the geometry of the board
    int sec = sector_unique_id(side, layer, stave, face, module, sector);

    const uint32_t firstStrip = geometry.firstStrip[sec];
    const float dp0diX = geometry.dp0diX[sec];
    const float dp0diY = geometry.dp0diY[sec];
    const float p0X = geometry.p0X[sec];
    const float p0Y = geometry.p0Y[sec];
    const float numstrips = 0.25f * fracStrip + strip - firstStrip;

    const auto yBegin = p0Y + numstrips * dp0diY;
    const auto xAtYEq0 = p0X + numstrips * dp0diX;
    const auto key = generate_sort_key(xAtYEq0, yBegin);

    const unsigned base_sector_group_offset = dev_unique_x_sector_offsets[sec];
    unsigned* hits_count_sector_group = hit_count + base_sector_group_offset;

    const unsigned current_hit_count = atomicAdd(hits_count_sector_group, 1);
    assert(current_hit_count < hit_offsets[base_sector_group_offset + 1] - hit_offsets[base_sector_group_offset]);

    const unsigned hit_index = hit_offsets[base_sector_group_offset] + current_hit_count;
    ut_pre_decoded_hits.sort_key(hit_index) = key;

    // Raw bank hit index:
    // [raw bank 8 bits] [hit id inside raw bank 24 bits]
    assert(i < (0x1 << 24));
    const uint32_t raw_bank_hit_index = raw_bank_index << 24 | i;
    ut_pre_decoded_hits.index(hit_index) = raw_bank_hit_index;
  }
}

template<>
__device__ void pre_decode_raw_bank<4>(
  unsigned const* dev_unique_x_sector_offsets,
  uint32_t const* hit_offsets,
  UTGeometry const& geometry,
  UTBoards const& boards,
  UTRawBank<4> const& raw_bank,
  unsigned const raw_bank_index,
  UT::PreDecodedHits& ut_pre_decoded_hits,
  uint32_t* hit_count)
{
  const uint32_t m_nStripsPerHybrid = boards.stripsPerHybrids[raw_bank.sourceID];
  for (unsigned lane = threadIdx.y; lane < UT::Decoding::ut_number_of_sectors_per_board; lane += blockDim.y) {
    // skip if there's nothing
    if (raw_bank.number_of_hits[lane] == 0) continue;
    // we can do some things that only depend on lane and sourceID before decoding individual hits
    const uint32_t fullChanIndex = raw_bank.sourceID * UT::Decoding::ut_number_of_sectors_per_board + lane;
    // const uint32_t s = boards.stations[fullChanIndex];
    // if (s == 0) continue;
    const uint32_t side = boards.sides[fullChanIndex];
    const uint32_t layer = boards.layers[fullChanIndex];
    const uint32_t stave = boards.staves[fullChanIndex];
    const uint32_t face = boards.faces[fullChanIndex];
    const uint32_t module = boards.modules[fullChanIndex];
    const uint32_t sector = boards.sectors[fullChanIndex];

    int sec = sector_unique_id(side, layer, stave, face, module, sector);
    const uint32_t firstStrip = geometry.firstStrip[sec];
    const float dp0diX = geometry.dp0diX[sec];
    const float dp0diY = geometry.dp0diY[sec];
    const float p0X = geometry.p0X[sec];
    const float p0Y = geometry.p0Y[sec];
    const float p0Z = geometry.p0Z[sec];

    // Now we can start decoding hits from the v5 RawBank. The RawBank header (64 bits) tells you how many hits there
    // are. The RawBank data itself contains lane-wise zero-padded "words". When casting to 32 bits, this looks like
    // 1280  0  0  0  0  669913862 for example. So there is something in lane 5 (1280) and 0 (669913862), all other
    // lanes don't have hits. These words have been encoded as 32 bit integers with the corresponding bitshifts that
    // allow reading them as 16 bit integers, which is what we will do. This means we can loop individual hits but have
    // to do slighty more complicated indexing gymnastics.
    for (unsigned ihit = 0; ihit < raw_bank.number_of_hits[lane]; ihit++) { // loop hits
      const auto hit_index_inside_raw_bank = 16 * (ihit / 2) + 2 * (5 - lane) + ihit % 2;
      const uint16_t word = raw_bank.data[hit_index_inside_raw_bank];
      // this is the magic step that tells us which strip was hit
      const auto stripID = (word & UT::Decoding::v5::strip_mask) >> UT::Decoding::v5::strip_offset;
      // we need to know whether or not a "stripflip" canges the numbering
      const auto numstrips = p0Z < 0 ? m_nStripsPerHybrid - (stripID - firstStrip) - 1 : stripID - firstStrip;

      // The magic of combining yBegin and xAtYEq0 has been explained in the v4 code above.
      const auto key = generate_sort_key(p0X + numstrips * dp0diX, p0Y + numstrips * dp0diY);

      // Finally we need to fill the global containers correctly
      const unsigned base_sector_group_offset =
        dev_unique_x_sector_offsets[sec]; // idx; //dev_unique_x_sector_offsets[idx_offset];
      unsigned* hits_count_sector_group = hit_count + base_sector_group_offset;

      const unsigned current_hit_count = atomicAdd(hits_count_sector_group, 1);
      assert(current_hit_count < hit_offsets[base_sector_group_offset + 1] - hit_offsets[base_sector_group_offset]);

      const unsigned hit_index = hit_offsets[base_sector_group_offset] + current_hit_count;
      ut_pre_decoded_hits.sort_key(hit_index) = key;

      // Raw bank hit index (encodes in which RawBank the hit is, and where it is inside that RawBank):
      // [raw bank 8 bits] [hit id inside raw bank 24 bits]
      assert(hit_index_inside_raw_bank < (0x1 << 24));
      const uint32_t raw_bank_hit_index = raw_bank_index << 24 | hit_index_inside_raw_bank;
      ut_pre_decoded_hits.index(hit_index) = raw_bank_hit_index;
    } // end loop hits
  }   // end loop lanes
}

/**
 * Iterate over raw banks / hits and store only the Y coordinate,
 * and an uint32_t encoding the following:
 * raw_bank number and hit id inside the raw bank.
 * Let's refer to this array as raw_bank_hits.
 *
 * Kernel suitable for decoding from Allen layout
 */
template<int decoding_version, bool mep>
__global__ void ut_pre_decode::ut_pre_decode(
  ut_pre_decode::Parameters parameters,
  const unsigned event_start,
  const char* ut_boards,
  const char* ut_geometry,
  const unsigned* dev_unique_x_sector_layer_offsets,
  const unsigned* dev_unique_x_sector_offsets)
{
  const unsigned number_of_events = parameters.dev_number_of_events[0];
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  const unsigned number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[UT::Constants::n_layers];
  const uint32_t* hit_offsets = parameters.dev_ut_hit_offsets + event_number * number_of_unique_x_sectors;
  uint32_t* hit_count = parameters.dev_ut_hit_count + event_number * number_of_unique_x_sectors;

  UT::PreDecodedHits ut_pre_decoded_hits {parameters.dev_ut_pre_decoded_hits,
                                          parameters.dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]};

  const UTGeometry geometry(ut_geometry);
  const UTBoards boards(ut_boards);

  const UTRawEvent<mep> raw_event {parameters.dev_ut_raw_input,
                                   parameters.dev_ut_raw_input_offsets,
                                   parameters.dev_ut_raw_input_sizes,
                                   event_number + event_start};
  for (unsigned raw_bank_index = threadIdx.x; raw_bank_index < raw_event.number_of_raw_banks();
       raw_bank_index += blockDim.x)
    pre_decode_raw_bank(
      dev_unique_x_sector_offsets,
      hit_offsets,
      geometry,
      boards,
      raw_event.template raw_bank<decoding_version>(raw_bank_index),
      raw_bank_index,
      ut_pre_decoded_hits,
      hit_count);
}
