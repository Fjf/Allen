/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <MEPTools.h>
#include <UTCalculateNumberOfHits.cuh>

void ut_calculate_number_of_hits::ut_calculate_number_of_hits_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const HostBuffers&) const
{
  set_size<dev_ut_hit_sizes_t>(
    arguments,
    first<host_number_of_events_t>(arguments) * constants.host_unique_x_sector_layer_offsets[UT::Constants::n_layers]);
}

void ut_calculate_number_of_hits::ut_calculate_number_of_hits_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers&,
  const Allen::Context& context) const
{
  initialize<dev_ut_hit_sizes_t>(arguments, 0, context);

  auto const bank_version = first<host_raw_bank_version_t>(arguments);

  if (runtime_options.mep_layout) {
    auto fun = bank_version == 4 ? global_function(ut_calculate_number_of_hits_mep<4>) :
                                   global_function(ut_calculate_number_of_hits_mep<3>);
    fun(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(
      arguments,
      constants.dev_ut_boards.data(),
      constants.dev_ut_region_offsets.data(),
      constants.dev_unique_x_sector_layer_offsets.data(),
      constants.dev_unique_x_sector_offsets.data());
  }
  else {
    auto fun = bank_version == 4 ? global_function(ut_calculate_number_of_hits<4>) :
                                   global_function(ut_calculate_number_of_hits<3>);
    fun(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(
      arguments,
      constants.dev_ut_boards.data(),
      constants.dev_ut_region_offsets.data(),
      constants.dev_unique_x_sector_layer_offsets.data(),
      constants.dev_unique_x_sector_offsets.data());
  }
}

/**
 * @brief Given a UT RawBank, this function calculates the number of hits in a sector_group
  ("virtual" structure for optimized processing; a group of sectors where the start X is of a certain value).
 */
template<int decoding_version>
__device__ void calculate_number_of_hits(
  unsigned const*,
  unsigned const*,
  uint32_t*,
  UTBoards const&,
  UTRawBank<decoding_version> const&)
{}

template<>
__device__ void calculate_number_of_hits<3>(
  unsigned const* dev_ut_region_offsets,
  unsigned const* dev_unique_x_sector_offsets,
  uint32_t* hit_offsets,
  UTBoards const& boards,
  UTRawBank<3> const& raw_bank)
{
  const uint32_t m_nStripsPerHybrid = boards.stripsPerHybrids[raw_bank.sourceID];

  for (unsigned i = threadIdx.y; i < raw_bank.get_n_hits(); i += blockDim.y) {
    const uint32_t channelID = (raw_bank.data[i] & UT::Decoding::v4::chan_mask) >> UT::Decoding::v4::chan_offset;
    const uint32_t index = channelID / m_nStripsPerHybrid;
    const uint32_t fullChanIndex = raw_bank.sourceID * UT::Decoding::ut_number_of_sectors_per_board + index;
    const uint32_t station = boards.stations[fullChanIndex] - 1;
    const uint32_t layer = boards.layers[fullChanIndex] - 1;
    const uint32_t detRegion = boards.detRegions[fullChanIndex] - 1;
    const uint32_t sector = boards.sectors[fullChanIndex] - 1;

    // Calculate the index to get the geometry of the board
    const uint32_t idx = station * UT::Decoding::ut_number_of_sectors_per_board + layer * 3 + detRegion;
    const uint32_t idx_offset = dev_ut_region_offsets[idx] + sector;

    unsigned* hits_sector_group = hit_offsets + dev_unique_x_sector_offsets[idx_offset];
    atomicAdd(hits_sector_group, 1);
  }
}

template<>
__device__ void calculate_number_of_hits<4>(
  unsigned const* dev_ut_region_offsets,
  unsigned const* dev_unique_x_sector_offsets,
  uint32_t* hit_offsets,
  UTBoards const& boards,
  UTRawBank<4> const& raw_bank)
{
  if (raw_bank.get_n_hits() == 0) return;
  for (unsigned lane = threadIdx.y; lane < UT::Decoding::ut_number_of_sectors_per_board; lane += blockDim.y) {
    if (raw_bank.number_of_hits[lane] == 0) continue;
    // find the sector group to which these hits are added
    const uint32_t fullChanIndex = raw_bank.sourceID * UT::Decoding::ut_number_of_sectors_per_board + lane;
    const uint32_t station = boards.stations[fullChanIndex] - 1;
    const uint32_t layer = boards.layers[fullChanIndex] - 1;
    const uint32_t detRegion = boards.detRegions[fullChanIndex] - 1;
    const uint32_t sector = boards.sectors[fullChanIndex] - 1;

    // add the hits to the global counters and offsets
    const uint32_t idx = station * UT::Decoding::ut_number_of_sectors_per_board + layer * 3 + detRegion;
    const uint32_t idx_offset = dev_ut_region_offsets[idx] + sector;
    unsigned* hits_sector_group = hit_offsets + dev_unique_x_sector_offsets[idx_offset];
    atomicAdd(hits_sector_group, raw_bank.number_of_hits[lane]);
  }
}

/**
 * @brief Calculates the number of hits to be decoded for the UT detector.
 */
template<int decoding_version>
__global__ void ut_calculate_number_of_hits::ut_calculate_number_of_hits(
  ut_calculate_number_of_hits::Parameters parameters,
  const char* ut_boards,
  const unsigned* dev_ut_region_offsets,
  const unsigned* dev_unique_x_sector_layer_offsets,
  const unsigned* dev_unique_x_sector_offsets)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  const uint32_t event_offset = parameters.dev_ut_raw_input_offsets[event_number];
  const unsigned number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[UT::Constants::n_layers];
  uint32_t* hit_offsets = parameters.dev_ut_hit_sizes + event_number * number_of_unique_x_sectors;

  const UTRawEvent raw_event(parameters.dev_ut_raw_input + event_offset);
  const UTBoards boards(ut_boards);

  for (unsigned raw_bank_index = threadIdx.x; raw_bank_index < raw_event.number_of_raw_banks;
       raw_bank_index += blockDim.x)
    calculate_number_of_hits(
      dev_ut_region_offsets,
      dev_unique_x_sector_offsets,
      hit_offsets,
      boards,
      raw_event.getUTRawBank<decoding_version>(raw_bank_index));
}

/**
 * @brief Calculates the number of hits to be decoded for the UT detector.
 */
template<int decoding_version>
__global__ void ut_calculate_number_of_hits::ut_calculate_number_of_hits_mep(
  ut_calculate_number_of_hits::Parameters parameters,
  const char* ut_boards,
  const unsigned* dev_ut_region_offsets,
  const unsigned* dev_unique_x_sector_layer_offsets,
  const unsigned* dev_unique_x_sector_offsets)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  const unsigned number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[UT::Constants::n_layers];
  uint32_t* hit_offsets = parameters.dev_ut_hit_sizes + event_number * number_of_unique_x_sectors;

  const UTBoards boards(ut_boards);
  auto const number_of_ut_raw_banks = parameters.dev_ut_raw_input_offsets[0];

  for (unsigned raw_bank_index = threadIdx.x; raw_bank_index < number_of_ut_raw_banks; raw_bank_index += blockDim.x) {

    // Construct UT raw bank from MEP layout
    const auto raw_bank = MEP::raw_bank<UTRawBank<decoding_version>>(
      parameters.dev_ut_raw_input, parameters.dev_ut_raw_input_offsets, event_number, raw_bank_index);

    calculate_number_of_hits(dev_ut_region_offsets, dev_unique_x_sector_offsets, hit_offsets, boards, raw_bank);
  }
}
