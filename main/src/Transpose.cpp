/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <Transpose.h>

namespace {
  std::unordered_set<LHCb::RawBank::BankType> dont_count = {LHCb::RawBank::DAQ,
                                                            LHCb::RawBank::HltDecReports,
                                                            LHCb::RawBank::HltSelReports,
                                                            LHCb::RawBank::HltRoutingBits,
                                                            LHCb::RawBank::HltLumiSummary};
}

std::array<int, LHCb::NBankTypes> Allen::bank_ids()
{
  // Cache the mapping of LHCb::RawBank::BankType to Allen::BankType
  std::array<int, LHCb::NBankTypes> ids;
  for (auto bt : LHCb::RawBank::types()) {
    auto it = Allen::bank_mapping.find(bt);
    ids[bt] = (it != Allen::bank_mapping.end() ? to_integral(it->second) : -1);
  }
  return ids;
}

/**
 * @brief      Check if any of the soruce IDs have a non-zero value
 *             in the 5 most-significant bits
 *
 * @param      span with banks in MDF layout
 *
 * @return     true if any of the sourceIDs has a non-zero value in
 *             its 5 most-significant bits
 */
bool check_sourceIDs(gsl::span<char const> bank_data)
{

  auto const* bank = bank_data.data();

  // Loop over all the banks and check if any of the sourceIDs has
  // the most-significant bits set. In MC data they are not set.
  size_t n_banks = 0;
  size_t has_top5 = 0;
  while (bank < bank_data.data() + bank_data.size()) {

    const auto* b = reinterpret_cast<const LHCb::RawBank*>(bank);
    if (!dont_count.count(b->type())) {
      has_top5 += (SourceId_sys(static_cast<short>(b->sourceID())) != 0);
      ++n_banks;
    }

    // Increment overall bank pointer
    bank += b->totalSize();
  }

  // In real data or simulation with all the 5 most significant bits
  // correctly set, there is only a single bank with those set to 0:
  // ODIN.
  return (n_banks - has_top5) != 1;
}

/**
 * @brief      Get the (Allen) subdetector from the bank type
 *
 * @param      raw bank
 *
 * @return     Allen subdetector
 */
BankTypes sd_from_bank_type(LHCb::RawBank const* raw_bank)
{
  static auto const bank_ids = Allen::bank_ids();
  auto const bt = bank_ids[raw_bank->type()];
  return bt == -1 ? BankTypes::Unknown : static_cast<BankTypes>(bt);
}

/**
 * @brief      Get the (Allen) subdetector from the 5
 *             most-significant bits of a source ID
 *
 * @param      raw bank
 *
 * @return     Allen subdetector
 */
BankTypes sd_from_sourceID(LHCb::RawBank const* raw_bank)
{
  auto sd = SourceId_sys(raw_bank->sourceID());
  auto it = Allen::subdetectors.find(static_cast<SourceIdSys>(sd));
  auto source_type = (it == Allen::subdetectors.end()) ? BankTypes::Unknown : it->second;
  if (dont_count.count(raw_bank->type())) {
    return BankTypes::Unknown;
  }
  else {
    return source_type;
  }
}

/**
 * @brief      read events from input file into prefetch buffer
 *
 * @details    NOTE: It is assumed that the header has already been
 *             read, calling read_events will read the subsequent
 *             banks and then header of the next event.
 *
 * @param      input stream
 * @param      prefetch buffer to read into
 * @param      storage for the MDF header
 * @param      buffer for temporary storage of the compressed banks
 * @param      number of events to read
 * @param      check the MDF checksum if it is available
 *
 * @return     (eof, error, full, n_bytes)
 */
std::tuple<bool, bool, size_t> read_events(
  Allen::IO& input,
  Allen::ReadBuffer& read_buffer,
  LHCb::MDFHeader& header,
  std::vector<char>& compress_buffer,
  size_t n_events,
  bool check_checksum)
{
  auto& [n_filled, event_offsets, buffer, transpose_start] = read_buffer;

  // Keep track of where to write and the end of the prefetch buffer
  size_t n_bytes = 0;
  bool eof = false, error = false;
  gsl::span<const char> bank_span;

  // Loop until the requested number of events is prefetched, the
  // maximum number of events per prefetch buffer is hit, an error
  // occurs or eof is reached
  while (!eof && !error && n_filled < event_offsets.size() - 1 && n_filled < n_events) {
    auto* buffer_start = &buffer[0];

    // Read the banks
    auto const buffer_offset = event_offsets[n_filled];
    assert(buffer_offset < buffer.size());
    gsl::span<char> buffer_span {buffer_start + buffer_offset, static_cast<events_size>(buffer.size() - buffer_offset)};
    std::tie(eof, error, bank_span) =
      MDF::read_banks(input, header, std::move(buffer_span), compress_buffer, check_checksum);
    if (eof || error) break;

    // Fill the start offset of the next event

    if (eof || error) {
      error_cout << "Failed to read banks " << strerror(errno) << "\n";
      break;
    }
    else {
      event_offsets[n_filled + 1] = bank_span.data() + bank_span.size() - buffer_start;
      n_bytes += bank_span.size();
    }

    // read the next header
    ssize_t n_bytes = input.read(reinterpret_cast<char*>(&header), mdf_header_size);
    if (n_bytes != 0) {
      // Check if there is enough space to read this event
      int compress = header.compression() & 0xF;
      int expand = (header.compression() >> 4) + 1;
      int event_size =
        (header.recordSize() + mdf_header_size + 2 * (sizeof(LHCb::RawBank) + sizeof(int)) +
         (compress ? expand * (header.recordSize() - mdf_header_size) : 0));
      if (event_offsets[n_filled + 1] + event_size > buffer.size()) {
        buffer.resize(static_cast<size_t>(1.5 * buffer.size()));
      }
    }
    else if (n_bytes == 0) {
      info_cout << "Cannot read more data (Header). End-of-File reached.\n";
      eof = true;
    }
    else {
      error_cout << "Failed to read header " << strerror(errno) << "\n";
      error = true;
    }
    if (!error) {
      ++n_filled;
    }
  }
  return {eof, error, n_bytes};
}

/**
 * @brief      Fill the array the contains the number of banks per type
 *
 * @details    detailed description
 *
 * @param      prefetched buffer of events (a single event is needed)
 *
 * @return     (success, number of banks per bank type; 0 if the bank is not needed)
 */
std::tuple<bool, std::array<unsigned int, NBankTypes>> fill_counts(
  gsl::span<char const> bank_data,
  Allen::sd_from_raw_bank sd_from_raw_bank,
  std::unordered_set<LHCb::RawBank::BankType> const& skip_banks)
{
  std::array<unsigned int, NBankTypes> mfp_count {0};

  auto const* bank = bank_data.data();

  // Loop over all the bank data
  while (bank < bank_data.data() + bank_data.size()) {
    const auto* b = reinterpret_cast<const LHCb::RawBank*>(bank);

    if (b->magic() != LHCb::RawBank::MagicPattern) {
      error_cout << "Magic pattern failed: " << std::hex << b->magic() << std::dec << "\n";
      return {false, mfp_count};
    }

    auto const sd = sd_from_raw_bank(b);
    if ((!skip_banks.count(b->type()) && sd != BankTypes::Unknown) || sd == BankTypes::ODIN) {
      auto const sd_idx = to_integral(sd);
      ++mfp_count[sd_idx];
    }

    // Increment overall bank pointer
    bank += b->totalSize();
  }

  return {true, mfp_count};
}

std::tuple<bool, bool, bool> transpose_event(
  Allen::Slices& slices,
  int const slice_index,
  std::unordered_set<BankTypes> const& bank_types,
  Allen::sd_from_raw_bank sd_from_raw_bank,
  Allen::bank_sorter bank_sort,
  std::array<unsigned int, NBankTypes>& bank_count,
  std::unordered_set<LHCb::RawBank::BankType> const& skip_banks,
  std::array<int, NBankTypes>& banks_version,
  EventIDs& event_ids,
  std::vector<char>& event_mask,
  const gsl::span<char const> bank_data,
  std::vector<LHCb::RawBank const*>& sorted_banks,
  bool split_by_run)
{

  unsigned int* banks_offsets = nullptr;
  // Number of offsets
  size_t* n_banks_offsets = nullptr;

  // Where to write the transposed bank data
  uint32_t* banks_write = nullptr;

  // Where should offsets to individual banks be written
  uint32_t* banks_offsets_write = nullptr;

  // Memory where bank sizes are kept. The first N entries are offsets to the sizes per event
  uint16_t* fragment_sizes = nullptr;

  // Memory where bank typs are kept. The first N entries are offsets to the types per event
  uint8_t* types = nullptr;

  unsigned int bank_offset = 0;
  unsigned int bank_counter = 1;

  std::array<unsigned int, NBankTypes> size_per_type;
  size_per_type.fill(2 * sizeof(unsigned int));

  auto bank = bank_data.data(), bank_end = bank_data.data() + bank_data.size();

  while (bank < bank_end) {
    const auto* b = reinterpret_cast<const LHCb::RawBank*>(bank);

    if (b->magic() != LHCb::RawBank::MagicPattern) {
      error_cout << "Magic pattern failed: " << std::hex << b->magic() << std::dec << "\n";
      return {false, false, false};
      // Decode the odin bank
    }

    // Allen bank type
    auto const allen_type = sd_from_raw_bank(b);

    if ((!skip_banks.count(b->type()) && bank_types.count(allen_type)) || allen_type == BankTypes::ODIN) {
      sorted_banks.push_back(b);
      bank_count[to_integral(allen_type)] += 1;
      size_per_type[to_integral(allen_type)] += sizeof(unsigned int) + b->size();
    }
    bank += b->totalSize();
  }
  std::stable_sort(sorted_banks.begin(), sorted_banks.end(), bank_sort);

  // Check if any of the per-bank-type slices potentially has too
  // little space to fit this event
  for (auto allen_type : bank_types) {
    auto const ia = to_integral(allen_type);
    auto& slice = slices[ia][slice_index];
    if ((slice.offsets[slice.n_offsets - 1] + size_per_type[ia]) > slice.fragments_mem_size) {
      char* new_data = nullptr;
      size_t new_size = 1.5 * (slice.offsets[slice.n_offsets - 1] + size_per_type[ia]);
      Allen::malloc_host(reinterpret_cast<void**>(&new_data), new_size);
      if (!slice.fragments.empty() && !slice.fragments[0].empty()) {
        ::memcpy(new_data, slice.fragments[0].data(), slice.fragments_mem_size);
        Allen::free_host(slice.fragments[0].data());
        slice.fragments.clear();
      }
      slice.fragments.emplace_back(new_data, new_size);
      slice.fragments_mem_size = new_size;
    }
  }
  BankTypes prev_type = BankTypes::Unknown;

  // Loop over all bank data of this event
  for (auto const* b : sorted_banks) {

    // Allen bank type
    auto const allen_type = sd_from_raw_bank(b);

    // Check what to do with this bank
    if (allen_type == BankTypes::ODIN) {
      auto const odin_error = b->type() >= LHCb::RawBank::DaqErrorFragmentThrottled;
      event_mask[event_ids.size()] = !odin_error;

      if (!odin_error) {
        // decode ODIN bank to obtain run and event numbers
        auto odin = MDF::decode_odin(b->range<unsigned>(), b->version());
        // if splitting by run, check for a run change since the last event
        if (split_by_run) {
          if (!event_ids.empty() && odin.runNumber() != std::get<0>(event_ids.front())) {
            return {true, false, true};
          }
        }
        event_ids.emplace_back(odin.runNumber(), odin.eventNumber());
      }
      else {
        event_ids.emplace_back(0, 0);
      }
    }

    if (!bank_types.count(allen_type)) {
      warning_cout << "Bank type " << to_integral(allen_type) << " is not in requested bank types " << std::endl;
      prev_type = allen_type;
      continue;
    }
    else if (allen_type != prev_type) {
      // Switch to new type of banks
      auto& slice = slices[to_integral(allen_type)][slice_index];
      prev_type = allen_type;

      // set bank version
      banks_version[to_integral(allen_type)] = b->version();

      // Reset bank count
      bank_counter = 1;
      banks_offsets = slice.offsets.data();
      n_banks_offsets = &slice.n_offsets;

      // Calculate the size taken by storing the number of banks
      // and offsets to all banks within the event
      auto preamble_words = 2 + bank_count[to_integral(allen_type)];

      // Initialize offset to start of this set of banks from the
      // previous one and increment with the preamble size
      banks_offsets[*n_banks_offsets] = (banks_offsets[*n_banks_offsets - 1] + preamble_words * sizeof(uint32_t));

      // Five things to write for a new set of banks:
      // - number of banks/offsets
      // - offsets to individual banks
      // - bank data
      // - offset to the start of the bank sizes
      // - the bank size

      // The offsets to the sizes for this batch of fragments is
      // copied from the current value, then the pointer is moved to
      // the point where bank sizes can be stored
      auto const n_banks = bank_count[to_integral(allen_type)];
      auto const fragment_sizes_offset = slice.sizes[*n_banks_offsets - 1];
      slice.sizes[*n_banks_offsets] = fragment_sizes_offset + n_banks;
      fragment_sizes = reinterpret_cast<unsigned short*>(slice.sizes.data()) + fragment_sizes_offset;

      // The offsets to the types for this batch of fragments is
      // copied from the current value, then the pointer is moved to
      // the point where bank types can be stored
      auto const types_offset = slice.types[*n_banks_offsets - 1];
      slice.types[*n_banks_offsets] = types_offset + n_banks;
      types = reinterpret_cast<uint8_t*>(slice.types.data()) + types_offset;

      // Initialize point to write from offset of previous set
      banks_write = reinterpret_cast<uint32_t*>(slice.fragments[0].data() + banks_offsets[*n_banks_offsets - 1]);

      // New offset to increment
      ++(*n_banks_offsets);

      // Write the number of banks
      banks_write[0] = n_banks;

      // All bank offsets are uit32_t so cast to that type
      banks_offsets_write = banks_write + 1;
      banks_offsets_write[0] = 0;

      // Offset in number of uint32_t
      bank_offset = 0;

      // Start writing bank data after the preamble
      banks_write += preamble_words;
    }
    else {
      ++bank_counter;

      if (allen_type != BankTypes::VP) {
        assert(banks_version[to_integral(allen_type)] == b->version());
      }
    }

    // Write sourceID
    banks_write[bank_offset] = b->sourceID();

    // Store bank size
    fragment_sizes[bank_counter - 1] = b->size();

    // Store bank type
    types[bank_counter - 1] = to_integral(b->type());

    // Copy padded data
    auto const padded_size = b->totalSize() - b->hdrSize();
    // Write bank data
    ::memcpy(banks_write + bank_offset + 1, b->data(), padded_size);

    auto n_word = padded_size / sizeof(uint32_t);
    bank_offset += 1 + n_word;

    // Write next offset in bytes
    banks_offsets_write[bank_counter] = bank_offset * sizeof(uint32_t);

    // Update "event" offset (in bytes)
    banks_offsets[*n_banks_offsets - 1] += sizeof(uint32_t) * (1 + n_word);
  }

  return {true, false, false};
}

std::tuple<bool, bool, size_t> transpose_events(
  const Allen::ReadBuffer& read_buffer,
  Allen::Slices& slices,
  int const slice_index,
  std::unordered_set<BankTypes> const& bank_types,
  Allen::sd_from_raw_bank sd_from_raw_bank,
  Allen::bank_sorter bank_sort,
  std::array<unsigned int, NBankTypes> const& mfp_count,
  std::unordered_set<LHCb::RawBank::BankType> const& skip_banks,
  std::array<int, NBankTypes>& banks_version,
  EventIDs& event_ids,
  std::vector<char>& event_mask,
  size_t n_events,
  bool split_by_run)
{
  bool full = false, success = true, run_change = false;
  auto const& [n_filled, event_offsets, buffer, event_start] = read_buffer;
  size_t event_end = event_start + n_events;
  if (n_filled < event_end) event_end = n_filled;

  std::vector<LHCb::RawBank const*> sorted_banks;
  auto n_banks = std::accumulate(mfp_count.begin(), mfp_count.end(), 0u);
  sorted_banks.reserve(n_banks);

  std::array<unsigned int, NBankTypes> bank_count;
  bank_count.fill(0);

  // Initialize the first size offset from the number of events. The
  // offsets at the start of the array are 32 bit unsigned, while the
  // sizes themselves are 16 bit unsigned. Since the offsets are stored at
  // the start of the same array they take twice as much space.
  for (auto allen_type : bank_types) {
    auto const ia = to_integral(allen_type);
    auto& fragment_sizes_offsets = slices[ia][slice_index].sizes;
    fragment_sizes_offsets[0] = 2 * (n_events + 1);
    auto& types_offsets = slices[ia][slice_index].types;
    types_offsets[0] = 4 * (n_events + 1);
  }

  // Loop over events in the prefetch buffer
  size_t i_event = event_start;
  for (; i_event < event_end && success; ++i_event) {
    // Offsets are to the start of the event, which includes the header
    auto const* bank = buffer.data() + event_offsets[i_event];
    auto const* bank_end = buffer.data() + event_offsets[i_event + 1];
    std::tie(success, full, run_change) = transpose_event(
      slices,
      slice_index,
      bank_types,
      sd_from_raw_bank,
      bank_sort,
      bank_count,
      skip_banks,
      banks_version,
      event_ids,
      event_mask,
      {bank, bank_end},
      sorted_banks,
      split_by_run);

    sorted_banks.clear();
    bank_count.fill(0);

    // break the loop if we detect a run change or the slice is full to avoid incrementing i_event
    if (run_change || full) break;
  }

  return {success, full, i_event - event_start};
}
