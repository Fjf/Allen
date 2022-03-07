/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <unordered_set>
#include <map>
#include <cassert>
#include <cmath>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <Logger.h>
#include <Event/RawBank.h>
#include <read_mdf.hpp>
#include <eb_header.hpp>
#include <read_mdf.hpp>
#include <read_mep.hpp>
#include <SliceUtils.h>
#include <Transpose.h>
#include <TransposeMEP.h>
#include <BackendCommon.h>
#include <HostGlobalEventCut.h>
#include <Argument.cuh>

using namespace std;

int main(int argc, char* argv[])
{
  if (argc != 3) {
    cout << "usage: read_mep file.mep n_mep" << endl;
    return -1;
  }

  string filename = {argv[1]};
  size_t n_meps = atol(argv[2]);

  // Some storage for reading the events into
  bool eof = false, success = false;
  EB::Header mep_header;
  gsl::span<char const> mep_span;

  auto input = MDF::open(filename.c_str(), O_RDONLY);
  if (input.good) {
    info_cout << "Opened " << filename << "\n";
  }
  else {
    cerr << "Failed to open file " << filename << " " << strerror(errno) << "\n";
    return -1;
  }

  vector<char> data;

  auto bank_ids = ::bank_ids();
  MEP::Blocks blocks;

  vector<unsigned> scifi_block_ids, ut_block_ids;

  bool count_success = false;
  std::array<unsigned int, LHCb::NBankTypes> banks_count;
  std::array<int, NBankTypes> banks_version;
  std::unordered_set<BankTypes> bank_types {
    BankTypes::VP, BankTypes::VPRetinaCluster, BankTypes::UT, BankTypes::FT, BankTypes::MUON};

  ::Slices slices;
  EventIDs events;

  size_t interval = 1000;

  size_t i_mep = 0;
  while (!eof && i_mep++ < n_meps) {

    // Read MEP
    std::tie(eof, success, mep_header, mep_span) = MEP::read_mep(input, data);

    //
    if (!count_success) {
      // Count banks per type
      std::tie(count_success, banks_count, banks_version) = MEP::fill_counts(mep_header, mep_span, bank_ids);

      // Allocate slices
      auto size_fun = [&banks_count, &bank_ids, interval](BankTypes bank_type) -> std::tuple<size_t, size_t> {
        auto it = std::find(bank_ids.begin(), bank_ids.end(), to_integral(bank_type));
        auto lhcb_type = std::distance(bank_ids.begin(), it);
        auto n_blocks = banks_count[lhcb_type];
        // 0 to not allocate data memory; -1 to correct for +1 in allocate_slices: re-evaluate
        return {0, 2 + n_blocks + (1 + interval) * (1 + n_blocks) - 2};
      };
      slices = allocate_slices(n_meps, bank_types, size_fun);
      blocks.resize(mep_header.n_blocks);
    }

    // Fill blocks
    MEP::find_blocks(mep_header, mep_span, blocks);

    size_t i_block = 0;
    for (auto const& [block_header, block_span] : blocks) {
      auto const lhcb_type = int {block_header.types[0]};
      auto const allen_type = bank_ids[lhcb_type];

      // Copy blocks and calculate block offsets
      for (auto& [ids, at] : {std::tuple {std::ref(scifi_block_ids), BankTypes::FT},
                              std::tuple {std::ref(ut_block_ids), BankTypes::UT}}) {
        if (allen_type == to_integral(at)) {
          auto& [spans, offset, offsets, offsets_size] = slices[allen_type][0];
          ids.get().emplace_back(i_block);
          spans.emplace_back(const_cast<char*>(block_span.data()), block_span.size());
          // auto* data_start = spans[0].begin();
          // std::memcpy(data_start + offset, block_span.data(), block_span.size());
          // offset += block_header.block_size;
        }
      }
      ++i_block;
    }

    MEP::mep_offsets(
      slices, 0, bank_ids, {BankTypes::UT, BankTypes::FT}, banks_count, events, mep_header, blocks, {0, interval});
    auto scifi_allen_type = to_integral(BankTypes::FT);

    auto const& [scifi_data, scifi_data_size, scifi_offsets, scifi_offsets_size] = slices[scifi_allen_type][0];

    auto n_scifi_fragments = scifi_block_ids.size();

    for (size_t i_block = 0; i_block < scifi_block_ids.size(); ++i_block) {
      for (size_t event = 0; event < interval; ++event) {
        auto const& sizes = std::get<0>(blocks[scifi_block_ids[i_block]]).sizes;
        [[maybe_unused]] auto fragment_size = sizes[event];

        unsigned const offset_index = 2 + n_scifi_fragments * (1 + event);
        [[maybe_unused]] unsigned bank_size =
          scifi_offsets[offset_index + i_block + n_scifi_fragments] - scifi_offsets[offset_index + i_block];
        assert(bank_size == fragment_size);
      }
    }

    auto slice_to_banks = [&slices](int slice_index, BankTypes bank_type) {
      auto bt = to_integral(bank_type);
      auto const& [data, data_size, offsets, offsets_size] = slices[bt][slice_index];
      BanksAndOffsets bno;
      auto& spans = std::get<0>(bno);
      spans.reserve(data.size());
      for (auto s : data) {
        spans.emplace_back(s);
      }
      std::get<1>(bno) = data_size;
      std::get<2>(bno) = offsets;
      return bno;
    };

    auto ut_banks = slice_to_banks(0, BankTypes::UT);
    auto scifi_banks = slice_to_banks(0, BankTypes::FT);

    vector<unsigned> host_total_number_of_events(interval, 0);
    vector<unsigned> host_event_list(interval, 0);
    vector<mask_t> event_list(interval, mask_t {0});
    unsigned dev_number_of_events = 0;
    unsigned number_of_selected_events = 0;

    host_global_event_cut::Parameters pars {std::get<0>(ut_banks).data(),
                                            &std::get<2>(ut_banks),
                                            &std::get<3>(ut_banks),
                                            std::get<0>(scifi_banks).data(),
                                            &std::get<2>(scifi_banks),
                                            host_event_list.data(),
                                            host_total_number_of_events.data(),
                                            &number_of_selected_events,
                                            &dev_number_of_events,
                                            event_list.data(),
                                            0,
                                            9750};

    host_global_event_cut::host_global_event_cut<true>(pars);

    cout << "selected " << number_of_selected_events << " events" << endl;
  }

  return 0;
}
