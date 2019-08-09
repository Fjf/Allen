#pragma once

#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <BankTypes.h>

#include <gsl-lite.hpp>

#include "odin.hpp"
#include "raw_bank.hpp"
#include "mdf_header.hpp"

namespace Allen {
  const std::unordered_map<LHCb::RawBank::BankType, BankTypes> bank_types = {{LHCb::RawBank::VP, BankTypes::VP},
                                                                             {LHCb::RawBank::UT, BankTypes::UT},
                                                                             {LHCb::RawBank::FTCluster, BankTypes::FT},
                                                                             {LHCb::RawBank::Muon, BankTypes::MUON}};

  using buffer_map = std::unordered_map<BankTypes, std::pair<std::vector<char>, std::vector<unsigned int>>>;
} // namespace Allen

namespace MDF {
  void dump_hex(const char* start, int size);

  std::tuple<bool, bool, gsl::span<char>> read_event(
    int input,
    LHCb::MDFHeader& h,
    gsl::span<char> buffer,
    std::vector<char>& decompression_buffer,
    bool checkChecksum = true,
    bool dbg = false);

  std::tuple<bool, bool, gsl::span<char>> read_banks(
    int input,
    const LHCb::MDFHeader& h,
    gsl::span<char> buffer,
    std::vector<char>& decompression_buffer,
    bool checkChecksum = true,
    bool dbg = false);

  std::tuple<size_t, Allen::buffer_map, std::vector<LHCb::ODIN>> read_events(
    size_t n,
    const std::vector<std::string>& files,
    const std::unordered_set<BankTypes>& types,
    bool checkChecksum = true,
    size_t offset = 0);

  void transpose_event(
    std::vector<char> const& input_buffer,
    size_t input_offset,
    char* output_buffer,
    size_t output_offset,
    size_t output_size);

  LHCb::ODIN decode_odin(const LHCb::RawBank* bank);

} // namespace MDF
