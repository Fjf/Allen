/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <vector>
#include <array>

#include <Common.h>
#include <Logger.h>
#include <SystemOfUnits.h>
#include <BankTypes.h>
#include <BackendCommon.h>

#include "SliceUtils.h"

/**
 * @brief      Reset a slice
 *
 * @param      slices
 * @param      slice_index
 * @param      event_ids
 */
void reset_slice(
  Allen::Slices& slices,
  int const slice_index,
  std::unordered_set<BankTypes> const& bank_types,
  EventIDs& event_ids,
  bool mep)
{
  // "Reset" the slice
  for (auto bank_type : bank_types) {
    auto ib = to_integral(bank_type);
    auto& slice = slices[ib][slice_index];
    std::fill(slice.offsets.begin(), slice.offsets.end(), 0);
    std::fill(slice.sizes.begin(), slice.sizes.end(), 0);
    std::fill(slice.types.begin(), slice.types.end(), 0);
    slice.n_offsets = 1;
    if (mep) {
      slice.fragments.clear();
      slice.fragments_mem_size = 0;
    }
  }
  event_ids.clear();
}

Allen::Slices allocate_slices(
  size_t n_slices,
  std::unordered_set<BankTypes> const& bank_types,
  std::function<std::tuple<size_t, size_t, size_t>(BankTypes)> size_fun)
{
  Allen::Slices slices;

  // Create empty slices for all subdetectors
  for (auto& bank_slices : slices) {
    bank_slices.resize(n_slices);
  }

  // Allocate memory only for those subdetectors that are requested
  for (auto bank_type : bank_types) {
    auto [n_bytes, n_sizes, n_offsets] = size_fun(bank_type);

    auto ib = to_integral(bank_type);
    auto& bank_slices = slices[ib];
    for (size_t i = 0; i < n_slices; ++i) {
      char* events_mem = nullptr;
      unsigned* sizes_mem = nullptr;
      unsigned* types_mem = nullptr;
      unsigned* offsets_mem = nullptr;

      if (n_bytes) Allen::malloc_host((void**) &events_mem, n_bytes);
      if (n_sizes) {
        Allen::malloc_host((void**) &sizes_mem, n_sizes * sizeof(unsigned));
        Allen::malloc_host((void**) &types_mem, n_sizes * sizeof(unsigned));
      }
      if (n_offsets) Allen::malloc_host((void**) &offsets_mem, (n_offsets + 1) * sizeof(unsigned));

      for (size_t i = 0; i < n_offsets + 1; ++i) {
        offsets_mem[i] = 0;
      }
      for (size_t i = 0; i < n_sizes; ++i) {
        sizes_mem[i] = 0;
        types_mem[i] = 0;
      }
      std::vector<gsl::span<char>> bank_spans {};
      if (n_bytes) {
        bank_spans.emplace_back(events_mem, n_bytes);
      }

      bank_slices[i] = Allen::Slice {std::move(bank_spans),
                                     offsets_span {offsets_mem, static_cast<offsets_size>(n_offsets + 1)},
                                     n_bytes,
                                     1,
                                     offsets_span {sizes_mem, static_cast<offsets_size>(n_sizes)},
                                     offsets_span {types_mem, static_cast<offsets_size>(n_sizes)}};
    }
  }
  return slices;
}

void free_slices(Allen::Slices& slices)
{
  for (auto& bank_slices : slices) {
    for (auto& slice : bank_slices) {
      if (!slice.fragments.empty() && !slice.fragments[0].empty()) Allen::free_host(slice.fragments[0].data());
      if (!slice.offsets.empty()) Allen::free_host(slice.offsets.data());
      if (!slice.sizes.empty()) Allen::free_host(slice.sizes.data());
    }
  }
}
