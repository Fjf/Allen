/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "GatherSelections.cuh"
#include "SelectionsEventModel.cuh"
#include "DeterministicScaler.cuh"
#include "Event/ODIN.h"
#include "ODINBank.cuh"
#include <algorithm>

INSTANTIATE_ALGORITHM(gather_selections::gather_selections_t)

namespace gather_selections {
  __global__ void postscaler(
    bool* dev_selections,
    const unsigned* dev_selections_offsets,
    const char* dev_odin_raw_input,
    const unsigned* dev_odin_raw_input_offsets,
    const float* scale_factors,
    const uint32_t* scale_hashes,
    const uint32_t* dev_mep_layout,
    const unsigned number_of_lines)
  {
    const auto number_of_events = gridDim.x;
    const auto event_number = blockIdx.x;

    Selections::Selections sels {dev_selections, dev_selections_offsets, number_of_events};

    const LHCb::ODIN odin {{*dev_mep_layout ?
                              odin_data_mep_t::data(dev_odin_raw_input, dev_odin_raw_input_offsets, event_number) :
                              odin_data_t::data(dev_odin_raw_input, dev_odin_raw_input_offsets, event_number),
                            10}};

    const uint32_t run_no = odin.runNumber();
    const uint32_t evt_hi = static_cast<uint32_t>(odin.eventNumber() >> 32);
    const uint32_t evt_lo = static_cast<uint32_t>(odin.eventNumber() & 0xffffffff);
    const uint32_t gps_hi = static_cast<uint32_t>(odin.gpsTime() >> 32);
    const uint32_t gps_lo = static_cast<uint32_t>(odin.gpsTime() & 0xffffffff);

    for (unsigned i = threadIdx.x; i < number_of_lines; i += blockDim.x) {
      auto span = sels.get_span(i, event_number);
      deterministic_post_scaler(
        scale_hashes[i], scale_factors[i], span.size(), span.data(), run_no, evt_hi, evt_lo, gps_hi, gps_lo);
    }
  }
} // namespace gather_selections

void gather_selections::gather_selections_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  // Sum all the sizes from input selections
  const auto sum_sizes_from_aggregate = [](const auto& agg) {
    size_t total_size = 0;
    for (size_t i = 0; i < agg.size_of_aggregate(); ++i) {
      total_size += agg.size(i);
    }
    return total_size;
  };

  const auto dev_input_selections = input_aggregate<dev_input_selections_t>(arguments);
  const auto total_size_dev_input_selections = sum_sizes_from_aggregate(dev_input_selections);
  const auto total_size_host_input_post_scale_factors =
    sum_sizes_from_aggregate(input_aggregate<host_input_post_scale_factors_t>(arguments));
  const auto host_input_post_scale_hashes =
    sum_sizes_from_aggregate(input_aggregate<host_input_post_scale_hashes_t>(arguments));
  const auto host_lhcbid_containers_agg = input_aggregate<host_lhcbid_containers_agg_t>(arguments);

  set_size<host_number_of_active_lines_t>(arguments, 1);
  set_size<dev_number_of_active_lines_t>(arguments, 1);
  set_size<host_names_of_active_lines_t>(arguments, std::string(property<names_of_active_lines_t>().get()).size() + 1);
  set_size<host_selections_lines_offsets_t>(arguments, dev_input_selections.size_of_aggregate() + 1);
  set_size<host_selections_offsets_t>(
    arguments, first<host_number_of_events_t>(arguments) * dev_input_selections.size_of_aggregate() + 1);
  set_size<dev_selections_offsets_t>(
    arguments, first<host_number_of_events_t>(arguments) * dev_input_selections.size_of_aggregate() + 1);
  set_size<dev_selections_t>(arguments, total_size_dev_input_selections);
  set_size<host_post_scale_factors_t>(arguments, total_size_host_input_post_scale_factors);
  set_size<host_post_scale_hashes_t>(arguments, host_input_post_scale_hashes);
  set_size<dev_post_scale_factors_t>(arguments, total_size_host_input_post_scale_factors);
  set_size<dev_post_scale_hashes_t>(arguments, host_input_post_scale_hashes);
  set_size<dev_lhcbid_containers_t>(arguments, host_lhcbid_containers_agg.size_of_aggregate());
  set_size<host_lhcbid_containers_t>(arguments, host_lhcbid_containers_agg.size_of_aggregate());

  if (property<verbosity_t>() >= logger::debug) {
    info_cout << "Sizes of gather_selections datatypes: " << size<host_selections_offsets_t>(arguments) << ", "
              << size<host_selections_lines_offsets_t>(arguments) << ", " << size<dev_selections_offsets_t>(arguments)
              << ", " << size<dev_selections_t>(arguments) << "\n";
  }
}

void gather_selections::gather_selections_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  HostBuffers& host_buffers,
  const Allen::Context& context) const
{
  // Save the names of active lines as output
  initialize<host_names_of_active_lines_t>(arguments, 0, context);
  const auto line_names = std::string(property<names_of_active_lines_t>());
  line_names.copy(data<host_names_of_active_lines_t>(arguments), line_names.size());

  // Pass the number of lines for posterior algorithms
  const auto dev_input_selections = input_aggregate<dev_input_selections_t>(arguments);
  data<host_number_of_active_lines_t>(arguments)[0] = dev_input_selections.size_of_aggregate();
  Allen::copy_async<dev_number_of_active_lines_t, host_number_of_active_lines_t>(arguments, context);

  // Calculate prefix sum of dev_input_selections_t sizes into host_selections_lines_offsets_t
  auto* container = data<host_selections_lines_offsets_t>(arguments);
  container[0] = 0;
  for (size_t i = 0; i < dev_input_selections.size_of_aggregate(); ++i) {
    container[i + 1] = container[i] + dev_input_selections.size(i);
  }

  // Populate the list of LHCbID containers?
  Allen::aggregate::store_contiguous_async<host_lhcbid_containers_t, host_lhcbid_containers_agg_t>(arguments, context);
  Allen::copy_async<dev_lhcbid_containers_t, host_lhcbid_containers_t>(arguments, context);

  // Populate dev_selections_t
  Allen::aggregate::store_contiguous_async<dev_selections_t, dev_input_selections_t>(arguments, context);

  // Copy dev_input_selections_offsets_t onto host_selections_lines_offsets_t
  Allen::aggregate::store_contiguous_async<host_selections_offsets_t, dev_input_selections_offsets_t>(
    arguments, context);

  // Populate host_post_scale_factors_t
  Allen::aggregate::store_contiguous_async<host_post_scale_factors_t, host_input_post_scale_factors_t>(
    arguments, context);

  // Populate host_post_scale_hashes_t
  Allen::aggregate::store_contiguous_async<host_post_scale_hashes_t, host_input_post_scale_hashes_t>(
    arguments, context);

  // Copy host_post_scale_factors_t to dev_post_scale_factors_t,
  // and host_post_scale_hashes_t to dev_post_scale_hashes_t
  Allen::copy_async<dev_post_scale_factors_t, host_post_scale_factors_t>(arguments, context);
  Allen::copy_async<dev_post_scale_hashes_t, host_post_scale_hashes_t>(arguments, context);

  // Synchronize after all the copies above
  Allen::synchronize(context);

  // Add partial sums from host_selections_lines_offsets_t to host_selections_offsets_t
  for (unsigned line_index = 1; line_index < first<host_number_of_active_lines_t>(arguments); ++line_index) {
    const auto line_offset = data<host_selections_lines_offsets_t>(arguments)[line_index];
    for (unsigned i = 0; i < first<host_number_of_events_t>(arguments); ++i) {
      data<host_selections_offsets_t>(arguments)[line_index * first<host_number_of_events_t>(arguments) + i] +=
        line_offset;
    }
  }

  // Add to last element the total sum
  data<host_selections_offsets_t>(
    arguments)[first<host_number_of_active_lines_t>(arguments) * first<host_number_of_events_t>(arguments)] =
    data<host_selections_lines_offsets_t>(arguments)[dev_input_selections.size_of_aggregate()];

  // Copy host_selections_offsets_t onto dev_selections_offsets_t
  Allen::copy_async<dev_selections_offsets_t, host_selections_offsets_t>(arguments, context);

  // Run the postscaler
  global_function(postscaler)(first<host_number_of_events_t>(arguments), property<block_dim_x_t>().get(), context)(
    data<dev_selections_t>(arguments),
    data<dev_selections_offsets_t>(arguments),
    data<dev_odin_raw_input_t>(arguments),
    data<dev_odin_raw_input_offsets_t>(arguments),
    data<dev_post_scale_factors_t>(arguments),
    data<dev_post_scale_hashes_t>(arguments),
    data<dev_mep_layout_t>(arguments),
    first<host_number_of_active_lines_t>(arguments));

  if (property<verbosity_t>() >= logger::debug) {
    std::vector<uint8_t> host_selections(size<dev_selections_t>(arguments));
    assign_to_host_buffer<dev_selections_t>(host_selections.data(), arguments, context);
    Allen::copy<host_selections_offsets_t, dev_selections_offsets_t>(arguments, context);

    Selections::ConstSelections sels {reinterpret_cast<bool*>(host_selections.data()),
                                      data<host_selections_offsets_t>(arguments),
                                      first<host_number_of_events_t>(arguments)};

    std::vector<uint8_t> event_decisions {};
    for (auto i = 0u; i < first<host_number_of_events_t>(arguments); ++i) {
      bool dec = false;
      for (auto j = 0u; j < first<host_number_of_active_lines_t>(arguments); ++j) {
        auto decs = sels.get_span(j, i);
        std::cout << "Size of span (event " << i << ", line " << j << "): " << decs.size() << "\n";
        for (auto k = 0u; k < decs.size(); ++k) {
          dec |= decs[k];
        }
      }
      event_decisions.emplace_back(dec);
    }

    const float sum_events = std::accumulate(event_decisions.begin(), event_decisions.end(), 0);
    std::cout << sum_events / event_decisions.size() << std::endl;

    const float sum = std::accumulate(host_selections.begin(), host_selections.end(), 0);
    std::cout << sum / host_selections.size() << std::endl;
  }

  // If running the validation, save relevant information
  if (runtime_options.fill_extra_host_buffers) {
    host_buffers.host_names_of_lines = std::string(property<names_of_active_lines_t>());
    host_buffers.host_number_of_lines = first<host_number_of_active_lines_t>(arguments);
    safe_assign_to_host_buffer<dev_selections_t>(host_buffers.host_selections, arguments, context);
    safe_assign_to_host_buffer<dev_selections_offsets_t>(host_buffers.host_selections_offsets, arguments, context);
  }
}
