/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#pragma once

#include <string>
#include <ArgumentOps.cuh>
#include <DeterministicScaler.cuh>
#include "Event/ODIN.h"
#include "ODINBank.cuh"
#include "LHCbIDContainer.cuh"
#include "AlgorithmTypes.cuh"

// Helper macro to explicitly instantiate lines
#define INSTANTIATE_LINE(LINE, PARAMETERS)          \
  template void Line<LINE, PARAMETERS>::operator()( \
    const ArgumentReferences<PARAMETERS>&,          \
    const RuntimeOptions&,                          \
    const Constants&,                               \
    HostBuffers&,                                   \
    const Allen::Context&) const;                   \
  INSTANTIATE_ALGORITHM(LINE)

// "Enum of types" to determine dispatch to global_function
namespace LineIteration {
  struct default_iteration_tag {
  };
  struct event_iteration_tag {
  };
} // namespace LineIteration

/**
 * @brief A generic Line.
 * @detail It assumes the line has the following parameters:
 *
 *  HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
 *  MASK_INPUT(dev_event_list_t) dev_event_list;
 *  MASK_OUTPUT(dev_selected_events_t) dev_selected_events;
 *  HOST_OUTPUT(host_selected_events_size_t, unsigned) host_selected_events_size;
 *  DEVICE_OUTPUT(dev_selected_events_size_t, unsigned) dev_selected_events_size;
 *  DEVICE_INPUT(dev_odin_raw_input_t, char) dev_odin_raw_input;
 *  DEVICE_INPUT(dev_odin_raw_input_offsets_t, unsigned) dev_odin_raw_input_offsets;
 *  DEVICE_INPUT(dev_mep_layout_t, unsigned) dev_mep_layout;
 *  DEVICE_OUTPUT(dev_decisions_t, bool) dev_decisions;
 *  DEVICE_OUTPUT(dev_decisions_offsets_t, unsigned) dev_decisions_offsets;
 *  HOST_OUTPUT(host_post_scaler_t, float) host_post_scaler;
 *  HOST_OUTPUT(host_post_scaler_hash_t, uint32_t) host_post_scaler_hash;
 *  PROPERTY(pre_scaler_t, "pre_scaler", "Pre-scaling factor", float) pre_scaler;
 *  PROPERTY(post_scaler_t, "post_scaler", "Post-scaling factor", float) post_scaler;
 *  PROPERTY(pre_scaler_hash_string_t, "pre_scaler_hash_string", "Pre-scaling hash string", std::string);
 *  PROPERTY(post_scaler_hash_string_t, "post_scaler_hash_string", "Post-scaling hash string", std::string);
 *
 * The inheriting line must also provide the following methods:
 *
 *     __device__ unsigned offset(const Parameters& parameters, const unsigned event_number) const;
 *
 *     unsigned get_decisions_size(ArgumentReferences<Parameters>& arguments) const;
 *
 *     __device__ std::tuple<types...>
 *     get_input(const Parameters& parameters, const unsigned event_number, const unsigned i) const;
 *
 *     __device__ bool select(const Parameters& parameters, std::tuple<types...> input) const;
 *
 *     where "types..." is a list of types that can be freely configured.
 *
 *
 * The following methods can optionally be defined in an inheriting class:
 *
 *     unsigned get_grid_dim_x(const ArgumentReferences<Parameters>&) const;
 *
 *     unsigned get_block_dim_x(const ArgumentReferences<Parameters>&) const;
 */
template<typename Derived, typename Parameters>
struct Line {
private:
  uint32_t m_pre_scaler_hash;
  uint32_t m_post_scaler_hash;

public:
  using iteration_t = LineIteration::default_iteration_tag;
  constexpr static auto lhcbid_container = LHCbIDContainer::none;
  constexpr static auto has_particle_container = false;

  void init()
  {
    auto derived_instance = static_cast<const Derived*>(this);
    const std::string pre_scaler_hash_string =
      derived_instance->template property<typename Parameters::pre_scaler_hash_string_t>();
    const std::string post_scaler_hash_string =
      derived_instance->template property<typename Parameters::post_scaler_hash_string_t>();

    if (pre_scaler_hash_string.empty() || post_scaler_hash_string.empty()) {
      throw HashNotPopulatedException(derived_instance->name());
    }

    m_pre_scaler_hash = mixString(pre_scaler_hash_string.size(), pre_scaler_hash_string);
    m_post_scaler_hash = mixString(post_scaler_hash_string.size(), post_scaler_hash_string);
  }

  void set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&,
    const HostBuffers&) const
  {
    set_size<typename Parameters::dev_decisions_t>(arguments, Derived::get_decisions_size(arguments));
    set_size<typename Parameters::dev_decisions_offsets_t>(
      arguments, first<typename Parameters::host_number_of_events_t>(arguments));
    set_size<typename Parameters::dev_selected_events_t>(
      arguments, first<typename Parameters::host_number_of_events_t>(arguments));
    set_size<typename Parameters::host_post_scaler_t>(arguments, 1);
    set_size<typename Parameters::host_post_scaler_hash_t>(arguments, 1);
    set_size<typename Parameters::host_lhcbid_container_t>(arguments, 1);
    set_size<typename Parameters::host_selected_events_size_t>(arguments, 1);
    set_size<typename Parameters::dev_selected_events_size_t>(arguments, 1);
    set_size<typename Parameters::host_particle_container_t>(arguments, 1);
  }

  void operator()(
    const ArgumentReferences<Parameters>&,
    const RuntimeOptions&,
    const Constants&,
    HostBuffers&,
    const Allen::Context& context) const;

  /**
   * @brief Grid dimension of kernel call. get_grid_dim returns the size of the event list.
   */
  static unsigned get_grid_dim_x(const ArgumentReferences<Parameters>& arguments)
  {
    if constexpr (std::is_same<typename Derived::iteration_t, LineIteration::default_iteration_tag>::value) {
      return size<typename Parameters::dev_event_list_t>(arguments);
    }
    // else if (std::is_same<typename Derived::iteration_t, LineIteration::event_iteration_tag>::value) {
    return 1;
  }

  /**
   * @brief Default block dim x of kernel call. Can be "overriden".
   */
  static unsigned get_block_dim_x(const ArgumentReferences<Parameters>&) { return 256; }

  /**
   * @brief Default monitor function.
   */
  // template<typename... Args>
  // static __device__  void monitor(Args...) {}

  // template<typename... Args>
  // static __host__ void output_monitor(Args...) {}
  static void init_monitor(const ArgumentReferences<Parameters>&, const Allen::Context&) {}
  template<typename INPUT>
  static __device__ void monitor(const Parameters&, INPUT, unsigned, bool)
  {}
  static __host__ void
  output_monitor(const ArgumentReferences<Parameters>&, const RuntimeOptions&, const Allen::Context&)
  {}
};

#if defined(DEVICE_COMPILER)

/**
 * @brief Processes a line by iterating over all events and all "input sizes" (ie. tracks, vertices, etc.).
 *        The way process line parallelizes is highly configurable.
 */
template<typename Derived, typename Parameters>
__global__ void process_line(Parameters parameters, const unsigned number_of_events, const unsigned pre_scaler_hash)
{
  __shared__ int event_decision;
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  // const unsigned input_size = Derived::offset(parameters, event_number + 1) - Derived::offset(parameters, event_number);
  const unsigned input_size = Derived::input_size(parameters, event_number);

  if (threadIdx.x == 0) {
    event_decision = 0;
  }

  __syncthreads();

  // ODIN data
  const LHCb::ODIN odin {
    {*parameters.dev_mep_layout ?
       odin_data_mep_t::data(parameters.dev_odin_raw_input, parameters.dev_odin_raw_input_offsets, event_number) :
       odin_data_t::data(parameters.dev_odin_raw_input, parameters.dev_odin_raw_input_offsets, event_number),
     10}};

  const uint32_t run_no = odin.runNumber();
  const uint32_t evt_hi = static_cast<uint32_t>(odin.eventNumber() >> 32);
  const uint32_t evt_lo = static_cast<uint32_t>(odin.eventNumber() & 0xffffffff);
  const uint32_t gps_hi = static_cast<uint32_t>(odin.gpsTime() >> 32);
  const uint32_t gps_lo = static_cast<uint32_t>(odin.gpsTime() & 0xffffffff);

  bool thread_local_event_decision = false;
  // Pre-scaler
  if (deterministic_scaler(pre_scaler_hash, parameters.pre_scaler, run_no, evt_hi, evt_lo, gps_hi, gps_lo)) {
    // Do selection
    for (unsigned i = threadIdx.x; i < input_size; i += blockDim.x) {
      auto input = Derived::get_input(parameters, event_number, i);
      bool sel = Derived::select(parameters, input);
      unsigned index = Derived::offset(parameters, event_number) + i;
      parameters.dev_decisions[index] = sel;
      Derived::monitor(parameters, input, index, sel);
      thread_local_event_decision |= sel;
    }
  }

  // Populate offsets in first block
  if (blockIdx.x == 0) {
    for (unsigned i = threadIdx.x; i < number_of_events; i += blockDim.x) {
      parameters.dev_decisions_offsets[i] = Derived::offset(parameters, i);
    }
  }

  // Note: This could be done more efficiently with warp intrinsics
  atomicOr(&event_decision, thread_local_event_decision);

  // Synchronize the event_decision
  __syncthreads();

  if (threadIdx.x == 0 && event_decision) {
    const auto index = atomicAdd(parameters.dev_selected_events_size.get(), 1);
    parameters.dev_selected_events[index] = mask_t {event_number};
  }
}

/**
 * @brief Processes a line by iterating over events and applying the line.
 */
template<typename Derived, typename Parameters>
__global__ void process_line_iterate_events(
  Parameters parameters,
  const unsigned number_of_events_in_event_list,
  const unsigned number_of_events,
  const unsigned pre_scaler_hash)
{
  // Do selection
  for (unsigned i = threadIdx.x; i < number_of_events_in_event_list; i += blockDim.x) {
    const auto event_number = parameters.dev_event_list[i];

    // ODIN data
    const LHCb::ODIN odin {
      {*parameters.dev_mep_layout ?
         odin_data_mep_t::data(parameters.dev_odin_raw_input, parameters.dev_odin_raw_input_offsets, event_number) :
         odin_data_t::data(parameters.dev_odin_raw_input, parameters.dev_odin_raw_input_offsets, event_number),
       10}};

    const uint32_t run_no = odin.runNumber();
    const uint32_t evt_hi = static_cast<uint32_t>(odin.eventNumber() >> 32);
    const uint32_t evt_lo = static_cast<uint32_t>(odin.eventNumber() & 0xffffffff);
    const uint32_t gps_hi = static_cast<uint32_t>(odin.gpsTime() >> 32);
    const uint32_t gps_lo = static_cast<uint32_t>(odin.gpsTime() & 0xffffffff);

    bool decision = false;
    if (deterministic_scaler(pre_scaler_hash, parameters.pre_scaler, run_no, evt_hi, evt_lo, gps_hi, gps_lo)) {
      auto input = Derived::get_input(parameters, event_number);
      decision = Derived::select(parameters, input);
      parameters.dev_decisions[event_number] = decision;
      Derived::monitor(parameters, input, event_number, decision);
    }

    if (decision) {
      const auto index = atomicAdd(parameters.dev_selected_events_size.get(), 1);
      parameters.dev_selected_events[index] = mask_t {event_number};
    }
  }

  // Populate offsets
  for (unsigned event_number = threadIdx.x; event_number < number_of_events; event_number += blockDim.x) {
    parameters.dev_decisions_offsets[event_number] = event_number;
  }
}

template<typename Derived, typename Parameters, typename GlobalFunctionDispatch>
struct LineIterationDispatch;

template<typename Derived, typename Parameters>
struct LineIterationDispatch<Derived, Parameters, LineIteration::default_iteration_tag> {
  static void dispatch(
    const ArgumentReferences<Parameters>& arguments,
    const Allen::Context& context,
    const Derived* derived_instance,
    const unsigned grid_dim_x,
    const unsigned pre_scaler_hash)
  {
    derived_instance->global_function(process_line<Derived, Parameters>)(
      grid_dim_x, Derived::get_block_dim_x(arguments), context)(
      arguments, first<typename Parameters::host_number_of_events_t>(arguments), pre_scaler_hash);
  }
};

template<typename Derived, typename Parameters>
struct LineIterationDispatch<Derived, Parameters, LineIteration::event_iteration_tag> {
  static void dispatch(
    const ArgumentReferences<Parameters>& arguments,
    const Allen::Context& context,
    const Derived* derived_instance,
    const unsigned grid_dim_x,
    const unsigned pre_scaler_hash)
  {
    derived_instance->global_function(process_line_iterate_events<Derived, Parameters>)(
      grid_dim_x, Derived::get_block_dim_x(arguments), context)(
      arguments,
      size<typename Parameters::dev_event_list_t>(arguments),
      first<typename Parameters::host_number_of_events_t>(arguments),
      pre_scaler_hash);
  }
};

template<typename Derived, typename Parameters>
void Line<Derived, Parameters>::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  HostBuffers&,
  const Allen::Context& context) const
{
  initialize<typename Parameters::dev_decisions_t>(arguments, 0, context);
  initialize<typename Parameters::dev_decisions_offsets_t>(arguments, 0, context);
  initialize<typename Parameters::dev_selected_events_size_t>(arguments, 0, context);

  // Populate container with tag.
  data<typename Parameters::host_lhcbid_container_t>(arguments)[0] = to_integral(Derived::lhcbid_container);
  if constexpr (Derived::has_particle_container) {
    data<typename Parameters::host_particle_container_t>(arguments)[0] = data<typename Parameters::dev_particle_container_t>(arguments);
  }

  const auto* derived_instance = static_cast<const Derived*>(this);

  // Copy post scaler and hash to an output, such that GatherSelections can later
  // perform the postscaling
  data<typename Parameters::host_post_scaler_t>(arguments)[0] =
    derived_instance->template property<typename Parameters::post_scaler_t>();
  data<typename Parameters::host_post_scaler_hash_t>(arguments)[0] = m_post_scaler_hash;

  Derived::init_monitor(arguments, context);

  // Dispatch the executing global function.
  LineIterationDispatch<Derived, Parameters, typename Derived::iteration_t>::dispatch(
    arguments, context, derived_instance, Derived::get_grid_dim_x(arguments), m_pre_scaler_hash);

  Allen::copy<typename Parameters::host_selected_events_size_t, typename Parameters::dev_selected_events_size_t>(
    arguments, context);
  reduce_size<typename Parameters::dev_selected_events_t>(
    arguments, first<typename Parameters::host_selected_events_size_t>(arguments));

  derived_instance->output_monitor(arguments, runtime_options, context);
}

#endif
