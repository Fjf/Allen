/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "Stream.cuh"
#include "StreamWrapper.cuh"

// Include the sequence checker specializations
#include "VeloSequenceCheckers_impl.cuh"
#include "UTSequenceCheckers_impl.cuh"
#include "SciFiSequenceCheckers_impl.cuh"
#include "PVSequenceCheckers_impl.cuh"
#include "KalmanSequenceCheckers_impl.cuh"
#include "RateCheckers_impl.cuh"

StreamWrapper::StreamWrapper() {}

void StreamWrapper::initialize_streams(
  const unsigned n,
  const bool print_memory_usage,
  const unsigned start_event_offset,
  const size_t reserve_mb,
  const size_t reserve_host_mb,
  const unsigned required_memory_alignment,
  const Constants& constants,
  const std::map<std::string, std::map<std::string, std::string>>& config)
{
  for (unsigned i = 0; i < n; ++i) {
    streams.push_back(new Stream());
  }

  for (size_t i = 0; i < streams.size(); ++i) {
    streams[i]->initialize(
      print_memory_usage, start_event_offset, reserve_mb, reserve_host_mb, required_memory_alignment, constants);

    // Configuration of the algorithms must happen after stream initialization
    streams[i]->configure_algorithms(config);
  }
}

void StreamWrapper::initialize_streams_host_buffers_manager(HostBuffersManager* buffers_manager)
{
  for (size_t i = 0; i < streams.size(); ++i) {
    streams[i]->set_host_buffer_manager(buffers_manager);
  }
}

Allen::error StreamWrapper::run_stream(const unsigned i, const unsigned buf_idx, const RuntimeOptions& runtime_options)
{
  return streams[i]->run_sequence(buf_idx, runtime_options);
}

void StreamWrapper::run_monte_carlo_test(
  unsigned const i,
  CheckerInvoker& invoker,
  MCEvents const& mc_events,
  std::vector<Checker::Tracks> const& forward_tracks)
{
  streams[i]->run_monte_carlo_test(invoker, mc_events, forward_tracks);
}

std::map<std::string, std::map<std::string, std::string>> StreamWrapper::get_algorithm_configuration()
{
  return streams.front()->get_algorithm_configuration();
}

StreamWrapper::~StreamWrapper()
{
  for (auto& stream : streams) {
    delete stream;
  }
}

void print_configured_sequence()
{
  info_cout << "\nConfigured sequence of algorithms:\n";
  Sch::PrintAlgorithmSequence<configured_sequence_t>::print();
  info_cout << std::endl;
}

/**
 * @brief Sets up the chain that will be executed later.
 */
Allen::error Stream::initialize(
  const bool param_do_print_memory_manager,
  const unsigned param_start_event_offset,
  const size_t reserve_mb,
  const size_t reserve_host_mb,
  const unsigned required_memory_alignment,
  const Constants& param_constants)
{
  // Set options
  do_print_memory_manager = param_do_print_memory_manager;
  start_event_offset = param_start_event_offset;
  constants = param_constants;

  // Initialize context
  m_context.initialize();

  // Prepare scheduler
  scheduler.initialize(do_print_memory_manager, reserve_mb, reserve_host_mb, required_memory_alignment);

  // Populate names of parameters in the sequence
  populate_sequence_argument_names(scheduler.argument_manager);

  return Allen::error::success;
}

void Stream::set_host_buffer_manager(HostBuffersManager* buffers_manager)
{
  // Set host buffers manager
  host_buffers_manager = buffers_manager;
}

Allen::error Stream::run_sequence(const unsigned buf_idx, const RuntimeOptions& runtime_options)
{
  host_buffers = host_buffers_manager->getBuffers(buf_idx);
  // The sequence is only run if there are events to run on
  auto event_start = std::get<0>(runtime_options.event_interval);
  auto event_end = std::get<1>(runtime_options.event_interval);

  number_of_input_events = event_end - event_start;
  if (event_end > event_start) {
    for (unsigned repetition = 0; repetition < runtime_options.number_of_repetitions; ++repetition) {
      // Initialize selected_number_of_events with requested_number_of_events
      host_buffers->host_number_of_events = event_end - event_start;

      // Reset scheduler
      scheduler.reset();

      try {
        // Visit all algorithms in configured sequence
        scheduler.run(runtime_options, constants, host_buffers, m_context);

        // deterministic injection of ~random memory failures
        if (runtime_options.inject_mem_fail > 0) {
          // compare the least significant N bits of two ~unrelated buffers
          // test should fire one time in 2^N slices on average
          // limit ourselves to a maximum of 15-bit comparison (1/2 - ~1/32k of slices)
          uint test_mask = (1 << 15) - 1;
          if (runtime_options.inject_mem_fail < 15) test_mask = (1 << runtime_options.inject_mem_fail) - 1;
          if (
            (host_buffers->host_number_of_selected_events & test_mask) ==
            (host_buffers->host_total_number_of_velo_clusters[0] & test_mask))
            throw MemoryException("Test : Injected fake memory exception to test failure handling");
        }

        // Synchronize device
        Allen::synchronize(m_context);
      } catch (const MemoryException& e) {
        warning_cout << "Insufficient memory to process slice - will sub-divide and retry." << std::endl;
        return Allen::error::errorMemoryAllocation;
      }
    }
  }

  return Allen::error::success;
}

void Stream::run_monte_carlo_test(
  CheckerInvoker& invoker,
  MCEvents const& mc_events,
  std::vector<Checker::Tracks> const& forward_tracks)
{
  Sch::RunChecker<configured_sequence_t>::check(*host_buffers, constants, invoker, mc_events);

  if (forward_tracks.size() > 0) {
    info_cout << "Running test on imported tracks" << std::endl;
    auto& checker = invoker.checker<TrackCheckerForward>("PrCheckerPlots.root");
    checker.accumulate<TrackCheckerForward>(mc_events, forward_tracks);
  }
}
