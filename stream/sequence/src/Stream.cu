#include "Stream.cuh"

/**
 * @brief Sets up the chain that will be executed later.
 */
cudaError_t Stream::initialize(
  const std::vector<char>& velopix_geometry,
  const std::vector<char>& ut_boards,
  const std::vector<char>& ut_geometry,
  const std::vector<char>& ut_magnet_tool,
  const std::vector<char>& scifi_geometry,
  const uint max_number_of_events,
  const bool param_do_check,
  const bool param_do_simplified_kalman_filter,
  const bool param_do_print_memory_manager,
  const bool param_run_on_x86,
  const std::string& param_folder_name_MC,
  const uint param_start_event_offset,
  const size_t reserve_mb,
  const uint param_stream_number,
  const Constants& param_constants
) {
  // Set stream and events
  cudaCheck(cudaStreamCreate(&cuda_stream));
  cudaCheck(cudaEventCreate(&cuda_generic_event));
  cudaCheck(cudaEventCreate(&cuda_event_start));
  cudaCheck(cudaEventCreate(&cuda_event_stop));

  // Set stream options
  stream_number = param_stream_number;
  do_check = param_do_check;
  do_simplified_kalman_filter = param_do_simplified_kalman_filter;
  do_print_memory_manager = param_do_print_memory_manager;
  run_on_x86 = param_run_on_x86;
  folder_name_MC = param_folder_name_MC;
  start_event_offset = param_start_event_offset;
  constants = param_constants;

  // Special case
  // Populate velo geometry
  cudaCheck(cudaMalloc((void**)&dev_velo_geometry, velopix_geometry.size()));
  cudaCheck(cudaMemcpyAsync(dev_velo_geometry, velopix_geometry.data(), velopix_geometry.size(), cudaMemcpyHostToDevice, cuda_stream));

  // Populate UT boards and geometry
  cudaCheck(cudaMalloc((void**)&dev_ut_boards, ut_boards.size()));
  cudaCheck(cudaMemcpyAsync(dev_ut_boards, ut_boards.data(), ut_boards.size(), cudaMemcpyHostToDevice, cuda_stream));

  cudaCheck(cudaMalloc((void**)&dev_ut_geometry, ut_geometry.size()));
  cudaCheck(cudaMemcpyAsync(dev_ut_geometry, ut_geometry.data(), ut_geometry.size(), cudaMemcpyHostToDevice, cuda_stream));

  // Populate UT magnet tool values
  cudaCheck(cudaMalloc((void**)&dev_ut_magnet_tool, ut_magnet_tool.size()));
  cudaCheck(cudaMemcpyAsync(dev_ut_magnet_tool, ut_magnet_tool.data(), ut_magnet_tool.size(), cudaMemcpyHostToDevice, cuda_stream));

  // Populate FT geometry
  cudaCheck(cudaMalloc((void**)&dev_scifi_geometry, scifi_geometry.size()));
  cudaCheck(cudaMemcpyAsync(dev_scifi_geometry, scifi_geometry.data(), scifi_geometry.size(), cudaMemcpyHostToDevice, cuda_stream));

  // Reserve host buffers
  host_buffers.reserve();

  // Define sequence of algorithms to execute
  sequence.set(sequence_algorithms());

  // Get dependencies for each algorithm
  std::vector<std::vector<int>> sequence_dependencies = get_sequence_dependencies();

  // Get output arguments from the sequence
  std::vector<int> sequence_output_arguments = get_sequence_output_arguments();

  // Prepare dynamic scheduler
  scheduler = {get_sequence_names(), get_argument_names(),
    sequence_dependencies, sequence_output_arguments,
    reserve_mb * 1024 * 1024, do_print_memory_manager};

  // Malloc a configurable reserved memory
  cudaCheck(cudaMalloc((void**)&dev_base_pointer, reserve_mb * 1024 * 1024));

  return cudaSuccess;
}

cudaError_t Stream::run_sequence(const RuntimeOptions& runtime_options) {
  for (uint repetition=0; repetition<runtime_options.number_of_repetitions; ++repetition) {
    // Generate object for populating arguments
    ArgumentManager<argument_tuple_t> arguments {dev_base_pointer};

    // Reset scheduler
    scheduler.reset();

    // Visit all algorithms in configured sequence
    run_sequence_tuple(
      stream_visitor,
      sequence_tuple,
      runtime_options,
      constants,
      arguments,
      host_buffers,
      cuda_stream,
      cuda_generic_event);
  }

  return cudaSuccess;
}

void Stream::run_monte_carlo_test(const RuntimeOptions& runtime_options) {
  std::cout << "Checking Velo tracks reconstructed on GPU" << std::endl;

  const std::vector<trackChecker::Tracks> tracks_events = prepareTracks(
    host_buffers.host_velo_tracks_atomics,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_velo_track_hits,
    runtime_options.number_of_events);

  call_pr_checker(
    tracks_events,
    folder_name_MC,
    start_event_offset,
    "Velo"
  );

  /* CHECKING VeloUT TRACKS */
  const std::vector<trackChecker::Tracks> veloUT_tracks = prepareVeloUTTracks(
    host_buffers.host_veloUT_tracks,
    host_buffers.host_atomics_veloUT,
    runtime_options.number_of_events
  );

  std::cout << "Checking VeloUT tracks reconstructed on GPU" << std::endl;
  call_pr_checker(
    veloUT_tracks,
    folder_name_MC,
    start_event_offset,
    "VeloUT"
  );
}
