#include "Stream.cuh"

#include <iostream>
#include <fstream>

cudaError_t Stream::run_sequence(
  const uint i_stream,
  const char* host_velopix_events,
  const uint* host_velopix_event_offsets,
  const size_t host_velopix_events_size,
  const size_t host_velopix_event_offsets_size,
  const char* host_ut_events,
  const uint* host_ut_event_offsets,
  const size_t host_ut_events_size,
  const size_t host_ut_event_offsets_size,
  const uint number_of_events,
  const uint number_of_repetitions
) {
  // Generate object for populating arguments
  DynamicArgumentGenerator<argument_tuple_t> argen {arguments, dev_base_pointer};

  // Sizes and offsets of arguments
  std::array<size_t, std::tuple_size<argument_tuple_t>::value> argument_sizes;
  std::array<uint, std::tuple_size<argument_tuple_t>::value> argument_offsets;

  for (uint repetition=0; repetition<number_of_repetitions; ++repetition) {
    uint sequence_step = 0;

    // Reset scheduler
    scheduler.reset();

    // Estimate input size
    // Set arguments and reserve memory
    argument_sizes[arg::dev_raw_input] = argen.size<arg::dev_raw_input>(host_velopix_events_size);
    argument_sizes[arg::dev_raw_input_offsets] = argen.size<arg::dev_raw_input_offsets>(host_velopix_event_offsets_size);
    argument_sizes[arg::dev_estimated_input_size] = argen.size<arg::dev_estimated_input_size>(number_of_events * VeloTracking::n_modules + 1);
    argument_sizes[arg::dev_module_cluster_num] = argen.size<arg::dev_module_cluster_num>(number_of_events * VeloTracking::n_modules);
    argument_sizes[arg::dev_module_candidate_num] = argen.size<arg::dev_raw_input_offsets>(number_of_events);
    argument_sizes[arg::dev_cluster_candidates] = argen.size<arg::dev_cluster_candidates>(number_of_events * VeloClustering::max_candidates_event);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    // Setup opts and arguments for kernel call
    sequence.item<seq::estimate_input_size>().set_opts(dim3(number_of_events), dim3(32, 26), stream);
    sequence.item<seq::estimate_input_size>().set_arguments(
      argen.generate<arg::dev_raw_input>(argument_offsets),
      argen.generate<arg::dev_raw_input_offsets>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_module_candidate_num>(argument_offsets),
      argen.generate<arg::dev_cluster_candidates>(argument_offsets),
      gpu_constants.dev_velo_candidate_ks
    );
    cudaCheck(cudaMemcpyAsync(argen.generate<arg::dev_raw_input>(argument_offsets), host_velopix_events, host_velopix_events_size, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(argen.generate<arg::dev_raw_input_offsets>(argument_offsets), host_velopix_event_offsets, host_velopix_event_offsets_size * sizeof(uint), cudaMemcpyHostToDevice, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Kernel call
    sequence.item<seq::estimate_input_size>().invoke();

    // Convert the estimated sizes to module hit start format (argument_offsets)
    // Set arguments and reserve memory
    argument_sizes[arg::dev_cluster_offset] = argen.size<arg::dev_cluster_offset>(number_of_events);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    // Setup sequence step
    const auto prefix_sum_blocks = (VeloTracking::n_modules * number_of_events + 511) / 512;
    sequence.item<seq::prefix_sum_reduce>().set_opts(dim3(prefix_sum_blocks), dim3(256), stream);
    sequence.item<seq::prefix_sum_reduce>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_cluster_offset>(argument_offsets),
      VeloTracking::n_modules * number_of_events
    );
    // Kernel call
    sequence.item<seq::prefix_sum_reduce>().invoke();

    // Prefix Sum Single Block
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::prefix_sum_single_block>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(argument_offsets) + VeloTracking::n_modules * number_of_events,
      argen.generate<arg::dev_cluster_offset>(argument_offsets),
      prefix_sum_blocks
    );
    sequence.item<seq::prefix_sum_single_block>().invoke();

    // Prefix sum scan
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    const auto prefix_sum_scan_blocks = prefix_sum_blocks==1 ? 1 : (prefix_sum_blocks-1);
    sequence.item<seq::prefix_sum_scan>().set_opts(dim3(prefix_sum_scan_blocks), dim3(512), stream);
    sequence.item<seq::prefix_sum_scan>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_cluster_offset>(argument_offsets),
      VeloTracking::n_modules * number_of_events
    );
    sequence.item<seq::prefix_sum_scan>().invoke();

    // Fetch the number of hits we require
    cudaCheck(cudaMemcpyAsync(host_total_number_of_velo_clusters, argen.generate<arg::dev_estimated_input_size>(argument_offsets) + number_of_events * VeloTracking::n_modules, sizeof(uint), cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Masked Velo clustering
    argument_sizes[arg::dev_velo_cluster_container] = argen.size<arg::dev_velo_cluster_container>(6 * host_total_number_of_velo_clusters[0]);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::masked_velo_clustering>().set_opts(dim3(number_of_events), dim3(256), stream);
    sequence.item<seq::masked_velo_clustering>().set_arguments(
      argen.generate<arg::dev_raw_input>(argument_offsets),
      argen.generate<arg::dev_raw_input_offsets>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_module_candidate_num>(argument_offsets),
      argen.generate<arg::dev_cluster_candidates>(argument_offsets),
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      dev_velo_geometry,
      gpu_constants.dev_velo_sp_patterns,
      gpu_constants.dev_velo_sp_fx,
      gpu_constants.dev_velo_sp_fy
    );
    sequence.item<seq::masked_velo_clustering>().invoke();

    // Calculate phi and sort
    argument_sizes[arg::dev_hit_permutation] = argen.size<arg::dev_hit_permutation>(host_total_number_of_velo_clusters[0]);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::calculate_phi_and_sort>().set_opts(dim3(number_of_events), dim3(64), stream);
    sequence.item<seq::calculate_phi_and_sort>().set_arguments(
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      argen.generate<arg::dev_hit_permutation>(argument_offsets)
    );
    sequence.item<seq::calculate_phi_and_sort>().invoke();

    // Fill candidates
    argument_sizes[arg::dev_h0_candidates] = argen.size<arg::dev_h0_candidates>(2 * host_total_number_of_velo_clusters[0]);
    argument_sizes[arg::dev_h2_candidates] = argen.size<arg::dev_h2_candidates>(2 * host_total_number_of_velo_clusters[0]);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    // Setup opts and arguments
    sequence.item<seq::fill_candidates>().set_opts(dim3(number_of_events, 48), dim3(128), stream);
    sequence.item<seq::fill_candidates>().set_arguments(
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_h0_candidates>(argument_offsets),
      argen.generate<arg::dev_h2_candidates>(argument_offsets)
    );
    sequence.item<seq::fill_candidates>().invoke();

    // Search by triplet
    argument_sizes[arg::dev_tracks] = argen.size<arg::dev_tracks>(number_of_events * VeloTracking::max_tracks);
    argument_sizes[arg::dev_tracklets] = argen.size<arg::dev_tracklets>(number_of_events * VeloTracking::ttf_modulo);
    argument_sizes[arg::dev_tracks_to_follow] = argen.size<arg::dev_tracks_to_follow>(number_of_events * VeloTracking::ttf_modulo);
    argument_sizes[arg::dev_weak_tracks] = argen.size<arg::dev_weak_tracks>(number_of_events * VeloTracking::max_weak_tracks);
    argument_sizes[arg::dev_hit_used] = argen.size<arg::dev_hit_used>(host_total_number_of_velo_clusters[0]);
    argument_sizes[arg::dev_atomics_storage] = argen.size<arg::dev_atomics_storage>(number_of_events * VeloTracking::num_atomics);
    argument_sizes[arg::dev_rel_indices] = argen.size<arg::dev_rel_indices>(number_of_events * 2 * VeloTracking::max_numhits_in_module);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    // Setup opts and arguments
    sequence.item<seq::search_by_triplet>().set_opts(dim3(number_of_events), dim3(32), stream, 32 * sizeof(float));
    sequence.item<seq::search_by_triplet>().set_arguments(
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_tracks>(argument_offsets),
      argen.generate<arg::dev_tracklets>(argument_offsets),
      argen.generate<arg::dev_tracks_to_follow>(argument_offsets),
      argen.generate<arg::dev_weak_tracks>(argument_offsets),
      argen.generate<arg::dev_hit_used>(argument_offsets),
      argen.generate<arg::dev_atomics_storage>(argument_offsets),
      argen.generate<arg::dev_h0_candidates>(argument_offsets),
      argen.generate<arg::dev_h2_candidates>(argument_offsets),
      argen.generate<arg::dev_rel_indices>(argument_offsets)
    );
    sequence.item<seq::search_by_triplet>().invoke();

    // Weak tracks adder
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    // Setup opts and arguments
    sequence.item<seq::weak_tracks_adder>().set_opts(dim3(number_of_events), dim3(32), stream);
    sequence.item<seq::weak_tracks_adder>().set_arguments(
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_tracks>(argument_offsets),
      argen.generate<arg::dev_weak_tracks>(argument_offsets),
      argen.generate<arg::dev_hit_used>(argument_offsets),
      argen.generate<arg::dev_atomics_storage>(argument_offsets)
    );
    sequence.item<seq::weak_tracks_adder>().invoke();
    
    // Calculate prefix sum of found tracks
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::copy_and_prefix_sum_single_block>().set_arguments(
      (uint*) argen.generate<arg::dev_atomics_storage>(argument_offsets) + number_of_events*2,
      (uint*) argen.generate<arg::dev_atomics_storage>(argument_offsets),
      (uint*) argen.generate<arg::dev_atomics_storage>(argument_offsets) + number_of_events,
      number_of_events
    );
    sequence.item<seq::copy_and_prefix_sum_single_block>().invoke();

    // Fetch number of reconstructed tracks
    cudaCheck(cudaMemcpyAsync(host_number_of_reconstructed_velo_tracks, argen.generate<arg::dev_atomics_storage>(argument_offsets) + number_of_events * 2, sizeof(uint), cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);
    size_t velo_track_hit_number_size = host_number_of_reconstructed_velo_tracks[0] + 1;

    // Prefix sum of tracks hits
    // 1. Copy velo track hit number to a consecutive container
    // 2. Reduce
    // 3. Single block
    // 4. Scan

    // Copy Velo track hit number
    argument_sizes[arg::dev_velo_track_hit_number] = argen.size<arg::dev_velo_track_hit_number>(velo_track_hit_number_size);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::copy_velo_track_hit_number>().set_opts(dim3(number_of_events), dim3(512), stream);
    sequence.item<seq::copy_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_tracks>(argument_offsets),
      argen.generate<arg::dev_atomics_storage>(argument_offsets),
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets)
    );
    sequence.item<seq::copy_velo_track_hit_number>().invoke();

    // Prefix sum: Reduce
    const size_t prefix_sum_auxiliary_array_2_size = (host_number_of_reconstructed_velo_tracks[0] + 511) / 512;
    argument_sizes[arg::dev_prefix_sum_auxiliary_array_2] = argen.size<arg::dev_prefix_sum_auxiliary_array_2>(prefix_sum_auxiliary_array_2_size);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::prefix_sum_reduce_velo_track_hit_number>().set_opts(dim3(prefix_sum_auxiliary_array_2_size), dim3(256), stream);
    sequence.item<seq::prefix_sum_reduce_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets),
      argen.generate<arg::dev_prefix_sum_auxiliary_array_2>(argument_offsets),
      host_number_of_reconstructed_velo_tracks[0]
    );
    sequence.item<seq::prefix_sum_reduce_velo_track_hit_number>().invoke();

    // Prefix sum: Single block
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::prefix_sum_single_block_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets) + host_number_of_reconstructed_velo_tracks[0],
      argen.generate<arg::dev_prefix_sum_auxiliary_array_2>(argument_offsets),
      prefix_sum_auxiliary_array_2_size
    );
    sequence.item<seq::prefix_sum_single_block_velo_track_hit_number>().invoke();

    // Prefix sum: Scan
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    const uint pss_velo_track_hit_number_opts =
      prefix_sum_auxiliary_array_2_size==1 ? 1 : (prefix_sum_auxiliary_array_2_size-1);
    sequence.item<seq::prefix_sum_scan_velo_track_hit_number>().set_opts(dim3(pss_velo_track_hit_number_opts), dim3(512), stream);
    sequence.item<seq::prefix_sum_scan_velo_track_hit_number>().set_arguments(
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets),
      argen.generate<arg::dev_prefix_sum_auxiliary_array_2>(argument_offsets),
      host_number_of_reconstructed_velo_tracks[0]
    );
    sequence.item<seq::prefix_sum_scan_velo_track_hit_number>().invoke();

    // Fetch total number of hits accumulated with all tracks
    cudaCheck(cudaMemcpyAsync(host_accumulated_number_of_hits_in_velo_tracks,
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets) + host_number_of_reconstructed_velo_tracks[0],
      sizeof(uint), cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // Consolidate tracks
    argument_sizes[arg::dev_velo_track_hits] = argen.size<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0]);
    argument_sizes[arg::dev_velo_states] = argen.size<arg::dev_velo_states>(host_number_of_reconstructed_velo_tracks[0]);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::consolidate_tracks>().set_opts(dim3(number_of_events), dim3(32), stream);
    sequence.item<seq::consolidate_tracks>().set_arguments(
      argen.generate<arg::dev_atomics_storage>(argument_offsets),
      argen.generate<arg::dev_tracks>(argument_offsets),
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets),
      argen.generate<arg::dev_velo_cluster_container>(argument_offsets),
      argen.generate<arg::dev_estimated_input_size>(argument_offsets),
      argen.generate<arg::dev_module_cluster_num>(argument_offsets),
      argen.generate<arg::dev_velo_track_hits>(argument_offsets),
      argen.generate<arg::dev_velo_states>(argument_offsets)
    );
    sequence.item<seq::consolidate_tracks>().invoke();

    // Estimate number of UT hits
    // Set arguments and reserve memory
    argument_sizes[arg::dev_ut_raw_input] = argen.size<arg::dev_ut_raw_input>(host_ut_events_size);
    argument_sizes[arg::dev_ut_raw_input_offsets] = argen.size<arg::dev_ut_raw_input_offsets>(host_ut_event_offsets_size);
    argument_sizes[arg::dev_ut_hit_count] = argen.size<arg::dev_ut_hit_count>(2 * number_of_events * VeloUTTracking::n_layers + 1);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    // Setup opts and arguments for kernel call
    cudaCheck(cudaMemcpyAsync(argen.generate<arg::dev_ut_raw_input>(argument_offsets), host_ut_events, host_ut_events_size, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(argen.generate<arg::dev_ut_raw_input_offsets>(argument_offsets), host_ut_event_offsets, host_ut_event_offsets_size * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);
    sequence.item<seq::ut_calculate_number_of_hits>().set_opts(dim3(number_of_events), dim3(192), stream);
    sequence.item<seq::ut_calculate_number_of_hits>().set_arguments(
      argen.generate<arg::dev_ut_raw_input>(argument_offsets),
      argen.generate<arg::dev_ut_raw_input_offsets>(argument_offsets),
      dev_ut_boards,
      argen.generate<arg::dev_ut_hit_count>(argument_offsets)
    );
    // Invoke kernel
    sequence.item<seq::ut_calculate_number_of_hits>().invoke();

    // // Print UT hit count per event per layer
    // cudaCheck(cudaMemcpyAsync(host_ut_hit_count, argen.generate<arg::dev_ut_hit_count>(argument_offsets), argen.size<arg::dev_ut_hit_count>(2 * number_of_events * VeloUTTracking::n_layers + 1), cudaMemcpyDeviceToHost, stream));
    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);
    // for (int e=0; e<number_of_events; ++e) {
    //   info_cout << "Event " << e << ", #hit per layer (0, 1, 2, 3): ";
    //   uint32_t* count = host_ut_hit_count + e * VeloUTTracking::n_layers;
    //   for (uint32_t i = 0; i < 4; ++i) {
    //     info_cout << count[i] << ", ";
    //   }
    //   info_cout << std::endl;
    // }

    // Prefix sum of hit count (becomes hit offset)
    // 1. Reduce
    // 2. Single block
    // 3. Scan

    // Prefix sum: Reduce
    const uint total_number_of_layers = number_of_events * VeloUTTracking::n_layers;
    const size_t prefix_sum_auxiliary_array_3_size = (total_number_of_layers + 511) / 512;
    argument_sizes[arg::dev_prefix_sum_auxiliary_array_3] = argen.size<arg::dev_prefix_sum_auxiliary_array_3>(prefix_sum_auxiliary_array_3_size);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::prefix_sum_reduce_ut_hits>().set_opts(dim3(prefix_sum_auxiliary_array_3_size), dim3(256), stream);
    sequence.item<seq::prefix_sum_reduce_ut_hits>().set_arguments(
      argen.generate<arg::dev_ut_hit_count>(argument_offsets),
      argen.generate<arg::dev_prefix_sum_auxiliary_array_3>(argument_offsets),
      total_number_of_layers
    );
    sequence.item<seq::prefix_sum_reduce_ut_hits>().invoke();

    // Prefix sum: Single block
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::prefix_sum_single_block_ut_hits>().set_arguments(
      argen.generate<arg::dev_ut_hit_count>(argument_offsets) + total_number_of_layers,
      argen.generate<arg::dev_prefix_sum_auxiliary_array_3>(argument_offsets),
      prefix_sum_auxiliary_array_3_size
    );
    sequence.item<seq::prefix_sum_single_block_ut_hits>().invoke();

    // Prefix sum: Scan
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    const uint pss_ut_hits_blocks = prefix_sum_auxiliary_array_3_size==1 ? 1 : (prefix_sum_auxiliary_array_3_size-1);
    sequence.item<seq::prefix_sum_scan_ut_hits>().set_opts(dim3(pss_ut_hits_blocks), dim3(512), stream);
    sequence.item<seq::prefix_sum_scan_ut_hits>().set_arguments(
      argen.generate<arg::dev_ut_hit_count>(argument_offsets),
      argen.generate<arg::dev_prefix_sum_auxiliary_array_3>(argument_offsets),
      total_number_of_layers
    );
    sequence.item<seq::prefix_sum_scan_ut_hits>().invoke();

    // Fetch total number of hits accumulated with all tracks
    cudaCheck(cudaMemcpyAsync(host_accumulated_number_of_ut_hits,
      argen.generate<arg::dev_ut_hit_count>(argument_offsets) + total_number_of_layers,
      sizeof(uint), cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    // // Now, we should have the offset instead, and the sum of all in host_accumulated_number_of_ut_hits
    // // Check that
    // cudaCheck(cudaMemcpyAsync(host_ut_hit_count, argen.generate<arg::dev_ut_hit_count>(argument_offsets), argen.size<arg::dev_ut_hit_count>(2 * number_of_events * VeloUTTracking::n_layers + 1), cudaMemcpyDeviceToHost, stream));
    // cudaEventRecord(cuda_generic_event, stream);
    // cudaEventSynchronize(cuda_generic_event);
    // for (int e=0; e<number_of_events; ++e) {
    //   info_cout << "Event " << e << ", offset per layer (0, 1, 2, 3): ";
    //   uint32_t* offset = host_ut_hit_count + e * VeloUTTracking::n_layers;
    //   for (uint32_t i = 0; i < 4; ++i) {
    //     info_cout << offset[i] << ", ";
    //   }
    //   info_cout << std::endl;
    // }
    // info_cout << "Total number of UT hits: " << *host_accumulated_number_of_ut_hits << std::endl;

    // Decode UT raw banks
    argument_sizes[arg::dev_ut_hits] = argen.size<arg::dev_ut_hits>(10 * host_accumulated_number_of_ut_hits[0]);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::decode_raw_banks>().set_opts(dim3(number_of_events), dim3(192), stream);
    sequence.item<seq::decode_raw_banks>().set_arguments(
      argen.generate<arg::dev_ut_raw_input>(argument_offsets),
      argen.generate<arg::dev_ut_raw_input_offsets>(argument_offsets),
      dev_ut_boards,
      dev_ut_geometry,
      argen.generate<arg::dev_ut_hits>(argument_offsets),
      argen.generate<arg::dev_ut_hit_count>(argument_offsets)
    );
    sequence.item<seq::decode_raw_banks>().invoke();

    // UT hit sorting by x
    argument_sizes[arg::dev_ut_hit_permutations] = argen.size<arg::dev_ut_hit_permutations>(host_accumulated_number_of_ut_hits[0]);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::sort_by_x>().set_opts(dim3(number_of_events), dim3(32), stream);
    sequence.item<seq::sort_by_x>().set_arguments(
      argen.generate<arg::dev_ut_hits>(argument_offsets),
      argen.generate<arg::dev_ut_hit_count>(argument_offsets),
      argen.generate<arg::dev_ut_hit_permutations>(argument_offsets)
    );
    sequence.item<seq::sort_by_x>().invoke();
    
    // VeloUT tracking
    argument_sizes[arg::dev_veloUT_tracks] = argen.size<arg::dev_veloUT_tracks>(number_of_events * VeloUTTracking::max_num_tracks);
    argument_sizes[arg::dev_atomics_veloUT] = argen.size<arg::dev_atomics_veloUT>(VeloUTTracking::num_atomics * number_of_events);
    scheduler.setup_next(argument_sizes, argument_offsets, sequence_step++);
    sequence.item<seq::veloUT>().set_opts(dim3(number_of_events), dim3(32), stream);
    sequence.item<seq::veloUT>().set_arguments(
      argen.generate<arg::dev_ut_hits>(argument_offsets),
      argen.generate<arg::dev_ut_hit_count>(argument_offsets),
      argen.generate<arg::dev_atomics_storage>(argument_offsets),
      argen.generate<arg::dev_velo_track_hit_number>(argument_offsets),
      argen.generate<arg::dev_velo_track_hits>(argument_offsets),
      argen.generate<arg::dev_velo_states>(argument_offsets),
      argen.generate<arg::dev_veloUT_tracks>(argument_offsets),
      argen.generate<arg::dev_atomics_veloUT>(argument_offsets),
      dev_ut_magnet_tool
    );
    sequence.item<seq::veloUT>().invoke();

    // Transmission device to host
    if (transmit_device_to_host) {
      // Velo tracks
      cudaCheck(cudaMemcpyAsync(host_number_of_tracks, argen.generate<arg::dev_atomics_storage>(argument_offsets), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_accumulated_tracks, argen.generate<arg::dev_atomics_storage>(argument_offsets) + number_of_events, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_velo_track_hit_number, argen.generate<arg::dev_velo_track_hit_number>(argument_offsets), argen.size<arg::dev_velo_track_hit_number>(velo_track_hit_number_size), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_velo_track_hits, argen.generate<arg::dev_velo_track_hits>(argument_offsets), argen.size<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0]), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_velo_states, argen.generate<arg::dev_velo_states>(argument_offsets), argen.size<arg::dev_velo_states>(host_number_of_reconstructed_velo_tracks[0]), cudaMemcpyDeviceToHost, stream)); 

      // VeloUT tracks
      cudaCheck(cudaMemcpyAsync(host_atomics_veloUT, argen.generate<arg::dev_atomics_veloUT>(argument_offsets), argen.size<arg::dev_atomics_veloUT>(VeloUTTracking::num_atomics*number_of_events), cudaMemcpyDeviceToHost, stream));
      cudaCheck(cudaMemcpyAsync(host_veloUT_tracks, argen.generate<arg::dev_veloUT_tracks>(argument_offsets), argen.size<arg::dev_veloUT_tracks>(number_of_events*VeloUTTracking::max_num_tracks), cudaMemcpyDeviceToHost, stream));
    }

    cudaEventRecord(cuda_generic_event, stream);
    cudaEventSynchronize(cuda_generic_event);

    ///////////////////////
    // Monte Carlo Check //
    ///////////////////////
    
    if (do_check && i_stream == 0) {
      if (repetition == 0) { // only check efficiencies once

        /* CHECKING Velo TRACKS */
        if (!transmit_device_to_host) { // Fetch data
          cudaCheck(cudaMemcpyAsync(host_number_of_tracks, argen.generate<arg::dev_atomics_storage>(argument_offsets), number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
          cudaCheck(cudaMemcpyAsync(host_accumulated_tracks, argen.generate<arg::dev_atomics_storage>(argument_offsets) + number_of_events, number_of_events * sizeof(int), cudaMemcpyDeviceToHost, stream));
          cudaCheck(cudaMemcpyAsync(host_velo_track_hit_number, argen.generate<arg::dev_velo_track_hit_number>(argument_offsets), argen.size<arg::dev_velo_track_hit_number>(velo_track_hit_number_size), cudaMemcpyDeviceToHost, stream));
          cudaCheck(cudaMemcpyAsync(host_velo_track_hits, argen.generate<arg::dev_velo_track_hits>(argument_offsets), argen.size<arg::dev_velo_track_hits>(host_accumulated_number_of_hits_in_velo_tracks[0]), cudaMemcpyDeviceToHost, stream));
          cudaCheck(cudaMemcpyAsync(host_velo_states, argen.generate<arg::dev_velo_states>(argument_offsets), argen.size<arg::dev_velo_states>(host_number_of_reconstructed_velo_tracks[0]), cudaMemcpyDeviceToHost, stream)); 
          cudaEventRecord(cuda_generic_event, stream);
          cudaEventSynchronize(cuda_generic_event);
        }

        std::cout << "Checking Velo tracks reconstructed on GPU" << std::endl; 
  
        const std::vector<trackChecker::Tracks> tracks_events = prepareTracks(
          host_velo_track_hit_number,
          reinterpret_cast<VeloTracking::Hit<true>*>(host_velo_track_hits),
          host_accumulated_tracks,
          host_number_of_tracks,
          number_of_events);
      
        std::string trackType = "Velo";
        call_pr_checker(
          tracks_events,
          folder_name_MC,
          start_event_offset,
          trackType
        );
      
        /* CHECKING VeloUT TRACKS */
        if ( !transmit_device_to_host ) { // Fetch data
          cudaCheck(cudaMemcpyAsync(host_atomics_veloUT, argen.generate<arg::dev_atomics_veloUT>(argument_offsets), argen.size<arg::dev_atomics_veloUT>(VeloUTTracking::num_atomics*number_of_events), cudaMemcpyDeviceToHost, stream));
          cudaCheck(cudaMemcpyAsync(host_veloUT_tracks, argen.generate<arg::dev_veloUT_tracks>(argument_offsets), argen.size<arg::dev_veloUT_tracks>(number_of_events*VeloUTTracking::max_num_tracks), cudaMemcpyDeviceToHost, stream));
        }
      
        const std::vector< trackChecker::Tracks > veloUT_tracks = prepareVeloUTTracks(
          host_veloUT_tracks,
          host_atomics_veloUT,
          number_of_events
        );  
      
        std::cout << "Checking VeloUT tracks reconstructed on GPU" << std::endl;
        trackType = "VeloUT";
        call_pr_checker (
          veloUT_tracks,
          folder_name_MC,
          start_event_offset,
          trackType
        );
      
        // /* Run VeloUT on x86 architecture */
        // if ( run_on_x86 ) {
        //   std::vector<trackChecker::Tracks> ut_tracks_events;

        //   std::vector<uint> ut_hit_count (2 * number_of_events * VeloUTTracking::n_layers + 1);
        //   std::vector<uint> ut_hits (10 * host_accumulated_number_of_ut_hits[0]);

        //   cudaCheck(cudaMemcpyAsync(ut_hit_count.data(),
        //     argen.generate<arg::dev_ut_hit_count>(argument_offsets),
        //     argen.size<arg::dev_ut_hit_count>(ut_hit_count.size()),
        //     cudaMemcpyDeviceToHost,
        //     stream
        //   ));

        //   cudaCheck(cudaMemcpyAsync(ut_hits.data(),
        //     argen.generate<arg::dev_ut_hits>(argument_offsets),
        //     argen.size<arg::dev_ut_hits>(ut_hit_count.size()),
        //     cudaMemcpyDeviceToHost,
        //     stream
        //   ));
        
        //   int rv = run_veloUT_on_CPU(
        //              ut_tracks_events,
        //              ut_hits,
        //              ut_hit_count,
        //              host_ut_magnet_tool,
        //              host_velo_states,
        //              host_accumulated_tracks,
        //              host_velo_track_hit_number,
        //              host_velo_track_hits,
        //              host_number_of_tracks,
        //              number_of_events
        //             );

        //   if ( rv != 0 )
        //     continue;
          
        //   std::cout << "CHECKING VeloUT TRACKS reconstructed on x86" << std::endl;
        //   trackType = "VeloUT";
        //   call_pr_checker (
        //     ut_tracks_events,
        //     folder_name_MC,
        //     start_event_offset,
        //     trackType);
        // }
      } // only in first repetition
    } // mc_check_enabled
  } // repetitions

  return cudaSuccess;
}
