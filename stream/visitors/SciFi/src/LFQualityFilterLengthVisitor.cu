#include "LFQualityFilterLength.cuh"
#include "SequenceVisitor.cuh"

template<>
void SequenceVisitor::set_arguments_size<lf_quality_filter_length_t>(
  lf_quality_filter_length_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_atomics_scifi>(host_buffers.host_number_of_selected_events[0] * LookingForward::num_atomics * 2 + 1);
  arguments.set_size<dev_scifi_tracks>(host_buffers.host_number_of_selected_events[0] * SciFi::Constants::max_tracks);
}

template<>
void SequenceVisitor::visit<lf_quality_filter_length_t>(
  lf_quality_filter_length_t& state,
  const lf_quality_filter_length_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  cudaCheck(cudaMemsetAsync(
    arguments.offset<dev_atomics_scifi>(),
    0,
    arguments.size<dev_atomics_scifi>(),
    cuda_stream));

  cudaCheck(cudaMemsetAsync(
    arguments.offset<dev_scifi_lf_atomics>(),
    0,
    arguments.size<dev_scifi_lf_atomics>(),
    cuda_stream));

  // // Code for running this algorithm last, in the SciFi sequence
  // state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(256), cuda_stream);
  // state.set_arguments(
  //   arguments.offset<dev_scifi_lf_filtered_tracks>(),
  //   arguments.offset<dev_scifi_lf_filtered_atomics>(),
  //   arguments.offset<dev_scifi_tracks>(),
  //   arguments.offset<dev_atomics_scifi>());
  // state.invoke();

  // Code for running the quality filter after this algorithm
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), dim3(256), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_scifi_lf_filtered_tracks>(),
    arguments.offset<dev_scifi_lf_filtered_atomics>(),
    arguments.offset<dev_scifi_lf_tracks>(),
    arguments.offset<dev_scifi_lf_atomics>());
  state.invoke();

  // cudaCheck(cudaMemcpyAsync(
  //   host_buffers.host_atomics_scifi,
  //   arguments.offset<dev_atomics_scifi>(),
  //   arguments.size<dev_atomics_scifi>(),
  //   cudaMemcpyDeviceToHost,
  //   cuda_stream));

  // cudaCheck(cudaMemcpyAsync(
  //   host_buffers.host_scifi_tracks,
  //   arguments.offset<dev_scifi_tracks>(),
  //   arguments.size<dev_scifi_tracks>(),
  //   cudaMemcpyDeviceToHost,
  //   cuda_stream));

  // cudaEventRecord(cuda_generic_event, cuda_stream);
  // cudaEventSynchronize(cuda_generic_event);

  // for (uint i=0; i<host_buffers.host_number_of_selected_events[0]; ++i) {
  //   const auto number_of_tracks = host_buffers.host_atomics_scifi[i];
  //   info_cout << "Event " << i << ", number of tracks " << number_of_tracks << std::endl;

  //   for (int j=0; j<number_of_tracks; ++j) {
  //     const auto track = host_buffers.host_scifi_tracks[i * SciFi::Constants::max_tracks + j];
  //     info_cout << "Track #" << j << ", " << ((int) track.hitsNum) << " hits: ";
  //     for (int k=0; k<track.hitsNum; ++k) {
  //       info_cout << track.hits[k] << ", ";
  //     }
  //     info_cout << " chi2: " << track.get_quality() << std::endl;
  //   }
  //   info_cout << std::endl;
  // }
}