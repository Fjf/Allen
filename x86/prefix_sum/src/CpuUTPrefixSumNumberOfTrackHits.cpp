#include "CpuUTPrefixSumNumberOfTrackHits.h"

void cpu_ut_prefix_sum_number_of_track_hits_t::operator()(
    const ArgumentRefManager<T>& arguments,
    const RuntimeOptions& runtime_options,
    const Constants& constants,
    HostBuffers& host_buffers,
    cudaStream_t& cuda_stream,
    cudaEvent_t& cuda_generic_event) const {
  // Invokes the function
  function(
    host_buffers.host_prefix_sum_buffer,
    host_buffers.host_allocated_prefix_sum_space,
    offset<dev_ut_track_hit_number_t>(arguments),
    size<dev_ut_track_hit_number_t>(arguments),
    cuda_stream,
    cuda_generic_event,
    host_buffers.host_accumulated_number_of_hits_in_ut_tracks);
}
