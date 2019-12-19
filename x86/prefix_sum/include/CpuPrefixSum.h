#pragma once

#include "CudaCommon.h"
#include "CpuAlgorithm.cuh"

namespace host_prefix_sum {
  // Arguments
  // struct host_total_sum_holder_t : output_host_datatype<uint> {};
  HOST_OUTPUT(host_total_sum_holder_t, uint)
  DEVICE_INPUT(dev_input_buffer_t, uint)
  DEVICE_OUTPUT(dev_output_buffer_t, uint)

  /**
   * @brief An algorithm that performs the prefix sum on the CPU.
   */
  void host_prefix_sum(
    uint* host_prefix_sum_buffer,
    size_t& host_allocated_prefix_sum_space,
    dev_input_buffer_t dev_input_buffer,
    dev_output_buffer_t dev_output_buffer,
    const size_t dev_input_buffer_size,
    const size_t dev_output_buffer_size,
    cudaStream_t& cuda_stream,
    cudaEvent_t& cuda_generic_event,
    host_total_sum_holder_t host_total_sum_holder);

  template<typename Arguments>
  struct host_prefix_sum_t : public HostAlgorithm {
    constexpr static auto name {"host_prefix_sum_t"};
    decltype(host_function(host_prefix_sum)) function {host_prefix_sum};

    void set_arguments_size(
      ArgumentRefManager<Arguments> arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const HostBuffers& host_buffers) const
    {
      // The total sum holder just holds a single unsigned integer.
      set_size<host_total_sum_holder_t>(arguments, 1);
      set_size<dev_output_buffer_t>(arguments, size<dev_input_buffer_t>(arguments) / sizeof(uint) + 1);
    }

    void operator()(
      const ArgumentRefManager<Arguments>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      cudaStream_t& cuda_stream,
      cudaEvent_t& cuda_generic_event) const
    {
      // Invokes the function
      function(
        host_buffers.host_prefix_sum_buffer,
        host_buffers.host_allocated_prefix_sum_space,
        offset<dev_input_buffer_t>(arguments),
        offset<dev_output_buffer_t>(arguments),
        size<dev_input_buffer_t>(arguments),
        size<dev_output_buffer_t>(arguments),
        cuda_stream,
        cuda_generic_event,
        offset<host_total_sum_holder_t>(arguments));

      arguments.template print<dev_output_buffer_t>();
    }
  };
} // namespace cpu_velo_prefix_sum_number_of_clusters

// TODO: Remove
void cpu_prefix_sum(
    uint* host_prefix_sum_buffer,
    size_t& host_allocated_prefix_sum_space,
    uint* dev_prefix_sum_offset,
    const size_t dev_prefix_sum_size,
    cudaStream_t& cuda_stream,
    cudaEvent_t& cuda_generic_event,
    uint* host_total_sum_holder);