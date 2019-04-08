#include "CpuPrefixSum.cuh"

void cpu_prefix_sum(
  uint* host_prefix_sum_buffer,
  uint* dev_prefix_sum_offset,
  const size_t dev_prefix_sum_size,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event,
  uint* host_total_sum_holder)
{
  cudaCheck(cudaMemcpyAsync(
    host_prefix_sum_buffer,
    dev_prefix_sum_offset,
    dev_prefix_sum_size,
    cudaMemcpyDeviceToHost,
    cuda_stream));

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);
  
  // Do prefix sum on CPU
  const size_t number_of_elements = (dev_prefix_sum_size >> 2) - 1;
  uint temp = 0;
  uint temp_sum = 0;
  for (int i=0; i<number_of_elements; ++i) {
    temp_sum += host_prefix_sum_buffer[i];
    host_prefix_sum_buffer[i] = temp;
    temp = temp_sum;
  }
  host_prefix_sum_buffer[number_of_elements] = temp;

  if (host_total_sum_holder != nullptr) {
    host_total_sum_holder[0] = host_prefix_sum_buffer[number_of_elements];
  }

  cudaCheck(cudaMemcpyAsync(
    dev_prefix_sum_offset,
    host_prefix_sum_buffer,
    dev_prefix_sum_size,
    cudaMemcpyHostToDevice,
    cuda_stream));
}