#include "SequenceVisitor.cuh" 
#include "InitEventList.cuh"

template<>
void SequenceVisitor::set_arguments_size<init_event_list_t>(
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers,
  argument_manager_t& arguments)
{
  arguments.set_size<dev_raw_input>(runtime_options.host_velopix_events_size);
  arguments.set_size<dev_raw_input_offsets>(runtime_options.host_velopix_event_offsets_size);
  arguments.set_size<dev_ut_raw_input>(runtime_options.host_ut_events_size);
  arguments.set_size<dev_ut_raw_input_offsets>(runtime_options.host_ut_event_offsets_size); 
  arguments.set_size<dev_scifi_raw_input>(runtime_options.host_scifi_events_size);
  arguments.set_size<dev_scifi_raw_input_offsets>(runtime_options.host_scifi_event_offsets_size); 
  arguments.set_size<dev_event_list>(runtime_options.number_of_events);
} 

template<>
void SequenceVisitor::visit<init_event_list_t>(
  init_event_list_t& state,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  argument_manager_t& arguments,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{ 
  // Setup opts and arguments for kernel call
  state.set_opts(dim3(1), dim3(runtime_options.number_of_events), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_event_list>() );
  
  // Fetch required arguments for the global event cuts algorithm and
  // the various decoding algorithms 
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_raw_input>(), 
    runtime_options.host_velopix_events, 
    arguments.size<dev_raw_input>(), 
    cudaMemcpyHostToDevice, 
    cuda_stream));
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_raw_input_offsets>(), 
    runtime_options.host_velopix_event_offsets, 
    arguments.size<dev_raw_input_offsets>(), 
    cudaMemcpyHostToDevice, cuda_stream)); 
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_ut_raw_input>(),
    runtime_options.host_ut_events,
    runtime_options.host_ut_events_size,
    cudaMemcpyHostToDevice,
    cuda_stream));
  cudaCheck(cudaMemcpyAsync(
    arguments.offset<dev_ut_raw_input_offsets>(),
    runtime_options.host_ut_event_offsets,
    runtime_options.host_ut_event_offsets_size * sizeof(uint32_t),
    cudaMemcpyHostToDevice,
    cuda_stream)); 
  cudaCheck(cudaMemcpyAsync(arguments.offset<dev_scifi_raw_input>(),
    runtime_options.host_scifi_events,
    runtime_options.host_scifi_events_size,
    cudaMemcpyHostToDevice,
    cuda_stream));
  cudaCheck(cudaMemcpyAsync(arguments.offset<dev_scifi_raw_input_offsets>(),
    runtime_options.host_scifi_event_offsets,
    runtime_options.host_scifi_event_offsets_size * sizeof(uint),
    cudaMemcpyHostToDevice,
    cuda_stream)); 

  cudaEventRecord(cuda_generic_event, cuda_stream);
  cudaEventSynchronize(cuda_generic_event);

  state.invoke();
}
