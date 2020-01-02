#include "RunHlt1.cuh"
#include "SequenceVisitor.cuh"
#include "RawBanksDefinitions.cuh"

template<>
void SequenceVisitor::set_arguments_size<run_hlt1_t>(
  const run_hlt1_t& state,
  run_hlt1_t::arguments_t arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers)
{
  arguments.set_size<dev_sel_results>(1000 * host_buffers.host_number_of_selected_events[0] * Hlt1::Hlt1Lines::End);
  arguments.set_size<dev_sel_results_atomics>(2 * Hlt1::Hlt1Lines::End + 1);
}

template<>
void SequenceVisitor::visit<run_hlt1_t>(
  run_hlt1_t& state,
  const run_hlt1_t::arguments_t& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t& cuda_generic_event)
{
  cudaCheck(cudaMemsetAsync(
    arguments.offset<dev_sel_results_atomics>(),
    0,
    arguments.size<dev_sel_results_atomics>(),
    cuda_stream));
  cudaCheck(cudaMemsetAsync(
    arguments.offset<dev_sel_results>(),
    false,
    arguments.size<dev_sel_results>(),
    cuda_stream));
  
  state.set_opts(dim3(host_buffers.host_number_of_selected_events[0]), cuda_stream);
  state.set_arguments(
    arguments.offset<dev_kf_tracks>(),
    arguments.offset<dev_consolidated_svs>(),
    arguments.offset<dev_atomics_scifi>(),
    arguments.offset<dev_sv_atomics>(),
    arguments.offset<dev_sel_results>(),
    arguments.offset<dev_sel_results_atomics>());

  state.invoke();

  if (runtime_options.do_check) {
    cudaCheck(cudaMemcpyAsync(
      host_buffers.host_sel_results,
      arguments.offset<dev_sel_results>(),
      arguments.size<dev_sel_results>(),
      cudaMemcpyDeviceToHost,
      cuda_stream));
    cudaCheck(cudaMemcpyAsync(
      host_buffers.host_sel_results_atomics,
      arguments.offset<dev_sel_results_atomics>(),
      arguments.size<dev_sel_results_atomics>(),
      cudaMemcpyDeviceToHost,
      cuda_stream));
  }
}
