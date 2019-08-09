#include "PrimaryVertexChecker.h"

/**
 * @brief Specialization for patPV PV finding algorithm
 */
template<>
void SequenceVisitor::check<pv_fit_seeds_t>(
  HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker,
  const MCEvents& mc_events) const
{
  info_cout << "Checking GPU PVs " << std::endl;
  auto& checker = checker_invoker.checker<GPUPVChecker>("GPU_PVChecker.root");
  checker.accumulate(
    mc_events,
    host_buffers.host_reconstructed_pvs,
    host_buffers.host_number_of_vertex);
}

/**
 * @brief Specialization for beamline PV finding algorithm
 */
template<>
void SequenceVisitor::check<cpu_pv_beamline_t>(
  HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker,
  const MCEvents& mc_events) const
{
  /*
  for(int i_event = 0; i_event < host_buffers.host_number_of_selected_events[0]; i_event++ ) {
  std::cout << "event " << i_event << std::endl;
    for(int i = 0; i < host_buffers.host_number_of_vertex[i_event]; i++ ) {
      PV::Vertex vertex = host_buffers.host_reconstructed_pvs[i*i_event * PV::max_number_vertices + i];
      std::cout << "----" << std::endl;
      std::cout << vertex.position.x << " " << vertex.cov00 << std::endl;
      std::cout << vertex.position.y << " " << vertex.cov11 << std::endl;
      std::cout << vertex.position.z << " " << vertex.cov22 << std::endl;
    }
  }
  */
  info_cout << "Checking CPU beamline PVs " << std::endl;
  auto& checker = checker_invoker.checker<CPUPVChecker>("CPU_PVChecker.root");
  checker.accumulate(
    mc_events,
    host_buffers.host_reconstructed_pvs,
    host_buffers.host_number_of_vertex);
}

/**
 * @brief Specialization for beamline PV finding algorithm on GPU
 */
template<>
void SequenceVisitor::check<pv_beamline_cleanup_t>(
  HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker,
  const MCEvents& mc_events) const
{
  /*
  for(int i_event = 0; i_event < host_buffers.host_number_of_selected_events[0]; i_event++ ) {
    std::cout << "event " << i_event << std::endl;
      for(int i = 0; i < host_buffers.host_number_of_multivertex[i_event]; i++ ) {
        PV::Vertex vertex = host_buffers.host_reconstructed_multi_pvs[i*i_event * PV::max_number_vertices + i];
        std::cout << "----" << std::endl;
        std::cout << vertex.position.x << " " << vertex.cov00 << std::endl;
        std::cout << vertex.position.y << " " << vertex.cov11 << std::endl;
        std::cout << vertex.position.z << " " << vertex.cov22 << std::endl;
      }
    }
    */
  info_cout << "Checking GPU beamline PVs " << std::endl;
  auto& checker = checker_invoker.checker<GPUPVChecker>("GPU_PVChecker.root");
  checker.accumulate(
    mc_events,
    host_buffers.host_reconstructed_multi_pvs,
    host_buffers.host_number_of_multivertex);
}
