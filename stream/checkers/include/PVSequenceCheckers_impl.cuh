#include "PrimaryVertexChecker.h"

/**
 * @brief Specialization for any Velo reconstruction algorithm invoking
 *        consolidate_tracks_t as last step.
 */
template<>
void SequenceVisitor::check<pv_fit_seeds_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker)
{
  info_cout << "Checking GPU PVs " << checker_invoker.mc_pv_folder << std::endl;
  checkPVs( 
    checker_invoker.mc_pv_folder,  
    number_of_events_requested, 
    host_buffers.host_reconstructed_pvs, 
    host_buffers.host_number_of_vertex);
  
}
