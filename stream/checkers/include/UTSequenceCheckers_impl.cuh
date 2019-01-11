#include "VeloUT.cuh"
#include "RunMomentumForwardCPU.h"

/**
 * @brief Specialization for any Velo reconstruction algorithm invoking
 *        consolidate_ut_tracks_t as last step.
 */
template<>
void SequenceVisitor::check<consolidate_ut_tracks_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker)
{
  info_cout << "Checking Velo+UT tracks" << std::endl;

  const auto tracks = prepareUTTracks(
    host_buffers.host_atomics_velo,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_velo_track_hits,
    host_buffers.host_atomics_ut,
    host_buffers.host_ut_track_hit_number,
    host_buffers.host_ut_track_hits,
    host_buffers.host_ut_track_velo_indices,
    host_buffers.host_ut_qop,
    host_buffers.host_number_of_selected_events[0]);

  host_buffers.scifi_ids_ut_tracks = checker_invoker.check<TrackCheckerVeloUT>(start_event_offset, tracks);

  // Run MomentumForward on x86
 
  run_momentum_forward_on_CPU(
    host_buffers.scifi_tracks_events.data(),
    host_buffers.host_atomics_scifi,
    host_buffers.host_scifi_hits.data(),
    host_buffers.host_scifi_hit_count.data(),
    constants.host_scifi_geometry,
    constants.host_inv_clus_res, 
    constants.host_scifi_params,
    host_buffers.host_atomics_velo,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_velo_states,
    host_buffers.host_atomics_ut,
    host_buffers.host_ut_track_hit_number,
    host_buffers.host_ut_qop,
    host_buffers.host_ut_x,
    host_buffers.host_ut_tx,
    host_buffers.host_ut_z,
    host_buffers.host_ut_track_velo_indices,
    host_buffers.scifi_ids_ut_tracks,
    host_buffers.host_number_of_selected_events[0]);
  
}
