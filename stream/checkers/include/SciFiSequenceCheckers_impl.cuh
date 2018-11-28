#include "PrForward.cuh"
#include "RunForwardCPU.h"

/**
 * @brief Specialization when invoking scifi_pr_forward_t as last step.
 */
template<>
void SequenceVisitor::check<consolidate_scifi_tracks_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker) const
{
  info_cout << "Checking Velo+UT+SciFi tracks" << std::endl;

  const auto tracks = prepareSciFiTracks(
    host_buffers.host_atomics_velo,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_velo_track_hits,
    host_buffers.host_atomics_ut,
    host_buffers.host_ut_track_hit_number,
    host_buffers.host_ut_track_hits,
    host_buffers.host_ut_track_velo_indices,
    host_buffers.host_ut_qop,
    host_buffers.host_atomics_scifi,
    host_buffers.host_scifi_track_hit_number,
    host_buffers.host_scifi_track_hits,
    host_buffers.host_scifi_track_ut_indices,
    host_buffers.host_scifi_qop,
    constants.host_scifi_geometry,
    constants.host_inv_clus_res,
    number_of_events_requested);
  
  checker_invoker.check<TrackCheckerForward>(
    start_event_offset,
    tracks);
}

/**
 * @brief Specialization when invoking cpu_scifi_pr_forward_t as last step.
 */
template<>
void SequenceVisitor::check<cpu_scifi_pr_forward_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker) const
{
  info_cout << "Checking " << cpu_scifi_pr_forward_t::name << " tracks" << std::endl;
  
  const auto tracks = prepareSciFiTracks(
    host_buffers.host_atomics_velo,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_velo_track_hits,
    host_buffers.host_atomics_ut,
    host_buffers.host_ut_track_hit_number,
    host_buffers.host_ut_track_hits,
    host_buffers.host_ut_track_velo_indices,
    host_buffers.host_ut_qop,
    host_buffers.host_atomics_scifi,
    host_buffers.host_scifi_track_hit_number,
    host_buffers.host_scifi_track_hits,
    host_buffers.host_scifi_track_ut_indices,
    host_buffers.host_scifi_qop,
    constants.host_scifi_geometry,
    constants.host_inv_clus_res,
    number_of_events_requested);
  
  checker_invoker.check<TrackCheckerForward>(
    start_event_offset,
    tracks);

}
