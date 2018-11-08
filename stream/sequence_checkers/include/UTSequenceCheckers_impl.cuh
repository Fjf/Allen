#include "VeloUT.cuh"
#include "CompassUT.cuh"

/**
 * @brief Specialization for any Velo reconstruction algorithm invoking
 *        veloUT_t as last step.
 */
template<>
void SequenceVisitor::check<veloUT_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const PrCheckerInvoker& pr_checker_invoker) const
{
  info_cout << "Checking " << veloUT_t::name << " tracks" << std::endl;

  const auto tracks = prepareTracks<TrackCheckerVeloUT>(
    host_buffers.host_velo_tracks_atomics,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_velo_track_hits,
    host_buffers.host_veloUT_tracks,
    host_buffers.host_atomics_veloUT,
    number_of_events_requested);

  pr_checker_invoker.check<TrackCheckerVeloUT>(
    start_event_offset,
    tracks);
}

/**
 * @brief Specialization for any Velo reconstruction algorithm invoking
 *        compass_ut_t as last step.
 */
template<>
void SequenceVisitor::check<compass_ut_t>(
  const uint& start_event_offset,
  const uint& number_of_events_requested,
  const HostBuffers& host_buffers,
  const PrCheckerInvoker& pr_checker_invoker) const
{
  info_cout << "Checking " << compass_ut_t::name << " tracks" << std::endl;

  const auto tracks = prepareTracks<TrackCheckerVeloUT>(
    host_buffers.host_velo_tracks_atomics,
    host_buffers.host_velo_track_hit_number,
    host_buffers.host_velo_track_hits,
    host_buffers.host_compassUT_tracks,
    host_buffers.host_atomics_compassUT,
    number_of_events_requested);

  pr_checker_invoker.check<TrackCheckerVeloUT>(
    start_event_offset,
    tracks);
}
