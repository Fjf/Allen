#include "ParKalmanDefinitions.cuh"
#include "ParKalmanFilter.cuh"
#include "ParKalmanVeloOnly.cuh"
#include "KalmanChecker.h"
#include "PrepareKalmanTracks.h"

template<>
void SequenceVisitor::check<kalman_filter_t>(
  HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker,
  const MCEvents& mc_events) const
{
// Note: Nothing happens if not compiled with ROOT
#ifdef WITH_ROOT
  info_cout << "Producing Kalman plots" << std::endl << std::endl;

  const auto tracks = prepareKalmanTracks(
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
    host_buffers.host_scifi_states,
    constants.host_scifi_geometry.data(),
    constants.host_inv_clus_res,
    host_buffers.host_kf_tracks,
    host_buffers.host_kalmanvelo_states,
    host_buffers.host_reconstructed_multi_pvs,
    host_buffers.host_number_of_multivertex,
    host_buffers.host_number_of_selected_events[0]);

  auto& checker = checker_invoker.checker<KalmanChecker>("KalmanIPCheckerOutput.root");
  checker.accumulate(mc_events, tracks);
#endif
}

template<>
void SequenceVisitor::check<kalman_velo_only_t>(
  HostBuffers& host_buffers,
  const Constants& constants,
  const CheckerInvoker& checker_invoker,
  const MCEvents& mc_events) const
{
// Note: Nothing happens if not compiled with ROOT
#ifdef WITH_ROOT
  info_cout << "Producing Kalman plots" << std::endl << std::endl;

  const auto tracks = prepareKalmanTracks(
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
    host_buffers.host_scifi_states,
    constants.host_scifi_geometry.data(),
    constants.host_inv_clus_res,
    host_buffers.host_kf_tracks,
    host_buffers.host_kalmanvelo_states,
    host_buffers.host_reconstructed_multi_pvs,
    host_buffers.host_number_of_multivertex,
    host_buffers.host_number_of_selected_events[0]);

  auto& checker = checker_invoker.checker<KalmanChecker>("KalmanIPCheckerOutput.root");
  checker.accumulate(mc_events, tracks);
#endif
}
