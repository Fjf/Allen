#include "RateChecker.h"

std::string const RateChecker::RateTag::name = "RateChecker";

void RateChecker::accumulate(
  const bool* one_track_decisions,
  const bool* two_track_decisions,
  const bool* single_muon_decisions,
  const bool* disp_dimuon_decisions,
  const bool* high_mass_dimuon_decisions,
  const int* track_atomics,
  const uint* sv_atomics,
  const uint selected_events)
{
  // Event loop.
  for (uint i_event = 0; i_event < selected_events; i_event++) {

    // Check one track decisions.
    bool one_track_pass = false;
    bool single_muon_pass = false;
    const int* event_tracks_offsets = track_atomics + selected_events;
    const bool* event_one_track_decisions = one_track_decisions + event_tracks_offsets[i_event];
    const bool* event_single_muon_decisions = single_muon_decisions + event_tracks_offsets[i_event];
    const int n_tracks_event = track_atomics[i_event];
    for (int i_track = 0; i_track < n_tracks_event; i_track++) {
      if (event_one_track_decisions[i_track]) {
        one_track_pass = true;
      }
      if (event_single_muon_decisions[i_track]) {
        single_muon_pass = true;
      }
    }

    // Check two track decisions.
    bool two_track_pass = false;
    bool disp_dimuon_pass = false;
    bool high_mass_dimuon_pass = false;
    const bool* event_two_track_decisions = two_track_decisions + sv_atomics[i_event];
    const bool* event_disp_dimuon_decisions = disp_dimuon_decisions + sv_atomics[i_event];
    const bool* event_high_mass_dimuon_decisions = high_mass_dimuon_decisions + sv_atomics[i_event];
    const int n_svs_event = sv_atomics[i_event + 1] - sv_atomics[i_event];
    for (int i_sv = 0; i_sv < n_svs_event; i_sv++) {
      if (event_two_track_decisions[i_sv]) {
        two_track_pass = true;
      }
      if (event_disp_dimuon_decisions[i_sv]) {
        disp_dimuon_pass = true;
      }
      if (event_high_mass_dimuon_decisions[i_sv]) {
        high_mass_dimuon_pass = true;
      }
    }

    m_evts_one_track += one_track_pass;
    m_evts_two_track += two_track_pass;
    m_evts_single_muon += single_muon_pass;
    m_evts_disp_dimuon += disp_dimuon_pass;
    m_evts_high_mass_dimuon += high_mass_dimuon_pass;
    m_evts_inc += one_track_pass || two_track_pass || single_muon_pass || disp_dimuon_pass || high_mass_dimuon_pass;
  }
}

void RateChecker::report(size_t requested_events) const
{

  // Assume 30 MHz input rate.
  float in_rate = 30000.0;
  printf(
    "One track:        %u / %lu --> %f kHz\n",
    m_evts_one_track,
    requested_events,
    1. * m_evts_one_track / requested_events * in_rate);
  printf(
    "Two track:        %u / %lu --> %f kHz\n",
    m_evts_two_track,
    requested_events,
    1. * m_evts_two_track / requested_events * in_rate);
  printf(
    "Single muon:      %u / %lu --> %f kHz\n",
    m_evts_single_muon,
    requested_events,
    1. * m_evts_single_muon / requested_events * in_rate);
  printf(
    "Displaced dimuon: %u / %lu --> %f kHz\n",
    m_evts_disp_dimuon,
    requested_events,
    1. * m_evts_disp_dimuon / requested_events * in_rate);
  printf(
    "High mass dimuon: %u / %lu --> %f kHz\n",
    m_evts_high_mass_dimuon,
    requested_events,
    1. * m_evts_high_mass_dimuon / requested_events * in_rate);
  printf("------------------------------\n");
  printf(
    "Inclusive:        %u / %lu --> %f kHz\n\n",
    m_evts_inc,
    requested_events,
    1. * m_evts_inc / requested_events * in_rate);
}
