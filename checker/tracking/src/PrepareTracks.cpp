#include "PrepareTracks.h"
#include "ClusteringDefinitions.cuh"
#include "InputTools.h"
#include "MCParticle.h"
#include "SciFiConsolidated.cuh"
#include "SciFiDefinitions.cuh"
#include "TrackChecker.h"
#include "CheckerTypes.h"
#include "UTConsolidated.cuh"
#include "UTDefinitions.cuh"
#include "VeloConsolidated.cuh"
#include "VeloEventModel.cuh"

#include "ROOTHeaders.h"

#include <random>

// LHCb::Track::pseudoRapidity() is based on slopes vector (Gaudi::XYZVector = ROOT::Match::XYZVector)
// slopes = (Tx=dx/dz,Ty=dy/dz,1.)
// eta() for XYZVector:
// https://root.cern.ch/doc/v608/namespaceROOT_1_1Math_1_1Impl.html#a7d4efefe2855d886fdbae73c81adc574 z = 1.f -> can
// simplify eta_from_rho_z
float eta_from_rho(const float rho)
{
  const float z = 1.f;
  if (rho > 0.f) {

    // value to control Taylor expansion of sqrt
    static const float big_z_scaled = std::pow(std::numeric_limits<float>::epsilon(), static_cast<float>(-.25));

    float z_scaled = z / rho;
    if (std::fabs(z_scaled) < big_z_scaled) {
      return std::log(z_scaled + std::sqrt(z_scaled * z_scaled + 1.f));
    }
    else {
      // apply correction using first order Taylor expansion of sqrt
      return z > 0.f ? std::log(2.f * z_scaled + 0.5f / z_scaled) : -std::log(-2.f * z_scaled);
    }
  }
  // case vector has rho = 0
  return z + 22756.f;
}

std::vector<Checker::Tracks> prepareVeloTracks(
  const uint* track_atomics,
  const uint* track_hit_number,
  const char* track_hits,
  const uint number_of_events)
{
  /* Tracks to be checked, save in format for checker */
  std::vector<Checker::Tracks> checker_tracks; // all tracks from all events
  for (uint i_event = 0; i_event < number_of_events; i_event++) {
    Checker::Tracks tracks; // all tracks within one event

    const Velo::Consolidated::Tracks velo_tracks {
      (uint*) track_atomics, (uint*) track_hit_number, i_event, number_of_events};
    const uint number_of_tracks_event = velo_tracks.number_of_tracks(i_event);

    for (uint i_track = 0; i_track < number_of_tracks_event; i_track++) {
      Checker::Track t;
      t.p = 0.f;

      const uint velo_track_number_of_hits = velo_tracks.number_of_hits(i_track);
      Velo::Consolidated::Hits velo_track_hits = velo_tracks.get_hits((char*) track_hits, i_track);

      for (int i_hit = 0; i_hit < velo_track_number_of_hits; ++i_hit) {
        t.addId(velo_track_hits.LHCbID[i_hit]);
      }
      tracks.push_back(t);
    } // tracks
    checker_tracks.emplace_back(tracks);
  }

  return checker_tracks;
}

std::vector<Checker::Tracks> prepareUTTracks(
  const uint* velo_track_atomics,
  const uint* velo_track_hit_number,
  const char* velo_track_hits,
  const char* kalman_velo_states,
  const int* ut_track_atomics,
  const uint* ut_track_hit_number,
  const char* ut_track_hits,
  const uint* ut_track_velo_indices,
  const float* ut_qop,
  const uint number_of_events)
{
  std::vector<Checker::Tracks> checker_tracks; // all tracks from all events
  for (uint i_event = 0; i_event < number_of_events; i_event++) {
    Checker::Tracks tracks; // all tracks within one event

    const Velo::Consolidated::Tracks velo_tracks {
      (uint*) velo_track_atomics, (uint*) velo_track_hit_number, i_event, number_of_events};
    const Velo::Consolidated::States velo_states {(char*) kalman_velo_states, velo_tracks.total_number_of_tracks};
    const uint velo_event_tracks_offset = velo_tracks.tracks_offset(i_event);
    const UT::Consolidated::Tracks ut_tracks {(uint*) ut_track_atomics,
                                              (uint*) ut_track_hit_number,
                                              (float*) ut_qop,
                                              (uint*) ut_track_velo_indices,
                                              i_event,
                                              number_of_events};
    const uint number_of_tracks_event = ut_tracks.number_of_tracks(i_event);

    for (uint i_track = 0; i_track < number_of_tracks_event; i_track++) {
      const int velo_track_index = ut_tracks.velo_track[i_track];
      const uint velo_state_index = velo_event_tracks_offset + velo_track_index;
      const VeloState velo_state = velo_states.get(velo_state_index);

      Checker::Track t;

      // momentum
      const float qop = ut_tracks.qop[i_track];
      t.p = 1.f / std::abs(qop);
      t.qop = qop;
      // direction at first state -> velo state of track
      const float tx = velo_state.tx;
      const float ty = velo_state.ty;
      const float slope2 = tx * tx + ty * ty;
      t.pt = std::sqrt(slope2 / (1.f + slope2)) / std::fabs(qop);
      // pseudorapidity
      const float rho = std::sqrt(slope2);
      t.eta = eta_from_rho(rho);

      // hits in UT
      const uint ut_track_number_of_hits = ut_tracks.number_of_hits(i_track);
      const UT::Consolidated::Hits track_hits_ut = ut_tracks.get_hits((char*) ut_track_hits, i_track);
      for (int i_hit = 0; i_hit < ut_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_ut.LHCbID[i_hit]);
      }
      // get index to corresponding velo track

      const uint velo_track_number_of_hits = velo_tracks.number_of_hits(velo_track_index);
      const Velo::Consolidated::Hits track_hits_velo = velo_tracks.get_hits((char*) velo_track_hits, velo_track_index);
      // hits in Velo
      for (int i_hit = 0; i_hit < velo_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_velo.LHCbID[i_hit]);
      }
      tracks.push_back(t);
    } // tracks
    checker_tracks.emplace_back(tracks);
  }

  return checker_tracks;
}

std::vector<Checker::Tracks> prepareSciFiTracks(
  const uint* velo_track_atomics,
  const uint* velo_track_hit_number,
  const char* velo_track_hits,
  const char* kalman_velo_states,
  const int* ut_track_atomics,
  const uint* ut_track_hit_number,
  const char* ut_track_hits,
  const uint* ut_track_velo_indices,
  const float* ut_qop,
  const int* scifi_track_atomics,
  const uint* scifi_track_hit_number,
  const char* scifi_track_hits,
  const uint* scifi_track_ut_indices,
  const float* scifi_qop,
  const MiniState* scifi_states,
  const char* scifi_geometry,
  const std::array<float, 9>& inv_clus_res,
  const float* muon_catboost_output,
  const bool* is_muon,
  const uint number_of_events)
{
  const SciFi::SciFiGeometry scifi_geom(scifi_geometry);
  std::vector<Checker::Tracks> checker_tracks; // all tracks from all events
  int n_is_muon = 0;
  int n_total_tracks = 0;
  for (uint i_event = 0; i_event < number_of_events; i_event++) {
    Checker::Tracks tracks; // all tracks within one event

    const Velo::Consolidated::Tracks velo_tracks {
      (uint*) velo_track_atomics, (uint*) velo_track_hit_number, i_event, number_of_events};
    const Velo::Consolidated::States velo_states {(char*) kalman_velo_states, velo_tracks.total_number_of_tracks};
    const uint velo_event_tracks_offset = velo_tracks.tracks_offset(i_event);
    const UT::Consolidated::Tracks ut_tracks {(uint*) ut_track_atomics,
                                              (uint*) ut_track_hit_number,
                                              (float*) ut_qop,
                                              (uint*) ut_track_velo_indices,
                                              i_event,
                                              number_of_events};

    SciFi::Consolidated::Tracks scifi_tracks {(uint*) scifi_track_atomics,
                                              (uint*) scifi_track_hit_number,
                                              (float*) scifi_qop,
                                              (MiniState*) scifi_states,
                                              (uint*) scifi_track_ut_indices,
                                              i_event,
                                              number_of_events};
    const uint number_of_tracks_event = scifi_tracks.number_of_tracks(i_event);
    const uint event_offset = scifi_tracks.tracks_offset(i_event);

    for (uint i_track = 0; i_track < number_of_tracks_event; i_track++) {
      const uint UT_track_index = scifi_tracks.ut_track[i_track];
      const int velo_track_index = ut_tracks.velo_track[UT_track_index];
      const uint velo_state_index = velo_event_tracks_offset + velo_track_index;
      const VeloState velo_state = velo_states.get(velo_state_index);

      Checker::Track t;

      // momentum
      const float qop = scifi_tracks.qop[i_track];
      t.p = 1.f / std::abs(qop);
      t.qop = qop;
      // direction at first state -> velo state of track
      const float tx = velo_state.tx;
      const float ty = velo_state.ty;
      const float slope2 = tx * tx + ty * ty;
      t.pt = std::sqrt(slope2 / (1.f + slope2)) / std::fabs(qop);
      // pseudorapidity
      const float rho = std::sqrt(slope2);
      t.eta = eta_from_rho(rho);

      // add SciFi hits
      const uint scifi_track_number_of_hits = scifi_tracks.number_of_hits(i_track);
      SciFi::Consolidated::Hits track_hits_scifi =
        scifi_tracks.get_hits((char*) scifi_track_hits, i_track, &scifi_geom, inv_clus_res.data());
      for (int i_hit = 0; i_hit < scifi_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_scifi.LHCbID(i_hit));
      }

      // add UT hits
      const uint ut_track_number_of_hits = ut_tracks.number_of_hits(UT_track_index);
      const UT::Consolidated::Hits track_hits_ut = ut_tracks.get_hits((char*) ut_track_hits, UT_track_index);
      for (int i_hit = 0; i_hit < ut_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_ut.LHCbID[i_hit]);
      }

      // add Velo hits
      const uint velo_track_number_of_hits = velo_tracks.number_of_hits(velo_track_index);
      const Velo::Consolidated::Hits track_hits_velo = velo_tracks.get_hits((char*) velo_track_hits, velo_track_index);
      for (int i_hit = 0; i_hit < velo_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_velo.LHCbID[i_hit]);
      }

      // add muon information
      t.muon_catboost_output = muon_catboost_output[event_offset + i_track];
      t.is_muon = is_muon[event_offset + i_track];

      if (t.is_muon) n_is_muon++;

      n_total_tracks++;

      tracks.push_back(t);
    } // tracks
    checker_tracks.emplace_back(tracks);
  }

  debug_cout << "Number of tracks with is_muon true = " << n_is_muon << " / " << n_total_tracks << std::endl;

  return checker_tracks;
}

std::vector<Checker::Tracks> prepareSciFiTracks(
  const uint* velo_track_atomics,
  const uint* velo_track_hit_number,
  const char* velo_track_hits,
  const char* kalman_velo_states,
  const int* ut_track_atomics,
  const uint* ut_track_hit_number,
  const char* ut_track_hits,
  const uint* ut_track_velo_indices,
  const float* ut_qop,
  const std::vector<std::vector<SciFi::TrackHits>>& scifi_tracks,
  const SciFi::Hits& scifi_hits,
  const uint* host_scifi_hit_count,
  const uint number_of_events)
{
  std::vector<Checker::Tracks> checker_tracks; // all tracks from all events
  for (uint i_event = 0; i_event < number_of_events; i_event++) {
    Checker::Tracks tracks; // all tracks within one event

    const Velo::Consolidated::Tracks velo_tracks {
      (uint*) velo_track_atomics, (uint*) velo_track_hit_number, i_event, number_of_events};
    const Velo::Consolidated::States velo_states {(char*) kalman_velo_states, velo_tracks.total_number_of_tracks};
    const uint velo_event_tracks_offset = velo_tracks.tracks_offset(i_event);

    const UT::Consolidated::Tracks ut_tracks {(uint*) ut_track_atomics,
                                              (uint*) ut_track_hit_number,
                                              (float*) ut_qop,
                                              (uint*) ut_track_velo_indices,
                                              i_event,
                                              number_of_events};

    const uint total_number_of_hits = host_scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats];
    const SciFi::HitCount scifi_hit_count {(uint32_t*) host_scifi_hit_count, i_event};

    const auto& scifi_tracks_event = scifi_tracks[i_event];
    for (uint i_track = 0; i_track < scifi_tracks_event.size(); i_track++) {
      const auto& scifi_track = scifi_tracks_event[i_track];

      const uint UT_track_index = scifi_track.ut_track_index;
      const int velo_track_index = ut_tracks.velo_track[UT_track_index];
      const uint velo_state_index = velo_event_tracks_offset + velo_track_index;
      const VeloState velo_state = velo_states.get(velo_state_index);

      Checker::Track t;

      // momentum
      const float qop = scifi_track.qop;
      t.p = 1.f / std::abs(qop);
      t.qop = qop;

      // direction at first state -> velo state of track
      const float tx = velo_state.tx;
      const float ty = velo_state.ty;
      const float slope2 = tx * tx + ty * ty;
      t.pt = std::sqrt(slope2 / (1.f + slope2)) / std::fabs(qop);
      // pseudorapidity
      const float rho = std::sqrt(slope2);
      t.eta = eta_from_rho(rho);

      // add SciFi hits
      for (int i_hit = 0; i_hit < scifi_track.hitsNum; ++i_hit) {
        t.addId(scifi_hits.LHCbID(scifi_hit_count.event_offset() + scifi_track.hits[i_hit]));
      }

      // add UT hits
      const uint ut_track_number_of_hits = ut_tracks.number_of_hits(UT_track_index);
      const UT::Consolidated::Hits track_hits_ut = ut_tracks.get_hits((char*) ut_track_hits, UT_track_index);
      for (int i_hit = 0; i_hit < ut_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_ut.LHCbID[i_hit]);
      }

      // add Velo hits
      const uint velo_track_number_of_hits = velo_tracks.number_of_hits(velo_track_index);
      const Velo::Consolidated::Hits track_hits_velo = velo_tracks.get_hits((char*) velo_track_hits, velo_track_index);
      for (int i_hit = 0; i_hit < velo_track_number_of_hits; ++i_hit) {
        t.addId(track_hits_velo.LHCbID[i_hit]);
      }

      tracks.push_back(t);
    } // tracks
    checker_tracks.emplace_back(tracks);
  }

  return checker_tracks;
}
