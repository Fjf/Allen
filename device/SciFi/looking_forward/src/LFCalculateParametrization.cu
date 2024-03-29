/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "LFCreateTracks.cuh"

template<bool with_ut, typename T>
__device__ void calculate_parametrization(
  lf_create_tracks::Parameters parameters,
  const LookingForward::Constants* dev_looking_forward_constants,
  const T* tracks)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const unsigned number_of_events = parameters.dev_number_of_events[0];

  const auto max_triplets_per_input_track = parameters.max_triplets_per_input_track;
  const auto input_tracks_view = tracks->container(event_number);

  const int event_tracks_offset = input_tracks_view.offset();
  // TODO: Don't do this. Will be replaced when SciFi EM is updated.
  const unsigned total_number_of_tracks =
    tracks->container(number_of_events - 1).offset() + tracks->container(number_of_events - 1).size();

  const auto velo_states_view = parameters.dev_velo_states_view[event_number];

  // SciFi hits
  const unsigned total_number_of_hits =
    parameters.dev_scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats];

  SciFi::ConstHitCount scifi_hit_count {parameters.dev_scifi_hit_count, event_number};
  SciFi::ConstHits scifi_hits {parameters.dev_scifi_hits, total_number_of_hits};

  const auto event_offset = scifi_hit_count.event_offset();
  const auto number_of_tracks = parameters.dev_scifi_lf_atomics[event_number];

  for (unsigned i = threadIdx.x; i < number_of_tracks; i += blockDim.x) {
    const auto scifi_track_index = event_tracks_offset * max_triplets_per_input_track + i;
    const SciFi::TrackHits& track = parameters.dev_scifi_lf_tracks[scifi_track_index];
    const auto input_track_number = track.input_track_index;
    const auto input_track = input_tracks_view.track(input_track_number);

    // different ways to access velo track depend on the input track
    const auto velo_state = [&input_track, velo_states_view]() {
      if constexpr (with_ut) {
        const auto velo_track = input_track.velo_track();
        return velo_track.state(velo_states_view);
      }
      else {
        return input_track.state(velo_states_view);
      }
    }();

    // Note: The notation 1, 2, 3 is used here (instead of h0, h1, h2)
    //       to avoid mistakes, as the code is similar to that of Hybrid Seeding
    //       noref here means the raw z position of the hits.
    //       The parameterization with the dratio values is valid for zHit - LookingForward::z_mid_t.
    //       If LookingForward::z_mid_t will change , the tuning has to be redone.
    //       The dRatio correction follows a second correctlion level as the seeding does when y-z plane motion becomes
    //       known, including a parameterization as a function of the track expected position at X(z = zT2 station), Y(z
    //       = zT2 station). Potentially something better can be done here parameterizing dRatio vs (tx,ty, txSciFi for
    //       example) See : https://cds.cern.ch/record/2296404/files/CERN-THESIS-2017-254.pdf  (page 129)
    const auto h1 = event_offset + track.hits[0];
    const auto h2 = event_offset + track.hits[1];
    const auto h3 = event_offset + track.hits[2];
    const auto x1 = scifi_hits.x0(h1);
    const auto x2 = scifi_hits.x0(h2);
    const auto x3 = scifi_hits.x0(h3);
    const auto z1_noref = dev_looking_forward_constants->Zone_zPos_xlayers[track.get_layer(0)];
    const auto z2_noref = dev_looking_forward_constants->Zone_zPos_xlayers[track.get_layer(1)];
    const auto z3_noref = dev_looking_forward_constants->Zone_zPos_xlayers[track.get_layer(2)];

    // Updated d_ratio
    const auto track_y_ref = velo_state.y + velo_state.ty * (z2_noref - velo_state.z);
    const auto radius_position = sqrtf((5.f * 5.f * 1.e-8f * x2 * x2 + 1e-6f * track_y_ref * track_y_ref));
    const auto d_ratio = -1.f * (LookingForward::d_ratio_par_0 + LookingForward::d_ratio_par_1 * radius_position +
                                 LookingForward::d_ratio_par_2 * radius_position * radius_position);

    const auto z1 = z1_noref - LookingForward::z_mid_t;
    const auto z2 = z2_noref - LookingForward::z_mid_t;
    const auto z3 = z3_noref - LookingForward::z_mid_t;
    const auto corrZ1 = 1.f + d_ratio * z1;
    const auto corrZ2 = 1.f + d_ratio * z2;
    const auto corrZ3 = 1.f + d_ratio * z3;

    const auto det = z1 * z1 * corrZ1 * z2 + z1 * z3 * z3 * corrZ3 + z2 * z2 * corrZ2 * z3 - z2 * z3 * z3 * corrZ3 -
                     z1 * z2 * z2 * corrZ2 - z3 * z1 * z1 * corrZ1;
    const auto det1 = x1 * z2 + z1 * x3 + x2 * z3 - z2 * x3 - z1 * x2 - z3 * x1;
    const auto det2 = z1 * z1 * corrZ1 * x2 + x1 * z3 * z3 * corrZ3 + z2 * z2 * corrZ2 * x3 - x2 * z3 * z3 * corrZ3 -
                      x1 * z2 * z2 * corrZ2 - x3 * z1 * z1 * corrZ1;
    const auto det3 = z1 * z1 * corrZ1 * z2 * x3 + z1 * z3 * z3 * corrZ3 * x2 + z2 * z2 * corrZ2 * z3 * x1 -
                      z2 * z3 * z3 * corrZ3 * x1 - z1 * z2 * z2 * corrZ2 * x3 - z3 * z1 * z1 * corrZ1 * x2;
    // Differently  from  the Seeding, if the determinant is too small, we don't assume is a line, but we still compute
    // values Not sure if numerically we might end up in a 1./0.f case.  How to protect from this?
    const auto recdet = 1.f / det;
    const auto a1 = recdet * det1;
    const auto b1 = recdet * det2;
    const auto c1 = recdet * det3;

    parameters.dev_scifi_lf_parametrization[scifi_track_index] = a1;
    parameters.dev_scifi_lf_parametrization[total_number_of_tracks * max_triplets_per_input_track + scifi_track_index] =
      b1;
    parameters
      .dev_scifi_lf_parametrization[2 * total_number_of_tracks * max_triplets_per_input_track + scifi_track_index] = c1;
    parameters
      .dev_scifi_lf_parametrization[3 * total_number_of_tracks * max_triplets_per_input_track + scifi_track_index] =
      d_ratio;
  }
}

__global__ void lf_create_tracks::lf_calculate_parametrization(
  lf_create_tracks::Parameters parameters,
  const LookingForward::Constants* dev_looking_forward_constants)
{
  const auto* ut_tracks =
    Allen::dyn_cast<const Allen::Views::UT::Consolidated::MultiEventTracks*>(*parameters.dev_tracks_view);
  if (ut_tracks) {
    calculate_parametrization<true>(parameters, dev_looking_forward_constants, ut_tracks);
  }
  else {
    const auto* velo_tracks =
      static_cast<const Allen::Views::Velo::Consolidated::MultiEventTracks*>(*parameters.dev_tracks_view);
    calculate_parametrization<false>(parameters, dev_looking_forward_constants, velo_tracks);
  }
}
