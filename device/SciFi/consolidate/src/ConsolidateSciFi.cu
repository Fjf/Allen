/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "ConsolidateSciFi.cuh"

INSTANTIATE_ALGORITHM(scifi_consolidate_tracks::scifi_consolidate_tracks_t)

template<bool with_ut, typename T>
__device__ void create_scifi_views_impl(const scifi_consolidate_tracks::Parameters& parameters, const T* tracks)
{
  const unsigned number_of_events = parameters.dev_number_of_events[0];
  const unsigned event_number = blockIdx.x;

  const auto event_tracks_offset = parameters.dev_atomics_scifi[event_number];
  const auto event_number_of_tracks = parameters.dev_atomics_scifi[event_number + 1] - event_tracks_offset;
  const auto event_scifi_track_ut_indices = parameters.dev_scifi_track_ut_indices + event_tracks_offset;
  for (unsigned track_index = threadIdx.x; track_index < event_number_of_tracks; track_index += blockDim.x) {
    const auto input_track_index = event_scifi_track_ut_indices[track_index];
    const auto input_tracks_view = tracks->container(event_number);
    if constexpr (with_ut) {

      const auto* ut_track = &input_tracks_view.track(input_track_index);
      const auto* velo_track = &ut_track->velo_track();
      new (parameters.dev_scifi_track_view + event_tracks_offset + track_index)
        Allen::Views::SciFi::Consolidated::Track {parameters.dev_scifi_hits_view,
                                                  parameters.dev_scifi_qop,
                                                  parameters.dev_atomics_scifi,
                                                  parameters.dev_scifi_track_hit_number,
                                                  track_index,
                                                  event_number};
      new (parameters.dev_long_track_view + event_tracks_offset + track_index)
        Allen::Views::Physics::LongTrack {velo_track,
                                          ut_track,
                                          parameters.dev_scifi_track_view + event_tracks_offset + track_index,
                                          parameters.dev_scifi_qop + event_tracks_offset + track_index};
    }
    else {

      const auto* velo_track = &input_tracks_view.track(input_track_index);

      new (parameters.dev_scifi_track_view + event_tracks_offset + track_index)
        Allen::Views::SciFi::Consolidated::Track {parameters.dev_scifi_hits_view,
                                                  parameters.dev_scifi_qop,
                                                  parameters.dev_atomics_scifi,
                                                  parameters.dev_scifi_track_hit_number,
                                                  track_index,
                                                  event_number};
      new (parameters.dev_long_track_view + event_tracks_offset + track_index)
        Allen::Views::Physics::LongTrack {velo_track,
                                          nullptr,
                                          parameters.dev_scifi_track_view + event_tracks_offset + track_index,
                                          parameters.dev_scifi_qop + event_tracks_offset + track_index};
    }
  }

  if (threadIdx.x == 0) {
    new (parameters.dev_scifi_hits_view + event_number)
      Allen::Views::SciFi::Consolidated::Hits {parameters.dev_scifi_track_hits,
                                               parameters.dev_atomics_scifi,
                                               parameters.dev_scifi_track_hit_number,
                                               event_number,
                                               number_of_events};

    new (parameters.dev_scifi_tracks_view + event_number) Allen::Views::SciFi::Consolidated::Tracks {
      parameters.dev_scifi_track_view, parameters.dev_atomics_scifi, event_number};

    new (parameters.dev_long_tracks_view + event_number)
      Allen::Views::Physics::LongTracks {parameters.dev_long_track_view, parameters.dev_atomics_scifi, event_number};
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    new (parameters.dev_scifi_multi_event_tracks_view)
      Allen::Views::SciFi::Consolidated::MultiEventTracks {parameters.dev_scifi_tracks_view, number_of_events};
    new (parameters.dev_multi_event_long_tracks_view)
      Allen::Views::Physics::MultiEventLongTracks {parameters.dev_long_tracks_view, number_of_events};
    parameters.dev_multi_event_long_tracks_ptr[0] = parameters.dev_multi_event_long_tracks_view.get();
  }
}

__global__ void create_scifi_views(scifi_consolidate_tracks::Parameters parameters)
{
  const auto* ut_tracks =
    Allen::dyn_cast<const Allen::Views::UT::Consolidated::MultiEventTracks*>(*parameters.dev_tracks_view);
  if (ut_tracks) {
    create_scifi_views_impl<true>(parameters, ut_tracks);
  }
  else {
    const auto* velo_tracks =
      static_cast<const Allen::Views::Velo::Consolidated::MultiEventTracks*>(*parameters.dev_tracks_view);
    create_scifi_views_impl<false>(parameters, velo_tracks);
  }
}

void scifi_consolidate_tracks::scifi_consolidate_tracks_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_scifi_track_hits_t>(
    arguments, first<host_accumulated_number_of_hits_in_scifi_tracks_t>(arguments) * sizeof(SciFi::Hit));
  set_size<dev_scifi_qop_t>(arguments, first<host_number_of_reconstructed_scifi_tracks_t>(arguments));
  set_size<dev_scifi_track_ut_indices_t>(arguments, first<host_number_of_reconstructed_scifi_tracks_t>(arguments));
  set_size<dev_scifi_states_t>(arguments, first<host_number_of_reconstructed_scifi_tracks_t>(arguments));
  set_size<dev_scifi_hits_view_t>(arguments, first<host_number_of_events_t>(arguments));
  set_size<dev_scifi_track_view_t>(arguments, first<host_number_of_reconstructed_scifi_tracks_t>(arguments));
  set_size<dev_scifi_tracks_view_t>(arguments, first<host_number_of_events_t>(arguments));
  set_size<dev_scifi_multi_event_tracks_view_t>(arguments, 1);
  set_size<dev_long_track_view_t>(arguments, first<host_number_of_reconstructed_scifi_tracks_t>(arguments));
  set_size<dev_long_tracks_view_t>(arguments, first<host_number_of_events_t>(arguments));
  set_size<dev_multi_event_long_tracks_view_t>(arguments, 1);
  set_size<dev_multi_event_long_tracks_ptr_t>(arguments, 1);
}

void scifi_consolidate_tracks::scifi_consolidate_tracks_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  HostBuffers&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_scifi_multi_event_tracks_view_t>(arguments, 0, context);
  Allen::memset_async<dev_scifi_tracks_view_t>(arguments, 0, context);

  global_function(scifi_consolidate_tracks)(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(
    arguments, constants.dev_looking_forward_constants, constants.dev_magnet_polarity.data());

  global_function(create_scifi_views)(first<host_number_of_events_t>(arguments), 256, context)(arguments);
}

template<typename F>
__device__ void populate(const SciFi::TrackHits& track, const F& assign)
{
  for (int i = 0; i < track.hitsNum; i++) {
    const auto hit_index = track.hits[i];
    assign(i, hit_index);
  }
}

__device__ float qop_calculation(
  LookingForward::Constants const* dev_looking_forward_constants,
  float const magSign,
  float const z0SciFi,
  float const x0SciFi,
  float const y0SciFi,
  float const xVelo,
  float const yVelo,
  float const zVelo,
  float const txO,
  float const tyO,
  float const txSciFi,
  float const tySciFi)
{
  const auto zMatch = (x0SciFi - xVelo + txO * zVelo - txSciFi * z0SciFi) / (txO - txSciFi);
  const auto xMatch = xVelo + txO * (zMatch - zVelo);
  const auto yMatch = yVelo + tyO * (zMatch - zVelo);
  const auto xVelo_at0 = xVelo - txO * zVelo;
  const auto yVelo_at0 = yVelo - tyO * zVelo;
  const auto FLIGHTPATH_MAGNET_SCI_SQ = (x0SciFi - xMatch) * (x0SciFi - xMatch) +
                                        (y0SciFi - yMatch) * (y0SciFi - yMatch) +
                                        (z0SciFi - zMatch) * (z0SciFi - zMatch);
  const auto FLIGHTPATH_VELO_MAGNET_SQ =
    (xVelo_at0 - xMatch) * (xVelo_at0 - xMatch) + (yVelo_at0 - yMatch) * (yVelo_at0 - yMatch) + zMatch * zMatch;
  const auto FLIGHTPATH = 0.001f * sqrtf(FLIGHTPATH_MAGNET_SCI_SQ + FLIGHTPATH_VELO_MAGNET_SQ);
  const auto MAGFIELD = FLIGHTPATH * cosf(asinf(tyO));
  const auto DSLOPE =
    txSciFi / (sqrtf(1.f + txSciFi * txSciFi + tySciFi * tySciFi)) - txO / (sqrtf(1.f + txO * txO + tyO * tyO));

  const auto txO2 = txO * txO;
  const auto txO3 = txO * txO * txO;
  const auto txO4 = txO * txO * txO * txO;
  const auto txO5 = txO * txO * txO * txO * txO;
  const auto txO6 = txO * txO * txO * txO * txO * txO;
  const auto txO7 = txO * txO * txO * txO * txO * txO * txO;
  const auto tyO2 = tyO * tyO;
  const auto tyO4 = tyO * tyO * tyO * tyO;
  const auto tyO5 = tyO * tyO * tyO * tyO * tyO;
  const auto tyO6 = tyO * tyO * tyO * tyO * tyO * tyO;

  const auto C0 = dev_looking_forward_constants->C0[0] + dev_looking_forward_constants->C0[1] * txO2 +
                  dev_looking_forward_constants->C0[2] * txO4 + dev_looking_forward_constants->C0[3] * tyO2 +
                  dev_looking_forward_constants->C0[4] * tyO4 + dev_looking_forward_constants->C0[5] * txO2 * tyO2 +
                  dev_looking_forward_constants->C0[6] * txO6 + dev_looking_forward_constants->C0[7] * tyO5 +
                  dev_looking_forward_constants->C0[8] * txO4 * tyO2 +
                  dev_looking_forward_constants->C0[9] * txO2 * tyO4;
  const auto C1 =
    dev_looking_forward_constants->C1[0] + dev_looking_forward_constants->C1[1] * txO +
    dev_looking_forward_constants->C1[2] * txO3 + dev_looking_forward_constants->C1[3] * txO5 +
    dev_looking_forward_constants->C1[4] * txO7 + dev_looking_forward_constants->C1[5] * tyO2 +
    dev_looking_forward_constants->C1[6] * tyO4 + dev_looking_forward_constants->C1[7] * tyO6 +
    dev_looking_forward_constants->C1[8] * txO * tyO2 + dev_looking_forward_constants->C1[9] * txO * tyO4 +
    dev_looking_forward_constants->C1[10] * txO * tyO6 + dev_looking_forward_constants->C1[11] * txO3 * tyO2 +
    dev_looking_forward_constants->C1[12] * txO3 * tyO4 + dev_looking_forward_constants->C1[13] * txO5 * tyO2;
  const auto C2 = dev_looking_forward_constants->C2[0] + dev_looking_forward_constants->C2[1] * txO2 +
                  dev_looking_forward_constants->C2[2] * txO4 + dev_looking_forward_constants->C2[3] * tyO2 +
                  dev_looking_forward_constants->C2[4] * tyO4 + dev_looking_forward_constants->C2[5] * txO2 * tyO2 +
                  dev_looking_forward_constants->C2[6] * txO6 + dev_looking_forward_constants->C2[7] * tyO5 +
                  dev_looking_forward_constants->C2[8] * txO4 * tyO2 +
                  dev_looking_forward_constants->C2[9] * txO2 * tyO4;
  const auto C3 =
    dev_looking_forward_constants->C3[0] + dev_looking_forward_constants->C3[1] * txO +
    dev_looking_forward_constants->C3[2] * txO3 + dev_looking_forward_constants->C3[3] * txO5 +
    dev_looking_forward_constants->C3[4] * txO7 + dev_looking_forward_constants->C3[5] * tyO2 +
    dev_looking_forward_constants->C3[6] * tyO4 + dev_looking_forward_constants->C3[7] * tyO6 +
    dev_looking_forward_constants->C3[8] * txO * tyO2 + dev_looking_forward_constants->C3[9] * txO * tyO4 +
    dev_looking_forward_constants->C3[10] * txO * tyO6 + dev_looking_forward_constants->C3[11] * txO3 * tyO2 +
    dev_looking_forward_constants->C3[12] * txO3 * tyO4 + dev_looking_forward_constants->C3[13] * txO5 * tyO2;
  const auto C4 = dev_looking_forward_constants->C4[0] + dev_looking_forward_constants->C4[1] * txO2 +
                  dev_looking_forward_constants->C4[2] * txO4 + dev_looking_forward_constants->C4[3] * tyO2 +
                  dev_looking_forward_constants->C4[4] * tyO4 + dev_looking_forward_constants->C4[5] * txO2 * tyO2 +
                  dev_looking_forward_constants->C4[6] * txO6 + dev_looking_forward_constants->C4[7] * tyO5 +
                  dev_looking_forward_constants->C4[8] * txO4 * tyO2 +
                  dev_looking_forward_constants->C4[9] * txO2 * tyO4;

  const auto MAGFIELD_updated =
    MAGFIELD * magSign *
    (C0 + C1 * DSLOPE + C2 * DSLOPE * DSLOPE + C3 * DSLOPE * DSLOPE * DSLOPE + C4 * DSLOPE * DSLOPE * DSLOPE * DSLOPE);
  const auto qop = DSLOPE / MAGFIELD_updated;
  return qop;
}

template<bool with_ut, typename T>
__device__ void scifi_consolidate_tracks_impl(
  const scifi_consolidate_tracks::Parameters& parameters,
  const LookingForward::Constants* dev_looking_forward_constants,
  const float* dev_magnet_polarity,
  const T* tracks)
{

  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const unsigned number_of_events = parameters.dev_number_of_events[0];

  const auto input_tracks_view = tracks->container(event_number);
  const int event_tracks_offset = input_tracks_view.offset();
  // TODO: Don't do this. Will be replaced when SciFi EM is updated.
  const unsigned total_number_of_tracks =
    tracks->container(number_of_events - 1).offset() + tracks->container(number_of_events - 1).size();

  const SciFi::TrackHits* event_scifi_tracks =
    parameters.dev_scifi_tracks + event_tracks_offset * SciFi::Constants::max_SciFi_tracks_per_UT_track;

  const unsigned total_number_of_scifi_hits =
    parameters.dev_scifi_hit_count[number_of_events * SciFi::Constants::n_mat_groups_and_mats];

  SciFi::ConstHits scifi_hits {parameters.dev_scifi_hits, total_number_of_scifi_hits};
  SciFi::ConstHitCount scifi_hit_count {parameters.dev_scifi_hit_count, event_number};

  const auto velo_states_view = parameters.dev_velo_states_view[event_number];

  // Create consolidated SoAs.
  SciFi::Consolidated::Tracks scifi_tracks {parameters.dev_atomics_scifi,
                                            parameters.dev_scifi_track_hit_number,
                                            parameters.dev_scifi_qop,
                                            parameters.dev_scifi_states,
                                            parameters.dev_scifi_track_ut_indices,
                                            event_number,
                                            number_of_events};
  const unsigned number_of_tracks_event = scifi_tracks.number_of_tracks(event_number);
  const unsigned event_offset = scifi_hit_count.event_offset();
  float* tracks_qop = parameters.dev_scifi_qop + parameters.dev_atomics_scifi[event_number];

  // Loop over tracks.
  for (unsigned i = threadIdx.x; i < number_of_tracks_event; i += blockDim.x) {

    // different ways to access velo track depend on the input track
    const auto velo_state = [&]() {
      if constexpr (with_ut) {
        const auto ut_track = input_tracks_view.track(event_scifi_tracks[i].input_track_index);
        const auto velo_track = ut_track.velo_track();
        return velo_track.state(velo_states_view);
      }
      else {
        const auto velo_track = input_tracks_view.track(event_scifi_tracks[i].input_track_index);
        return velo_track.state(velo_states_view);
      }
    }();

    scifi_tracks.ut_track(i) = event_scifi_tracks[i].input_track_index;
    const auto scifi_track_index = event_tracks_offset * SciFi::Constants::max_SciFi_tracks_per_UT_track + i;

    const auto curvature = parameters.dev_scifi_lf_parametrization_consolidate[scifi_track_index];
    const auto tx = parameters.dev_scifi_lf_parametrization_consolidate
                      [total_number_of_tracks * SciFi::Constants::max_SciFi_tracks_per_UT_track + scifi_track_index];
    const auto x0 =
      parameters.dev_scifi_lf_parametrization_consolidate
        [2 * total_number_of_tracks * SciFi::Constants::max_SciFi_tracks_per_UT_track + scifi_track_index];
    const auto d_ratio =
      parameters.dev_scifi_lf_parametrization_consolidate
        [3 * total_number_of_tracks * SciFi::Constants::max_SciFi_tracks_per_UT_track + scifi_track_index];
    const auto y0 =
      parameters.dev_scifi_lf_parametrization_consolidate
        [4 * total_number_of_tracks * SciFi::Constants::max_SciFi_tracks_per_UT_track + scifi_track_index];
    const auto ty =
      parameters.dev_scifi_lf_parametrization_consolidate
        [5 * total_number_of_tracks * SciFi::Constants::max_SciFi_tracks_per_UT_track + scifi_track_index];

    const auto dz = SciFi::Constants::ZEndT - LookingForward::z_mid_t;
    const MiniState scifi_state {x0 + tx * dz + curvature * dz * dz * (1.f + d_ratio * dz),
                                 y0 + ty * SciFi::Constants::ZEndT,
                                 SciFi::Constants::ZEndT,
                                 tx + 2.f * dz * curvature + 3.f * dz * dz * curvature * d_ratio,
                                 ty};

    scifi_tracks.states(i) = scifi_state;

    auto consolidated_hits = scifi_tracks.get_hits(parameters.dev_scifi_track_hits, i);
    const SciFi::TrackHits& track = event_scifi_tracks[i];

    // Update qop of the track
    const auto magSign = dev_magnet_polarity[0];
    const auto z0 = LookingForward::z_mid_t;
    const auto xVelo = velo_state.x();
    const auto yVelo = velo_state.y();
    const auto zVelo = velo_state.z();
    const auto txO = velo_state.tx();
    const auto tyO = velo_state.ty();

    // QoP for scifi tracks
    scifi_tracks.qop(i) =
      qop_calculation(dev_looking_forward_constants, magSign, z0, x0, y0, xVelo, yVelo, zVelo, txO, tyO, tx, ty);

    // QoP for long tracks
    tracks_qop[i] =
      qop_calculation(dev_looking_forward_constants, magSign, z0, x0, y0, xVelo, yVelo, zVelo, txO, tyO, tx, ty);

    // Populate arrays
    populate(track, [&consolidated_hits, &scifi_hits, &event_offset](const unsigned i, const unsigned hit_index) {
      consolidated_hits.x0(i) = scifi_hits.x0(event_offset + hit_index);
    });

    populate(track, [&consolidated_hits, &scifi_hits, &event_offset](const unsigned i, const unsigned hit_index) {
      consolidated_hits.z0(i) = scifi_hits.z0(event_offset + hit_index);
    });

    populate(track, [&consolidated_hits, &scifi_hits, &event_offset](const unsigned i, const unsigned hit_index) {
      consolidated_hits.endPointY(i) = scifi_hits.endPointY(event_offset + hit_index);
    });

    populate(track, [&consolidated_hits, &scifi_hits, &event_offset](const unsigned i, const unsigned hit_index) {
      consolidated_hits.channel(i) = scifi_hits.channel(event_offset + hit_index);
    });

    populate(track, [&consolidated_hits, &scifi_hits, &event_offset](const unsigned i, const unsigned hit_index) {
      consolidated_hits.assembled_datatype(i) = scifi_hits.assembled_datatype(event_offset + hit_index);
    });
  }
}

__global__ void scifi_consolidate_tracks::scifi_consolidate_tracks(
  scifi_consolidate_tracks::Parameters parameters,
  const LookingForward::Constants* dev_looking_forward_constants,
  const float* dev_magnet_polarity)
{
  const auto* ut_tracks =
    Allen::dyn_cast<const Allen::Views::UT::Consolidated::MultiEventTracks*>(*parameters.dev_tracks_view);
  if (ut_tracks) {
    scifi_consolidate_tracks_impl<true>(parameters, dev_looking_forward_constants, dev_magnet_polarity, ut_tracks);
  }
  else {
    const auto* velo_tracks =
      static_cast<const Allen::Views::Velo::Consolidated::MultiEventTracks*>(*parameters.dev_tracks_view);
    scifi_consolidate_tracks_impl<false>(parameters, dev_looking_forward_constants, dev_magnet_polarity, velo_tracks);
  }
}
