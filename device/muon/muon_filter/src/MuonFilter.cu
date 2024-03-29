/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "MuonFilter.cuh"

INSTANTIATE_ALGORITHM(MuonFilter::muon_filter_t)

void MuonFilter::muon_filter_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_event_list_mf_t>(arguments, first<host_number_of_events_t>(arguments));
  set_size<dev_selected_events_mf_t>(arguments, 1);
  set_size<host_selected_events_mf_t>(arguments, 1);
  set_size<dev_mf_decisions_t>(arguments, first<host_number_of_events_t>(arguments));
  set_size<dev_mf_track_atomics_t>(arguments, first<host_number_of_events_t>(arguments));
}

void MuonFilter::muon_filter_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_event_list_mf_t>(arguments, 0, context);
  Allen::memset_async<dev_selected_events_mf_t>(arguments, 0, context);
  Allen::memset_async<dev_mf_decisions_t>(arguments, 0, context);
  Allen::memset_async<dev_mf_track_atomics_t>(arguments, 0, context);

  global_function(muon_filter)(dim3(first<host_number_of_events_t>(arguments)), property<block_dim_t>(), context)(
    arguments);

  Allen::copy<host_selected_events_mf_t, dev_selected_events_mf_t>(arguments, context);
}

__global__ void MuonFilter::muon_filter(MuonFilter::Parameters parameters)
{

  const unsigned number_of_events = gridDim.x;
  const unsigned i_event = blockIdx.x;

  // Long tracks.
  const auto long_tracks = parameters.dev_long_tracks_view->container(i_event);

  Velo::Consolidated::ConstTracks velo_tracks {
    parameters.dev_atomics_velo, parameters.dev_velo_track_hit_number, i_event, number_of_events};

  Velo::Consolidated::ConstStates velo_states {parameters.dev_kalmanvelo_states, velo_tracks.total_number_of_tracks()};

  UT::Consolidated::ConstExtendedTracks ut_tracks {parameters.dev_atomics_ut,
                                                   parameters.dev_ut_track_hit_number,
                                                   parameters.dev_ut_qop,
                                                   parameters.dev_ut_track_velo_indices,
                                                   i_event,
                                                   number_of_events};

  SciFi::Consolidated::ConstTracks scifi_tracks {parameters.dev_atomics_scifi,
                                                 parameters.dev_scifi_track_hit_number,
                                                 parameters.dev_scifi_qop,
                                                 parameters.dev_scifi_states,
                                                 parameters.dev_scifi_track_ut_indices,
                                                 i_event,
                                                 number_of_events};

  const unsigned event_offset = scifi_tracks.tracks_offset(i_event);
  const unsigned number_of_tracks_event = scifi_tracks.number_of_tracks(i_event);
  unsigned* event_mf_decision = parameters.dev_mf_decisions.data() + i_event;

  Associate::Consolidated::ConstTable kalman_pv_ipchi2 {parameters.dev_kalman_pv_ipchi2,
                                                        scifi_tracks.total_number_of_tracks()};
  const auto pvchi2_table = kalman_pv_ipchi2.event_table(scifi_tracks, i_event);

  for (unsigned i_scifi_track = threadIdx.x; i_scifi_track < number_of_tracks_event; i_scifi_track += blockDim.x) {
    bool pTcut = false;
    bool pvcut = false;
    bool isMuon = false;

    // Pt cut.
    auto i_ut_track = scifi_tracks.ut_track(i_scifi_track);
    auto i_velo_track = ut_tracks.velo_track(i_ut_track);
    unsigned i_velo_state = velo_tracks.tracks_offset(i_event) + i_velo_track;
    const float p = 1.f / fabsf(long_tracks.qop(i_scifi_track));
    const float tx2 = velo_states.tx(i_velo_state) * velo_states.tx(i_velo_state);
    const float ty2 = velo_states.ty(i_velo_state) * velo_states.ty(i_velo_state);
    const float pT = p * sqrtf((tx2 + ty2) / (1.f + tx2 + ty2));
    pTcut = pT > parameters.mf_min_pt;

    // Is muon.
    isMuon = parameters.dev_is_muon[i_scifi_track + event_offset];

    // PV cut.
    const float min_ipchi2 = parameters.mf_min_ipchi2;
    pvcut = (pvchi2_table.value(i_scifi_track) > min_ipchi2);

    const bool dec = isMuon && pTcut && pvcut;
    if (dec) {
      atomicOr(event_mf_decision, dec);
    }
  }

  __syncthreads();

  // Passed cut.
  if (threadIdx.x == 0 && event_mf_decision[0]) {
    const auto selected_event = atomicAdd(parameters.dev_selected_events_mf.data(), 1);
    parameters.dev_event_list_mf[selected_event] = i_event;
    parameters.dev_mf_track_atomics[i_event] = ut_tracks.number_of_tracks(i_event);
  }
}
