/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "VeloKalmanFilter.cuh"
#include "ROOTService.h"

INSTANTIATE_ALGORITHM(velo_kalman_filter::velo_kalman_filter_t)

void velo_kalman_filter::velo_kalman_filter_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_is_backward_t>(arguments, first<host_number_of_reconstructed_velo_tracks_t>(arguments));
  set_size<dev_velo_kalman_beamline_states_t>(
    arguments, first<host_number_of_reconstructed_velo_tracks_t>(arguments) * Velo::Consolidated::States::size);
  set_size<dev_velo_kalman_endvelo_states_t>(
    arguments, first<host_number_of_reconstructed_velo_tracks_t>(arguments) * Velo::Consolidated::States::size);
  set_size<dev_velo_kalman_beamline_states_view_t>(arguments, first<host_number_of_events_t>(arguments));
  set_size<dev_velo_kalman_endvelo_states_view_t>(arguments, first<host_number_of_events_t>(arguments));
}

void velo_kalman_filter::velo_kalman_filter_t::init()
{
#ifndef ALLEN_STANDALONE
  histogram_velo_total_track_eta =
    new gaudi_monitoring::Lockable_Histogram<> {{this,
                                                 "velo_total_track_eta",
                                                 "#total_eta",
                                                 {property<histogram_velo_track_eta_nbins_t>(),
                                                  property<histogram_velo_track_eta_min_t>(),
                                                  property<histogram_velo_track_eta_max_t>()}},
                                                {}};
  histogram_velo_total_track_phi =
    new gaudi_monitoring::Lockable_Histogram<> {{this,
                                                 "velo_total_track_phi",
                                                 "#total_phi",
                                                 {property<histogram_velo_track_phi_nbins_t>(),
                                                  property<histogram_velo_track_phi_min_t>(),
                                                  property<histogram_velo_track_phi_max_t>()}},
                                                {}};
  histogram_velo_total_track_nhits =
    new gaudi_monitoring::Lockable_Histogram<> {{this,
                                                 "velo_total_track_nhits",
                                                 "total N. hits / track",
                                                 {property<histogram_velo_track_nhits_nbins_t>(),
                                                  property<histogram_velo_track_nhits_min_t>(),
                                                  property<histogram_velo_track_nhits_max_t>()}},
                                                {}};
  histogram_velo_forward_track_eta =
    new gaudi_monitoring::Lockable_Histogram<> {{this,
                                                 "velo_forward_track_eta",
                                                 "#forward_eta",
                                                 {property<histogram_velo_track_eta_nbins_t>(),
                                                  property<histogram_velo_track_eta_min_t>(),
                                                  property<histogram_velo_track_eta_max_t>()}},
                                                {}};
  histogram_velo_forward_track_phi =
    new gaudi_monitoring::Lockable_Histogram<> {{this,
                                                 "velo_forward_track_phi",
                                                 "#forward_phi",
                                                 {property<histogram_velo_track_phi_nbins_t>(),
                                                  property<histogram_velo_track_phi_min_t>(),
                                                  property<histogram_velo_track_phi_max_t>()}},
                                                {}};
  histogram_velo_forward_track_nhits =
    new gaudi_monitoring::Lockable_Histogram<> {{this,
                                                 "velo_forward_track_nhits",
                                                 "forward N. hits / track",
                                                 {property<histogram_velo_track_nhits_nbins_t>(),
                                                  property<histogram_velo_track_nhits_min_t>(),
                                                  property<histogram_velo_track_nhits_max_t>()}},
                                                {}};
  histogram_velo_backward_track_eta =
    new gaudi_monitoring::Lockable_Histogram<> {{this,
                                                 "velo_backward_track_eta",
                                                 "#backward_eta",
                                                 {property<histogram_velo_track_eta_nbins_t>(),
                                                  property<histogram_velo_track_eta_min_t>(),
                                                  property<histogram_velo_track_eta_max_t>()}},
                                                {}};
  histogram_velo_backward_track_phi =
    new gaudi_monitoring::Lockable_Histogram<> {{this,
                                                 "velo_backward_track_phi",
                                                 "#backward_phi",
                                                 {property<histogram_velo_track_phi_nbins_t>(),
                                                  property<histogram_velo_track_phi_min_t>(),
                                                  property<histogram_velo_track_phi_max_t>()}},
                                                {}};
  histogram_velo_backward_track_nhits =
    new gaudi_monitoring::Lockable_Histogram<> {{this,
                                                 "velo_backward_track_nhits",
                                                 "backward N. hits / track",
                                                 {property<histogram_velo_track_nhits_nbins_t>(),
                                                  property<histogram_velo_track_nhits_min_t>(),
                                                  property<histogram_velo_track_nhits_max_t>()}},
                                                {}};
#endif
}

void velo_kalman_filter::velo_kalman_filter_t::output_monitor(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Allen::Context& context) const
{
  auto handler = runtime_options.root_service->handle(name());
  auto tree = handler.tree("monitor_tree");
  if (tree == nullptr) return;

  float beamPOCA_x, beamPOCA_y, beamPOCA_z, beamPOCA_tx, beamPOCA_ty;
  bool backward;
  float eta;

  handler.branch(tree, "beamPOCA_tx", beamPOCA_tx);
  handler.branch(tree, "beamPOCA_ty", beamPOCA_ty);
  handler.branch(tree, "beamPOCA_x", beamPOCA_x);
  handler.branch(tree, "beamPOCA_y", beamPOCA_y);
  handler.branch(tree, "beamPOCA_z", beamPOCA_z);
  handler.branch(tree, "backward", backward);
  handler.branch(tree, "eta", eta);

  const auto host_event_list = make_host_buffer<dev_event_list_t>(arguments, context);
  const auto host_kalman_beamline_states_view =
    make_host_buffer<dev_velo_kalman_beamline_states_view_t>(arguments, context);
  const auto host_is_backward = make_host_buffer<dev_is_backward_t>(arguments, context);

  const auto host_velo_tracks_view = make_host_buffer<dev_velo_tracks_view_t>(arguments, context);

  for (unsigned i = 0; i < host_event_list.size(); i++) {
    const auto event_number = host_event_list[i];

    const auto velo_tracks_view = host_velo_tracks_view[event_number];
    const auto kalman_beamline_states_view = host_kalman_beamline_states_view[event_number];

    for (unsigned j = 0; j < velo_tracks_view.size(); j++) {
      const auto kalman_beamline_state = kalman_beamline_states_view.state(j);
      const auto track = velo_tracks_view.track(j);

      beamPOCA_tx = kalman_beamline_state.tx();
      beamPOCA_ty = kalman_beamline_state.ty();
      beamPOCA_x = kalman_beamline_state.x();
      beamPOCA_y = kalman_beamline_state.y();
      beamPOCA_z = kalman_beamline_state.z();
      backward = host_is_backward[velo_tracks_view.offset() + j];
      eta = track.eta(kalman_beamline_states_view, backward);

      tree->Fill();
    }
  }
}

void velo_kalman_filter::velo_kalman_filter_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  auto dev_histogram_velo_total_track_eta =
    make_device_buffer<unsigned>(arguments, property<histogram_velo_track_eta_nbins_t>());
  auto dev_histogram_velo_total_track_phi =
    make_device_buffer<unsigned>(arguments, property<histogram_velo_track_phi_nbins_t>());
  auto dev_histogram_velo_total_track_nhits =
    make_device_buffer<unsigned>(arguments, property<histogram_velo_track_nhits_nbins_t>());
  auto dev_histogram_velo_forward_track_eta =
    make_device_buffer<unsigned>(arguments, property<histogram_velo_track_eta_nbins_t>());
  auto dev_histogram_velo_forward_track_phi =
    make_device_buffer<unsigned>(arguments, property<histogram_velo_track_phi_nbins_t>());
  auto dev_histogram_velo_forward_track_nhits =
    make_device_buffer<unsigned>(arguments, property<histogram_velo_track_nhits_nbins_t>());
  auto dev_histogram_velo_backward_track_eta =
    make_device_buffer<unsigned>(arguments, property<histogram_velo_track_eta_nbins_t>());
  auto dev_histogram_velo_backward_track_phi =
    make_device_buffer<unsigned>(arguments, property<histogram_velo_track_phi_nbins_t>());
  auto dev_histogram_velo_backward_track_nhits =
    make_device_buffer<unsigned>(arguments, property<histogram_velo_track_nhits_nbins_t>());
  Allen::memset_async(
    dev_histogram_velo_total_track_eta.data(),
    0,
    dev_histogram_velo_total_track_eta.size() * sizeof(unsigned),
    context);
  Allen::memset_async(
    dev_histogram_velo_total_track_phi.data(),
    0,
    dev_histogram_velo_total_track_phi.size() * sizeof(unsigned),
    context);
  Allen::memset_async(
    dev_histogram_velo_total_track_nhits.data(),
    0,
    dev_histogram_velo_total_track_nhits.size() * sizeof(unsigned),
    context);
  Allen::memset_async(
    dev_histogram_velo_forward_track_eta.data(),
    0,
    dev_histogram_velo_forward_track_eta.size() * sizeof(unsigned),
    context);
  Allen::memset_async(
    dev_histogram_velo_forward_track_phi.data(),
    0,
    dev_histogram_velo_forward_track_phi.size() * sizeof(unsigned),
    context);
  Allen::memset_async(
    dev_histogram_velo_forward_track_nhits.data(),
    0,
    dev_histogram_velo_forward_track_nhits.size() * sizeof(unsigned),
    context);
  Allen::memset_async(
    dev_histogram_velo_backward_track_eta.data(),
    0,
    dev_histogram_velo_backward_track_eta.size() * sizeof(unsigned),
    context);
  Allen::memset_async(
    dev_histogram_velo_backward_track_phi.data(),
    0,
    dev_histogram_velo_backward_track_phi.size() * sizeof(unsigned),
    context);
  Allen::memset_async(
    dev_histogram_velo_backward_track_nhits.data(),
    0,
    dev_histogram_velo_backward_track_nhits.size() * sizeof(unsigned),
    context);
  global_function(velo_kalman_filter)(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(
    arguments,
    constants.dev_beamline.data(),
    dev_histogram_velo_total_track_eta.get(),
    dev_histogram_velo_total_track_phi.get(),
    dev_histogram_velo_total_track_nhits.get(),
    dev_histogram_velo_forward_track_eta.get(),
    dev_histogram_velo_forward_track_phi.get(),
    dev_histogram_velo_forward_track_nhits.get(),
    dev_histogram_velo_backward_track_eta.get(),
    dev_histogram_velo_backward_track_phi.get(),
    dev_histogram_velo_backward_track_nhits.get());

#ifndef ALLEN_STANDALONE
  gaudi_monitoring::fill(
    arguments,
    context,
    std::tuple {std::tuple {dev_histogram_velo_total_track_eta.get(),
                            histogram_velo_total_track_eta,
                            property<histogram_velo_track_eta_min_t>(),
                            property<histogram_velo_track_eta_max_t>()},
                std::tuple {dev_histogram_velo_total_track_phi.get(),
                            histogram_velo_total_track_phi,
                            property<histogram_velo_track_phi_min_t>(),
                            property<histogram_velo_track_phi_max_t>()},
                std::tuple {dev_histogram_velo_total_track_nhits.get(),
                            histogram_velo_total_track_nhits,
                            property<histogram_velo_track_nhits_min_t>(),
                            property<histogram_velo_track_nhits_max_t>()},
                std::tuple {dev_histogram_velo_forward_track_eta.get(),
                            histogram_velo_forward_track_eta,
                            property<histogram_velo_track_eta_min_t>(),
                            property<histogram_velo_track_eta_max_t>()},
                std::tuple {dev_histogram_velo_forward_track_phi.get(),
                            histogram_velo_forward_track_phi,
                            property<histogram_velo_track_phi_min_t>(),
                            property<histogram_velo_track_phi_max_t>()},
                std::tuple {dev_histogram_velo_forward_track_nhits.get(),
                            histogram_velo_forward_track_nhits,
                            property<histogram_velo_track_nhits_min_t>(),
                            property<histogram_velo_track_nhits_max_t>()},
                std::tuple {dev_histogram_velo_backward_track_eta.get(),
                            histogram_velo_backward_track_eta,
                            property<histogram_velo_track_eta_min_t>(),
                            property<histogram_velo_track_eta_max_t>()},
                std::tuple {dev_histogram_velo_backward_track_phi.get(),
                            histogram_velo_backward_track_phi,
                            property<histogram_velo_track_phi_min_t>(),
                            property<histogram_velo_track_phi_max_t>()},
                std::tuple {dev_histogram_velo_backward_track_nhits.get(),
                            histogram_velo_backward_track_nhits,
                            property<histogram_velo_track_nhits_min_t>(),
                            property<histogram_velo_track_nhits_max_t>()}});
#endif
}

/**
 * @brief Calculates the parameters according to a root means square fit
 */
__device__ MiniState least_means_square_fit(const Allen::Views::Velo::Consolidated::Track& track)
{
  MiniState state;

  // Fit parameters
  float s0, sx, sz, sxz, sz2;
  float u0, uy, uz, uyz, uz2;
  s0 = sx = sz = sxz = sz2 = 0.0f;
  u0 = uy = uz = uyz = uz2 = 0.0f;

  // Iterate over hits
  for (unsigned short h = 0; h < track.number_of_hits(); ++h) {
    const auto hit = track.hit(h);
    const auto x = hit.x();
    const auto y = hit.y();
    const auto z = hit.z();

    const auto wx = Velo::Tracking::param_w;
    const auto wx_t_x = wx * x;
    const auto wx_t_z = wx * z;
    s0 += wx;
    sx += wx_t_x;
    sz += wx_t_z;
    sxz += wx_t_x * z;
    sz2 += wx_t_z * z;

    const auto wy = Velo::Tracking::param_w;
    const auto wy_t_y = wy * y;
    const auto wy_t_z = wy * z;
    u0 += wy;
    uy += wy_t_y;
    uz += wy_t_z;
    uyz += wy_t_y * z;
    uz2 += wy_t_z * z;
  }

  // Calculate tx, ty and backward
  const auto dens = 1.0f / (sz2 * s0 - sz * sz);
  state.tx = (sxz * s0 - sx * sz) * dens;
  state.x = (sx * sz2 - sxz * sz) * dens;

  const auto denu = 1.0f / (uz2 * u0 - uz * uz);
  state.ty = (uyz * u0 - uy * uz) * denu;
  state.y = (uy * uz2 - uyz * uz) * denu;

  state.z = -(state.x * state.tx + state.y * state.ty) / (state.tx * state.tx + state.ty * state.ty);
  state.x = state.x + state.tx * state.z;
  state.y = state.y + state.ty * state.z;

  return state;
}

/**
 * @brief Calculates the parameters according to a linear fit between the first and last velo hit
 */
__device__ MiniState linear_fit(const Allen::Views::Velo::Consolidated::Track& track, float* dev_beamline)
{
  MiniState state;

  // Get first and last hits
  const auto first = static_cast<::Velo::HitBase>(track.hit(0));
  const auto last = static_cast<::Velo::HitBase>(track.hit(track.number_of_hits() - 1));

  // Calculate tx, ty
  state.tx = (last.x - first.x) / (last.z - first.z);
  state.ty = (last.y - first.y) / (last.z - first.z);

  // Propagate to the beamline
  auto delta_z = (state.tx * (dev_beamline[0] - last.x) + state.ty * (dev_beamline[1] - last.y)) /
                 (state.tx * state.tx + state.ty * state.ty);
  state.x = last.x + state.tx * delta_z;
  state.y = last.y + state.ty * delta_z;
  state.z = last.z + delta_z;

  return state;
}

__global__ void velo_kalman_filter::velo_kalman_filter(
  velo_kalman_filter::Parameters parameters,
  float* dev_beamline,
  gsl::span<unsigned> dev_histogram_velo_total_track_eta,
  gsl::span<unsigned> dev_histogram_velo_total_track_phi,
  gsl::span<unsigned> dev_histogram_velo_total_track_nhits,
  gsl::span<unsigned> dev_histogram_velo_forward_track_eta,
  gsl::span<unsigned> dev_histogram_velo_forward_track_phi,
  gsl::span<unsigned> dev_histogram_velo_forward_track_nhits,
  gsl::span<unsigned> dev_histogram_velo_backward_track_eta,
  gsl::span<unsigned> dev_histogram_velo_backward_track_phi,
  gsl::span<unsigned> dev_histogram_velo_backward_track_nhits)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const unsigned number_of_events = parameters.dev_number_of_events[0];

  const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];
  const auto total_number_of_tracks = parameters.dev_offsets_all_velo_tracks[number_of_events];

  parameters.dev_velo_kalman_beamline_states_view[event_number] = Allen::Views::Physics::KalmanStates {
    parameters.dev_velo_kalman_beamline_states, parameters.dev_offsets_all_velo_tracks, event_number, number_of_events};

  parameters.dev_velo_kalman_endvelo_states_view[event_number] = Allen::Views::Physics::KalmanStates {
    parameters.dev_velo_kalman_endvelo_states, parameters.dev_offsets_all_velo_tracks, event_number, number_of_events};

  Velo::Consolidated::States kalman_beamline_states {parameters.dev_velo_kalman_beamline_states,
                                                     total_number_of_tracks};
  Velo::Consolidated::States kalman_endvelo_states {parameters.dev_velo_kalman_endvelo_states, total_number_of_tracks};

  for (unsigned i = threadIdx.x; i < velo_tracks_view.size(); i += blockDim.x) {

    const auto track = velo_tracks_view.track(i);

    // Get first estimate of the state , changed least means square fit to linear fit between first and last hit
    const auto lin_fit_at_beamline = linear_fit(track, dev_beamline);
    bool backward = lin_fit_at_beamline.z > track.hit(0).z();
    parameters.dev_is_backward[velo_tracks_view.offset() + i] = backward;

    // Perform a Kalman fit to obtain state at beamline
    const auto kalman_beamline_state = simplified_fit<true>(track, lin_fit_at_beamline, dev_beamline, backward);

    // Perform a Kalman fit in the other direction to obtain state at the end of the Velo
    const auto kalman_endvelo_state = simplified_fit<false>(track, kalman_beamline_state, dev_beamline, backward);

    kalman_beamline_states.set(velo_tracks_view.offset() + i, kalman_beamline_state);
    kalman_endvelo_states.set(velo_tracks_view.offset() + i, kalman_endvelo_state);

    velo_kalman_filter::velo_kalman_filter_t::monitor(
      parameters,
      track,
      kalman_beamline_state,
      dev_histogram_velo_total_track_eta,
      dev_histogram_velo_total_track_phi,
      dev_histogram_velo_total_track_nhits,
      dev_histogram_velo_forward_track_eta,
      dev_histogram_velo_forward_track_phi,
      dev_histogram_velo_forward_track_nhits,
      dev_histogram_velo_backward_track_eta,
      dev_histogram_velo_backward_track_phi,
      dev_histogram_velo_backward_track_nhits);
  }
}

__device__ void velo_kalman_filter::velo_kalman_filter_t::monitor(
  const velo_kalman_filter::Parameters& parameters,
  Allen::Views::Velo::Consolidated::Track velo_track,
  KalmanVeloState beamline_state,
  gsl::span<unsigned> dev_histogram_velo_total_track_eta,
  gsl::span<unsigned> dev_histogram_velo_total_track_phi,
  gsl::span<unsigned> dev_histogram_velo_total_track_nhits,
  gsl::span<unsigned> dev_histogram_velo_forward_track_eta,
  gsl::span<unsigned> dev_histogram_velo_forward_track_phi,
  gsl::span<unsigned> dev_histogram_velo_forward_track_nhits,
  gsl::span<unsigned> dev_histogram_velo_backward_track_eta,
  gsl::span<unsigned> dev_histogram_velo_backward_track_phi,
  gsl::span<unsigned> dev_histogram_velo_backward_track_nhits)
{

  const auto tx = beamline_state.tx;
  const auto ty = beamline_state.ty;
  const auto z_beamline = beamline_state.z;
  const auto first_z = static_cast<::Velo::HitBase>(velo_track.hit(0)).z;
  const auto backward = z_beamline > first_z;
  const auto zeta = backward ? -1.f : 1.f;

  const float slope2 = tx * tx + ty * ty;
  const float rho = std::sqrt(slope2);
  const auto nhits = velo_track.number_of_hits();
  const auto eta = eta_from_rho_z(rho, zeta);
  const auto phi = std::atan2(ty, tx);
  // printf("tx %.4f , ty %.4f, nhits: %d \n", tx,ty,nhits);

  if (eta > parameters.histogram_velo_track_eta_min && eta < parameters.histogram_velo_track_eta_max) {
    const unsigned int bin = static_cast<unsigned int>(
      (eta - parameters.histogram_velo_track_eta_min) * parameters.histogram_velo_track_eta_nbins /
      (parameters.histogram_velo_track_eta_max - parameters.histogram_velo_track_eta_min));
    if (backward) {
      ++dev_histogram_velo_backward_track_eta[bin];
    }
    else {
      ++dev_histogram_velo_forward_track_eta[bin];
    }
    ++dev_histogram_velo_total_track_eta[bin];
  }
  if (phi > parameters.histogram_velo_track_phi_min && phi < parameters.histogram_velo_track_phi_max) {
    const unsigned int bin = static_cast<unsigned int>(
      (phi - parameters.histogram_velo_track_phi_min) * parameters.histogram_velo_track_phi_nbins /
      (parameters.histogram_velo_track_phi_max - parameters.histogram_velo_track_phi_min));
    if (backward) {
      ++dev_histogram_velo_backward_track_phi[bin];
    }
    else {
      ++dev_histogram_velo_forward_track_phi[bin];
    }
    ++dev_histogram_velo_total_track_phi[bin];
  }
  if (nhits > parameters.histogram_velo_track_nhits_min && nhits < parameters.histogram_velo_track_nhits_max) {
    const unsigned int bin = static_cast<unsigned int>(
      (nhits - parameters.histogram_velo_track_nhits_min) * parameters.histogram_velo_track_nhits_nbins /
      (parameters.histogram_velo_track_nhits_max - parameters.histogram_velo_track_nhits_min));
    if (backward) {
      ++dev_histogram_velo_backward_track_nhits[bin];
    }
    else {
      ++dev_histogram_velo_forward_track_nhits[bin];
    }
    ++dev_histogram_velo_total_track_nhits[bin];
  }
}
