/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "pv_beamline_extrapolate.cuh"

INSTANTIATE_ALGORITHM(pv_beamline_extrapolate::pv_beamline_extrapolate_t)

void pv_beamline_extrapolate::pv_beamline_extrapolate_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_pvtracks_t>(arguments, first<host_number_of_reconstructed_velo_tracks_t>(arguments));
}

void pv_beamline_extrapolate::pv_beamline_extrapolate_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers&,
  const Allen::Context& context) const
{
  global_function(pv_beamline_extrapolate)(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(
    arguments);

  if (property<enable_monitoring_t>()) {
    auto handler = runtime_options.root_service->handle(name());
    auto tree_event = handler.tree("event");
    auto tree_tracks = handler.tree("tracks");
    if (tree_event == nullptr || tree_tracks == nullptr) return;

    unsigned n_tracks = 0u;
    float trk_x = 0.f;
    float trk_y = 0.f;
    float trk_diff_beamline_x = 0.f;
    float trk_diff_beamline_y = 0.f;
    float blchi2 = 0.f;
    float zerr = 0.f;

    handler.branch(tree_event, "n_tracks", n_tracks);
    handler.branch(tree_tracks, "trk_diff_beamline_x", trk_diff_beamline_x);
    handler.branch(tree_tracks, "trk_diff_beamline_y", trk_diff_beamline_y);
    handler.branch(tree_tracks, "trk_x", trk_x);
    handler.branch(tree_tracks, "trk_y", trk_y);
    handler.branch(tree_tracks, "blchi2", blchi2);
    handler.branch(tree_tracks, "zerr", zerr);

    const auto host_pvtracks = make_host_buffer<dev_pvtracks_t>(arguments, context);
    const auto host_velo_tracks_view = make_host_buffer<dev_velo_tracks_view_t>(arguments, context);
    const auto host_event_list = make_host_buffer<dev_event_list_t>(arguments, context);

    // To do: get beamline from constants on the host
    float beamline_x = 0.;
    float beamline_y = 0.;

    for (unsigned i = 0; i < host_event_list.size(); i++) {
      const auto event_number = host_event_list[i];
      const auto velo_tracks_view = host_velo_tracks_view[event_number];
      n_tracks = velo_tracks_view.size();
      tree_event->Fill();

      for (unsigned j = 0; j < n_tracks; j++) {
        const auto trk = host_pvtracks[velo_tracks_view.offset() + j];
        trk_x = trk.x.x;
        trk_y = trk.x.y;
        trk_diff_beamline_x = trk_x - beamline_x;
        trk_diff_beamline_y = trk_y - beamline_y;
        const auto diffx2 = (trk_x - beamline_x) * (trk_x - beamline_x);
        const auto diffy2 = (trk_y - beamline_y) * (trk_y - beamline_y);
        blchi2 = diffx2 * trk.W_00 + diffy2 * trk.W_11;

        const float zweight = trk.tx.x * trk.tx.x * trk.W_00 + trk.tx.y * trk.tx.y * trk.W_11;
        zerr = 1.f / sqrtf(zweight);

        tree_tracks->Fill();
      }
    }
  }
}

__global__ void pv_beamline_extrapolate::pv_beamline_extrapolate(pv_beamline_extrapolate::Parameters parameters)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];
  const auto velo_states_view = parameters.dev_velo_states_view[event_number];

  for (unsigned index = threadIdx.x; index < velo_tracks_view.size(); index += blockDim.x) {
    parameters.dev_pvtracks[velo_tracks_view.offset() + index] = PVTrack {velo_states_view.state(index)};
  }
}
