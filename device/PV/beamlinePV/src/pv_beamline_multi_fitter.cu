/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "pv_beamline_multi_fitter.cuh"

INSTANTIATE_ALGORITHM(pv_beamline_multi_fitter::pv_beamline_multi_fitter_t)

void pv_beamline_multi_fitter::pv_beamline_multi_fitter_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_multi_fit_vertices_t>(arguments, first<host_number_of_events_t>(arguments) * PV::max_number_vertices);
  set_size<dev_number_of_multi_fit_vertices_t>(arguments, first<host_number_of_events_t>(arguments));
}

void pv_beamline_multi_fitter::pv_beamline_multi_fitter_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_number_of_multi_fit_vertices_t>(arguments, 0, context);

  const auto block_dimension = dim3(warp_size, property<block_dim_y_t>());
  global_function(pv_beamline_multi_fitter)(dim3(size<dev_event_list_t>(arguments)), block_dimension, context)(
    arguments, constants.dev_beamline.data());

  if (property<enable_monitoring_t>()) {
    auto handler = runtime_options.root_service->handle(name());
    auto tree_event = handler.tree("event");
    auto tree_seed_tracks = handler.tree("seed_tracks");
    auto tree_vertices = handler.tree("vertices");
    if (tree_event == nullptr || tree_seed_tracks == nullptr || tree_vertices == nullptr) return;

    unsigned n_multi_fit_vertices = 0u;
    float initial_chi2 = 0.f;

    float x, y, z = 0.f;
    float cov00, cov10, cov11, cov20, cov21, cov22 = 0.f;
    float chi2 = 0.f;
    float nTracks = 0.f;
    unsigned ndof = 0;

    handler.branch(tree_event, "n_multi_fit_vertices", n_multi_fit_vertices);
    handler.branch(tree_seed_tracks, "initial_chi2", initial_chi2);
    handler.branch(tree_vertices, "x", x);
    handler.branch(tree_vertices, "y", y);
    handler.branch(tree_vertices, "z", z);
    handler.branch(tree_vertices, "cov00", cov00);
    handler.branch(tree_vertices, "cov10", cov10);
    handler.branch(tree_vertices, "cov11", cov11);
    handler.branch(tree_vertices, "cov20", cov20);
    handler.branch(tree_vertices, "cov21", cov21);
    handler.branch(tree_vertices, "cov22", cov22);
    handler.branch(tree_vertices, "chi2", chi2);
    handler.branch(tree_vertices, "ndof", ndof);
    handler.branch(tree_vertices, "nTracks", nTracks);

    const auto host_number_of_multi_fit_vertices =
      make_host_buffer<dev_number_of_multi_fit_vertices_t>(arguments, context);
    const auto host_number_of_zpeaks = make_host_buffer<dev_number_of_zpeaks_t>(arguments, context);
    const auto host_zpeaks = make_host_buffer<dev_zpeaks_t>(arguments, context);
    const auto host_event_list = make_host_buffer<dev_event_list_t>(arguments, context);
    const auto host_pvtracks = make_host_buffer<dev_pvtracks_t>(arguments, context);
    const auto host_velo_tracks_view = make_host_buffer<dev_velo_tracks_view_t>(arguments, context);
    const auto host_multi_fit_vertices = make_host_buffer<dev_multi_fit_vertices_t>(arguments, context);

    for (unsigned i = 0; i < size<dev_event_list_t>(arguments); i++) {
      const auto event_number = host_event_list[i];
      n_multi_fit_vertices = host_number_of_multi_fit_vertices[event_number];
      const auto n_peaks = host_number_of_zpeaks[event_number];
      const auto velo_tracks_view = host_velo_tracks_view[event_number];
      const auto n_tracks = velo_tracks_view.size();

      tree_event->Fill();

      for (unsigned i_seed = 0; i_seed < n_peaks; i_seed++) {
        const float seed_pos_z = host_zpeaks[event_number * PV::max_number_vertices + i_seed];

        for (unsigned i_track = 0; i_track < n_tracks; i_track++) {
          const auto trk = host_pvtracks[velo_tracks_view.offset() + i_track];

          const float2 seed_pos_xy = {0.f, 0.f};

          const auto dz = seed_pos_z - trk.z;
          const float2 res = seed_pos_xy - (trk.x + trk.tx * dz);
          initial_chi2 = res.x * res.x * trk.W_00 + res.y * res.y * trk.W_11;

          tree_seed_tracks->Fill();
        }
      }

      for (unsigned i_vertex = 0; i_vertex < n_multi_fit_vertices; i_vertex++) {
        const auto vertex = host_multi_fit_vertices[event_number * PV::max_number_vertices + i_vertex];
        x = vertex.position.x;
        y = vertex.position.y;
        z = vertex.position.z;
        chi2 = vertex.chi2;
        ndof = vertex.ndof;
        nTracks = vertex.nTracks;
        cov00 = vertex.cov00;
        cov10 = vertex.cov10;
        cov11 = vertex.cov11;
        cov20 = vertex.cov20;
        cov21 = vertex.cov21;
        cov22 = vertex.cov22;

        tree_vertices->Fill();
      }
    }
  }
}

__global__ void pv_beamline_multi_fitter::pv_beamline_multi_fitter(
  pv_beamline_multi_fitter::Parameters parameters,
  const float* dev_beamline)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  unsigned* number_of_multi_fit_vertices = parameters.dev_number_of_multi_fit_vertices + event_number;

  const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];

  const float* zseeds = parameters.dev_zpeaks + event_number * PV::max_number_vertices;
  const unsigned number_of_seeds = parameters.dev_number_of_zpeaks[event_number];

  const PVTrack* tracks = parameters.dev_pvtracks + velo_tracks_view.offset();

  PV::Vertex* vertices = parameters.dev_multi_fit_vertices + event_number * PV::max_number_vertices;
  PV::Vertex vertex;
  const float* pvtracks_denom = parameters.dev_pvtracks_denom + velo_tracks_view.offset();

  const float2 seed_pos_xy {dev_beamline[0], dev_beamline[1]};

  // make sure that we have one thread per seed
  for (unsigned i_thisseed = threadIdx.y; i_thisseed < number_of_seeds; i_thisseed += blockDim.y) {
    bool converged = false;
    bool accept = true;
    float vtxcov[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // initial vertex posisiton, use x,y of the beamline and z of the seed
    float2 vtxpos_xy = seed_pos_xy;
    const float seed_pos_z = zseeds[i_thisseed];
    auto vtxpos_z = seed_pos_z;
    float chi2tot = 0.f;
    float sum_weights = 0.f;
    unsigned nselectedtracks = 0;
    const unsigned minTracks = seed_pos_z <= BeamlinePVConstants::Common::SMOG2_pp_separation ?
                                 BeamlinePVConstants::MultiFitter::SMOG2_minNumTracksPerVertex :
                                 BeamlinePVConstants::MultiFitter::pp_minNumTracksPerVertex;
    for (unsigned iter = 0;
         (iter < BeamlinePVConstants::MultiFitter::maxFitIter || iter < BeamlinePVConstants::MultiFitter::minFitIter) &&
         !converged;
         ++iter) {
      auto halfD2Chi2DX2_00 = 0.f;
      auto halfD2Chi2DX2_11 = 0.f;
      auto halfD2Chi2DX2_20 = 0.f;
      auto halfD2Chi2DX2_21 = 0.f;
      auto halfD2Chi2DX2_22 = 0.f;
      float3 halfDChi2DX {0.f, 0.f, 0.f};
      sum_weights = 0.f;

      nselectedtracks = 0;
      chi2tot = 0.f;
      float local_chi2tot = 0.f;
      float local_sum_weights = 0.f;

      for (unsigned i = threadIdx.x; i < velo_tracks_view.size(); i += blockDim.x) {
        // compute the chi2
        const PVTrackInVertex& trk = tracks[i];
        if (BeamlinePVConstants::Common::zmin >= trk.z && trk.z >= BeamlinePVConstants::Common::zmax) continue;

        const auto dz = vtxpos_z - trk.z;
        const float2 res = vtxpos_xy - (trk.x + trk.tx * dz);
        const auto chi2 = res.x * res.x * trk.W_00 + res.y * res.y * trk.W_11;

        // compute the weight.
        if (chi2 < parameters.max_chi2) {
          ++nselectedtracks;
          // for more information on the weighted fitting, see e.g.
          // Adaptive Multi-vertex fitting, R. Frühwirth, W. Waltenberger
          // https://cds.cern.ch/record/803519/files/p280.pdf
          // use seed position for chi2 calculation of nominator
          const float dz_seed = seed_pos_z - trk.z;
          const float2 res_seed = seed_pos_xy - (trk.x + trk.tx * dz_seed);
          const float chi2_seed = res_seed.x * res_seed.x * trk.W_00 + res_seed.y * res_seed.y * trk.W_11;
          const float exp_chi2_0 = expf(chi2_seed * (-0.5f));
          // calculating chi w.r.t to vtx position and not seed posotion very important for resolution of high mult
          // vetices
          const auto nom = expf(chi2 * (-0.5f));

          const auto denom = BeamlinePVConstants::MultiFitter::chi2CutExp + nom;
          // substract this term to avoid double counting

          const auto track_weight = nom / (denom + pvtracks_denom[i] - exp_chi2_0);

          // unfortunately branchy, but reduces fake rate
          // not cutting on the weights seems to be important for resolution of high multiplicity tracks
          if (track_weight > BeamlinePVConstants::MultiFitter::minWeight) {
            const float3 HWr {
              res.x * trk.W_00, res.y * trk.W_11, -trk.tx.x * res.x * trk.W_00 - trk.tx.y * res.y * trk.W_11};

            halfDChi2DX = halfDChi2DX + HWr * track_weight;
            halfD2Chi2DX2_00 += track_weight * trk.HWH_00;
            halfD2Chi2DX2_11 += track_weight * trk.HWH_11;
            halfD2Chi2DX2_20 += track_weight * trk.HWH_20;
            halfD2Chi2DX2_21 += track_weight * trk.HWH_21;
            halfD2Chi2DX2_22 += track_weight * trk.HWH_22;

            local_chi2tot += track_weight * chi2;
            local_sum_weights += track_weight;
          }
        }
      }

#if defined(TARGET_DEVICE_CUDA) || defined(TARGET_DEVICE_HIP)
      // Use CUDA warp-level primitives for adding up some numbers onto a single
      // thread without using any shared memory.
      // See https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
      for (int i = warp_size / 2; i > 0; i = i / 2) {
        halfD2Chi2DX2_00 += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_00, i);
        halfD2Chi2DX2_11 += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_11, i);
        halfD2Chi2DX2_20 += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_20, i);
        halfD2Chi2DX2_21 += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_21, i);
        halfD2Chi2DX2_22 += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_22, i);
        halfDChi2DX.x += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX.x, i);
        halfDChi2DX.y += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX.y, i);
        halfDChi2DX.z += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX.z, i);
        local_chi2tot += __shfl_down_sync(0xFFFFFFFF, local_chi2tot, i);
        local_sum_weights += __shfl_down_sync(0xFFFFFFFF, local_sum_weights, i);
        nselectedtracks += __shfl_down_sync(0xFFFFFFFF, nselectedtracks, i);
      }
#endif

      if (threadIdx.x == 0) {
        chi2tot += local_chi2tot;
        sum_weights += local_sum_weights;
        // printf("sum weights %f\n", sum_weights);
        if (nselectedtracks >= minTracks) {
          // compute the new vertex covariance using analytical inversion
          // dividing matrix elements not important for resoltuon of high mult pvs
          const auto a00 = halfD2Chi2DX2_00;
          const auto a11 = halfD2Chi2DX2_11;
          const auto a20 = halfD2Chi2DX2_20;
          const auto a21 = halfD2Chi2DX2_21;
          const auto a22 = halfD2Chi2DX2_22;

          const auto det = a00 * (a22 * a11 - a21 * a21) + a20 * (-a11 * a20);
          const auto inv_det = 1.f / det;

          // maybe we should catch the case when det = 0
          // if (det == 0) return false;

          vtxcov[0] = (a22 * a11 - a21 * a21) * inv_det;
          vtxcov[1] = -(-a20 * a21) * inv_det;
          vtxcov[2] = (a22 * a00 - a20 * a20) * inv_det;
          vtxcov[3] = (-a20 * a11) * inv_det;
          vtxcov[4] = -(a21 * a00) * inv_det;
          vtxcov[5] = (a11 * a00) * inv_det;

          const float2 delta_xy {
            -1.f * (vtxcov[0] * halfDChi2DX.x + vtxcov[1] * halfDChi2DX.y + vtxcov[3] * halfDChi2DX.z),
            -1.f * (vtxcov[1] * halfDChi2DX.x + vtxcov[2] * halfDChi2DX.y + vtxcov[4] * halfDChi2DX.z)};

          const auto delta_z =
            -1.f * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z);
          chi2tot += delta_xy.x * halfDChi2DX.x + delta_xy.y * halfDChi2DX.y + delta_z * halfDChi2DX.z;

          // update the position
          vtxpos_xy = vtxpos_xy + delta_xy;
          vtxpos_z = vtxpos_z + delta_z;
          converged = fabsf(delta_z) < BeamlinePVConstants::MultiFitter::maxDeltaZConverged;
        }
        else {
          // Finish loop and do not accept vertex
          converged = true;
          accept = false;
        }
      }

#if defined(TARGET_DEVICE_CUDA) || defined(TARGET_DEVICE_HIP)
      // Synchronize the value of thread 0 in the warp across the entire warp
      // See https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
      vtxpos_xy.x = __shfl_sync(0xFFFFFFFF, vtxpos_xy.x, 0);
      vtxpos_xy.y = __shfl_sync(0xFFFFFFFF, vtxpos_xy.y, 0);
      vtxpos_z = __shfl_sync(0xFFFFFFFF, vtxpos_z, 0);
      converged = __shfl_sync(0xFFFFFFFF, converged, 0);
#endif
    } // end iteration loop

    if (accept && threadIdx.x == 0) {
      vertex.chi2 = chi2tot;
      vertex.setPosition(vtxpos_xy, vtxpos_z);
      vertex.setCovMatrix(vtxcov);
      vertex.nTracks = sum_weights;

      const auto beamlinedx = vertex.position.x - dev_beamline[0];
      const auto beamlinedy = vertex.position.y - dev_beamline[1];
      const auto beamlinerho2 = beamlinedx * beamlinedx + beamlinedy * beamlinedy;
      const auto minTracks = vertex.position.z <= BeamlinePVConstants::Common::SMOG2_pp_separation ?
                               BeamlinePVConstants::MultiFitter::SMOG2_minNumTracksPerVertex :
                               BeamlinePVConstants::MultiFitter::pp_minNumTracksPerVertex;
      if (nselectedtracks >= minTracks && beamlinerho2 < BeamlinePVConstants::MultiFitter::maxVertexRho2) {
        unsigned vertex_index = atomicAdd(number_of_multi_fit_vertices, 1);
        vertices[vertex_index] = vertex;
      }
    }
  }
}
