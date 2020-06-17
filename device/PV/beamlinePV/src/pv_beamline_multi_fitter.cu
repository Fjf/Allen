#include "pv_beamline_multi_fitter.cuh"

void pv_beamline_multi_fitter::pv_beamline_multi_fitter_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_multi_fit_vertices_t>(
    arguments, first<host_number_of_selected_events_t>(arguments) * PV::max_number_vertices);
  set_size<dev_number_of_multi_fit_vertices_t>(arguments, first<host_number_of_selected_events_t>(arguments));
  set_size<dev_pvtracks_denom_t>(arguments, first<host_number_of_reconstructed_velo_tracks_t>(arguments));
}

void pv_beamline_multi_fitter::pv_beamline_multi_fitter_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  HostBuffers&,
  cudaStream_t& cuda_stream,
  cudaEvent_t&) const
{
  initialize<dev_number_of_multi_fit_vertices_t>(arguments, 0, cuda_stream);

  global_function(pv_beamline_multi_fitter)(
    dim3(first<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(
    arguments, constants.dev_beamline.data());
}

__global__ void pv_beamline_multi_fitter::pv_beamline_multi_fitter(
  pv_beamline_multi_fitter::Parameters parameters,
  const float* dev_beamline)
{
  const unsigned number_of_events = gridDim.x;
  const unsigned event_number = blockIdx.x;
  unsigned* number_of_multi_fit_vertices = parameters.dev_number_of_multi_fit_vertices + event_number;

  Velo::Consolidated::ConstTracks velo_tracks {
    parameters.dev_atomics_velo, parameters.dev_velo_track_hit_number, event_number, number_of_events};

  const unsigned number_of_tracks = velo_tracks.number_of_tracks(event_number);
  const unsigned event_tracks_offset = velo_tracks.tracks_offset(event_number);

  const float* zseeds = parameters.dev_zpeaks + event_number * PV::max_number_vertices;
  const unsigned number_of_seeds = parameters.dev_number_of_zpeaks[event_number];

  const PVTrack* tracks = parameters.dev_pvtracks + event_tracks_offset;

  PV::Vertex* vertices = parameters.dev_multi_fit_vertices + event_number * PV::max_number_vertices;
  PV::Vertex vertex;
  const float* pvtracks_denom = parameters.dev_pvtracks_denom + event_tracks_offset;

  const float2 seed_pos_xy {dev_beamline[0], dev_beamline[1]};

  // Find out the tracks we have to process
  // Exploit the fact tracks are sorted by z
  int first_track_in_range = -1;
  unsigned number_of_tracks_in_range = 0;
  for (unsigned i = 0; i < number_of_tracks; i++) {
    const auto z = parameters.dev_pvtrack_z[event_tracks_offset + i];
    if (BeamlinePVConstants::Common::zmin < z && z < BeamlinePVConstants::Common::zmax) {
      if (first_track_in_range == -1) {
        first_track_in_range = i;
      }
      ++number_of_tracks_in_range;
    }
  }

  // make sure that we have one thread per seed
  for (unsigned i_thisseed = threadIdx.x; i_thisseed < number_of_seeds; i_thisseed += blockDim.x) {

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

      for (unsigned i = threadIdx.y; i < number_of_tracks_in_range; i += blockDim.y) {
        // compute the chi2
        const PVTrackInVertex& trk = tracks[first_track_in_range + i];

        const auto dz = vtxpos_z - trk.z;
        const float2 res = vtxpos_xy - (trk.x + trk.tx * dz);
        const auto chi2 = res.x * res.x * trk.W_00 + res.y * res.y * trk.W_11;

        // compute the weight.
        if (chi2 < BeamlinePVConstants::MultiFitter::maxChi2) {
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

          const auto track_weight = nom / (denom + pvtracks_denom[first_track_in_range + i] - exp_chi2_0);

          // unfortunately branchy, but reduces fake rate
          // not cuttign on the weights seems to be important for reoslution of high multiplcitiy tracks
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

      // __syncthreads();

      // for (int i = 16; i > 0; i = i / 2) {
      //   halfD2Chi2DX2_00 += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_00, i);
      //   halfD2Chi2DX2_11 += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_11, i);
      //   halfD2Chi2DX2_20 += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_20, i);
      //   halfD2Chi2DX2_21 += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_21, i);
      //   halfD2Chi2DX2_22 += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_22, i);
      //   halfDChi2DX.x += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX.x, i);
      //   halfDChi2DX.y += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX.y, i);
      //   halfDChi2DX.z += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX.z, i);
      //   local_chi2tot += __shfl_down_sync(0xFFFFFFFF, chi2tot, i);
      //   local_sum_weights += __shfl_down_sync(0xFFFFFFFF, sum_weights, i);
      //   nselectedtracks += __shfl_down_sync(0xFFFFFFFF, nselectedtracks, i);
      // }

      // __syncthreads();

      if (threadIdx.y == 0) {
        chi2tot += local_chi2tot;
        sum_weights += local_sum_weights;
        // printf("sum weights %f\n", sum_weights);
        if (nselectedtracks >= BeamlinePVConstants::MultiFitter::minNumTracksPerVertex) {
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

      // converged = __any_sync(0xFFFFFFFF, converged);
    } // end iteration loop

    if (accept && threadIdx.y == 0) {
      vertex.chi2 = chi2tot;
      vertex.setPosition(vtxpos_xy, vtxpos_z);
      vertex.setCovMatrix(vtxcov);
      vertex.nTracks = sum_weights;

      const auto beamlinedx = vertex.position.x - dev_beamline[0];
      const auto beamlinedy = vertex.position.y - dev_beamline[1];
      const auto beamlinerho2 = beamlinedx * beamlinedx + beamlinedy * beamlinedy;
      if (
        nselectedtracks >= BeamlinePVConstants::MultiFitter::minNumTracksPerVertex &&
        beamlinerho2 < BeamlinePVConstants::MultiFitter::maxVertexRho2) {
        unsigned vertex_index = atomicAdd(number_of_multi_fit_vertices, 1);
        vertices[vertex_index] = vertex;
      }
    }
  }
}