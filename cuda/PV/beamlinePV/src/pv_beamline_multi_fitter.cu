#include "pv_beamline_multi_fitter.cuh"

__global__ void pv_beamline_multi_fitter(
  int* dev_atomics_storage,
  uint* dev_velo_track_hit_number,
  PVTrack* dev_pvtracks,
  float* dev_pvtracks_denom,
  float* dev_zpeaks,
  uint* dev_number_of_zpeaks,
  PV::Vertex* dev_multi_fit_vertices,
  uint* dev_number_of_multi_fit_vertices,
  float* dev_beamline)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  uint* number_of_multi_fit_vertices = dev_number_of_multi_fit_vertices + event_number;

  const Velo::Consolidated::Tracks velo_tracks {
    (uint*) dev_atomics_storage, dev_velo_track_hit_number, event_number, number_of_events};

  const uint number_of_tracks = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  const float* zseeds = dev_zpeaks + event_number * PV::max_number_vertices;
  const uint number_of_seeds = dev_number_of_zpeaks[event_number];

  const PVTrack* tracks = dev_pvtracks + event_tracks_offset;

  PV::Vertex* vertices = dev_multi_fit_vertices + event_number * PV::max_number_vertices;
  PV::Vertex vertex;
  float* pvtracks_denom = dev_pvtracks_denom + event_tracks_offset;

  // Precalculate all track denoms
  for (int i=threadIdx.x; i<number_of_tracks; i+=blockDim.x) {
    auto track_denom = 0.f;
    const auto track = tracks[i];

    for (int j=0; j<number_of_seeds; ++j) {
      const auto dz = zseeds[j] - track.z;
      const float2 res = track.x + track.tx * dz;
      const auto chi2 = res.x * res.x * track.W_00 + res.y * res.y * track.W_11;
      track_denom += expf(chi2 * (-0.5f));
    }

    pvtracks_denom[i] = track_denom;
  }

  __syncthreads();

  // make sure that we have one thread per seed
  for (uint i_thisseed = threadIdx.x; i_thisseed < number_of_seeds; i_thisseed += blockDim.x) {
    bool converged = false;
    bool accept = true;
    float vtxcov[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // initial vertex posisiton, use x,y of the beamline and z of the seed
    float2 vtxpos_xy {dev_beamline[0], dev_beamline[1]};
    auto vtxpos_z = zseeds[i_thisseed];
    auto chi2tot = 0.f;
    float sum_weights = 0.f;

    for (uint iter = 0; iter < maxFitIter && !converged; ++iter) {
      auto halfD2Chi2DX2_00 = 0.f;
      auto halfD2Chi2DX2_11 = 0.f;
      auto halfD2Chi2DX2_20 = 0.f;
      auto halfD2Chi2DX2_21 = 0.f;
      auto halfD2Chi2DX2_22 = 0.f;
      float3 halfDChi2DX {0.f, 0.f, 0.f};

      // TODO: Very confusing
      chi2tot = 0.f;

      uint nselectedtracks = 0;

      for (int i = 0; i < number_of_tracks; i++) {
        // compute the chi2
        PVTrackInVertex trk = tracks[i];
        // skip tracks lying outside histogram range
        if (zmin < trk.z && trk.z < zmax) {

          const auto dz = vtxpos_z - trk.z;
          float2 res {0.f, 0.f};
          res = vtxpos_xy - (trk.x + trk.tx * dz);
          const auto chi2 = res.x * res.x * trk.W_00 + res.y * res.y * trk.W_11;
          // compute the weight.
          if (chi2 < maxChi2) {
            ++nselectedtracks;
            // for more information on the weighted fitting, see e.g.
            // Adaptive Multi-vertex fitting, R. Frühwirth, W. Waltenberger
            // https://cds.cern.ch/record/803519/files/p280.pdf

            // expf(-chi2Cut * 0.5f) = 0.000003727f
            auto denom = 0.000003727f + expf(chi2 * (-0.5f));
            // auto nom = 1.f;

            // Calculate nom
            // TODO: This seems to be already calculated above
            // const auto dz_seed = zseeds[i_thisseed] - trk.z;
            // const float2 res_seed = trk.x + trk.tx * dz_seed;
            // const auto chi2_seed = res_seed.x * res_seed.x * trk.W_00 + res_seed.y * res_seed.y * trk.W_11;
            // const auto nom = expf(-chi2_seed * 0.5f);
            const auto nom = expf(chi2 * (-0.5f));

            trk.weight = nom / (denom + pvtracks_denom[i]);
            // trk.weight = nom / denom;

            // unfortunately branchy, but reduces fake rate
            if (trk.weight > minWeight) {
              float3 HWr;
              HWr.x = res.x * trk.W_00;
              HWr.y = res.y * trk.W_11;
              HWr.z = -trk.tx.x * res.x * trk.W_00 - trk.tx.y * res.y * trk.W_11;

              halfDChi2DX = halfDChi2DX + HWr * trk.weight;

              halfD2Chi2DX2_00 += trk.weight * trk.HWH_00;
              halfD2Chi2DX2_11 += trk.weight * trk.HWH_11;
              halfD2Chi2DX2_20 += trk.weight * trk.HWH_20;
              halfD2Chi2DX2_21 += trk.weight * trk.HWH_21;
              halfD2Chi2DX2_22 += trk.weight * trk.HWH_22;

              chi2tot += trk.weight * chi2;
              sum_weights += trk.weight;
            }
          }
        }
      }

      __syncthreads();

      if (nselectedtracks >= 2) {
        // compute the new vertex covariance using analytical inversion
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

        const auto delta_z = -1.f * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z);
        chi2tot += delta_xy.x * halfDChi2DX.x + delta_xy.y * halfDChi2DX.y + delta_z * halfDChi2DX.z;

        // update the position
        vtxpos_xy = vtxpos_xy + delta_xy;
        vtxpos_z = vtxpos_z + delta_z;
        converged = fabsf(delta_z) < maxDeltaZConverged;
      }
      else {
        // Finish loop and do not accept vertex
        converged = true;
        accept = false;
      }
    } // end iteration loop

    if (accept) {
      vertex.chi2 = chi2tot;
      vertex.setPosition(vtxpos_xy, vtxpos_z);
      vertex.setCovMatrix(vtxcov);
      vertex.nTracks = sum_weights;

      // TODO integrate beamline position
      const auto beamlinedx = vertex.position.x - dev_beamline[0];
      const auto beamlinedy = vertex.position.y - dev_beamline[1];
      const auto beamlinerho2 = beamlinedx * beamlinedx + beamlinedy * beamlinedy;
      if (vertex.nTracks >= minNumTracksPerVertex && beamlinerho2 < maxVertexRho2) {
        uint vertex_index = atomicAdd(number_of_multi_fit_vertices, 1);
        vertices[vertex_index] = vertex;
      }
    }
  }
}
