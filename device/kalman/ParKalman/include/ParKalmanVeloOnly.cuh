/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "Argument.cuh"
#include "KalmanParametrizations.cuh"
#include "ParKalmanDefinitions.cuh"
#include "ParKalmanMath.cuh"
#include "ParKalmanMethods.cuh"
#include "SciFiConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "VeloConsolidated.cuh"
#include "AssociateConsolidated.cuh"
#include "States.cuh"
#include "SciFiDefinitions.cuh"
#include "AlgorithmTypes.cuh"
#include "PackageKalman.cuh"
#include "PV_Definitions.cuh"
#include "ParticleTypes.cuh"

typedef Vector<10> Vector10;
typedef Vector<2> Vector2;
typedef SquareMatrix<true, 2> SymMatrix2x2;
typedef SquareMatrix<false, 2> Matrix2x2;

static constexpr float pixelErr = 0.0125f;

static constexpr float scatterSensorParameter_VPHit2VPHit_cms = 1.48;
static constexpr float scatterSensorParameter_VPHit2VPHit_etaxx = 0.643;
static constexpr float scatterSensorParameter_VPHit2VPHit_etaxtx = 0.526;
static constexpr float scatterSensorParameter_VPHit2VPHit_Eloss = 0.592;

static constexpr float scatterSensorParameter_VPHit2ClosestToBeam_cms = 2.91;
static constexpr float scatterSensorParameter_VPHit2ClosestToBeam_etaxx = 0.808;
static constexpr float scatterSensorParameter_VPHit2ClosestToBeam_etaxtx = 0.793;
static constexpr float scatterSensorParameter_VPHit2ClosestToBeam_Eloss = 1.29;

static constexpr float rffoilscatter = 0.6;

__device__ void simplified_step(
  const float z,
  const float zhit,
  const float xhit,
  const float winv,
  float& x,
  float& tx,
  float& qop,
  float& covXX,
  float& covXTx,
  float& covTxTx,
  float& chi2);

__device__ void extrapolate_velo_only(
  KalmanFloat zFrom,
  KalmanFloat zTo,
  Vector5& x,
  Matrix5x5& F,
  SymMatrix5x5& Q,
  const ParKalmanFilter::KalmanParametrizations* params);

__device__ void predict_velo_only(
  const Velo::Consolidated::Hits& hits,
  int nHit,
  Vector5& x,
  SymMatrix5x5& C,
  KalmanFloat& lastz,
  const ParKalmanFilter::KalmanParametrizations* params);

__device__ void
update_velo_only(const Velo::Consolidated::Hits& hits, int nHit, Vector5& x, SymMatrix5x5& C, KalmanFloat& chi2);

__device__ void velo_only_fit(
  const Velo::Consolidated::Hits& velo_hits,
  const unsigned n_velo_hits,
  const KalmanFloat init_qop,
  const KalmanParametrizations* kalman_params,
  FittedTrack& track);

__device__ void simplified_fit(
  const Allen::Views::Velo::Consolidated::Track& velo_track,
  const KalmanFloat init_qop,
  FittedTrack& track);

__device__ void propagate_to_beamline(FittedTrack& track);

namespace kalman_velo_only {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_scifi_tracks_t, unsigned) host_number_of_reconstructed_scifi_tracks;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    DEVICE_INPUT(dev_offsets_all_velo_tracks_t, unsigned) dev_atomics_velo;
    DEVICE_INPUT(dev_offsets_velo_track_hit_number_t, unsigned) dev_velo_track_hit_number;
    DEVICE_INPUT(dev_scifi_tracks_view_t, Allen::Views::SciFi::Consolidated::Tracks) dev_scifi_tracks_view;
    DEVICE_INPUT(dev_offsets_forward_tracks_t, unsigned) dev_atomics_scifi;
    DEVICE_INPUT(dev_offsets_scifi_track_hit_number_t, unsigned) dev_scifi_track_hit_number;
    DEVICE_INPUT(dev_scifi_qop_t, float) dev_scifi_qop;
    DEVICE_INPUT(dev_scifi_states_t, MiniState) dev_scifi_states;
    DEVICE_INPUT(dev_scifi_track_ut_indices_t, unsigned) dev_scifi_track_ut_indices;
    DEVICE_INPUT(dev_velo_pv_ip_t, char) dev_velo_pv_ip;
    DEVICE_INPUT(dev_multi_final_vertices_t, PV::Vertex) dev_multi_final_vertices;
    DEVICE_INPUT(dev_number_of_multi_final_vertices_t, unsigned) dev_number_of_multi_final_vertices;
    DEVICE_INPUT(dev_is_muon_t, bool) dev_is_muon;
    DEVICE_OUTPUT(dev_kf_tracks_t, ParKalmanFilter::FittedTrack) dev_kf_tracks;
    DEVICE_OUTPUT(dev_kalman_fit_results_t, char) dev_kalman_fit_results;
    DEVICE_OUTPUT(
      dev_kalman_states_view_t, 
      Allen::Views::Physics::KalmanStates,
      dev_kalman_fit_results_t,
      dev_offsets_forward_tracks_t) 
    dev_kalman_states_view;
    DEVICE_OUTPUT(dev_kalman_pv_ipchi2_t, char) dev_kalman_pv_ipchi2;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
  };

  __global__ void kalman_velo_only(Parameters parameters);

  __global__ void kalman_pv_ipchi2(Parameters parameters);

  struct kalman_velo_only_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      HostBuffers& host_buffers,
      const Allen::Context& context) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{256, 1, 1}}};
  };
} // namespace kalman_velo_only
