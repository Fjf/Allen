#pragma once

#include "ParKalmanMath.cuh"
#include <cstdio>

namespace ParKalmanFilter {

  typedef Vector<5> Vector5;
  typedef SquareMatrix<true, 5> SymMatrix5x5;
  typedef SquareMatrix<false, 5> Matrix5x5;

  // Set a 5x5 diagonal matrix for later use
  [[maybe_unused]] __constant__ static KalmanFloat F_diag[25] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                                                                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};

  // Max number of measurements.
  const int nMaxMeasurements = 41; // 25 VELO + 4 UT + 12 SciFi

  // Max number of bins for the UT <-> SciFi extrapolation.
  const int nBinXMax = 60;
  const int nBinYMax = 50;

  // Number of velo parameters.
  const int nParsV = 10;
  const int nSetsV = 2;

  // Number of velo-UT parameters.
  const int nParsVUT = 30;
  const int nSetsVUT = 2;

  // Number of UT parameters.
  const int nParsUT = 20;
  const int nSetsUT = 7;

  // Number of UTFUT parameters.
  const int nParsUTFUT = 1;
  const int nSetsUTFUT = 1;

  // Number of UTTF parameters.
  const int nParsUTTF = 20;
  const int nSetsUTTF = 2;

  // Number of TFT parameters.
  const int nParsTFT = 20;
  const int nSetsTFT = 2;

  // Number of T parameters.
  const int nParsT = 20;
  const int nSetsT = 46;

  // Number of TLayer parameters.
  const int nParsTLayer = 12;
  const int nSetsTLayer = 2;

  // Number of UTLayer parameters.
  const int nParsUTLayer = 4;
  const int nSetsUTLayer = 1;

  // Some options.
  const bool m_UseForwardMomEstimate = true;
  const bool m_UseForwardChi2Estimate = true;
  const int nMaxOutliers = 2;

  //----------------------------------------------------------------------
  // Tentative output structure.
  struct FittedTrack {

    SymMatrix5x5 cov;
    Vector5 state;

    KalmanFloat z;
    KalmanFloat first_qop;
    KalmanFloat best_qop;
    KalmanFloat chi2;
    KalmanFloat chi2V;
    KalmanFloat chi2T;
    KalmanFloat ipChi2;

    uint ndof;
    uint ndofV;
    uint ndofT;
    uint nhits;

    bool is_muon;

    // Functions for accessing momentum information.
    __device__ __host__ KalmanFloat p() const
    {
      KalmanFloat ret = 1.0f / std::abs(best_qop);
      return ret;
    }

    __device__ __host__ KalmanFloat pt() const {
      KalmanFloat sint = std::sqrt((state[2] * state[2] + state[3] * state[3]) /
        (1.0f + state[2] * state[2] + state[3] * state[3]));
      return sint / std::abs(best_qop);
    }

    __device__ __host__ KalmanFloat px() const
    {
      return state[2] / std::abs(best_qop) / std::sqrt(1.0f + state[2] * state[2] + state[3] * state[3]);
    }

    __device__ __host__ KalmanFloat py() const {
      return state[3] / std::abs(best_qop) /
        std::sqrt(1.0f + state[2] * state[2] + state[3] * state[3]);
    }

    __device__ __host__ KalmanFloat pz() const {
      KalmanFloat cost = 1.0f / std::sqrt(1.0 + state[2] * state[2] + state[3] * state[3]);
      return cost / std::abs(best_qop);
    }

    __device__ __host__ KalmanFloat eta() const {
      return std::atanh(pz() / p());
    }
  };

} // namespace ParKalmanFilter
