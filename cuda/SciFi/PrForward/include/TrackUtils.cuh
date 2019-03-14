#pragma once

#include <cmath>

#include "SciFiDefinitions.cuh"
#include "PrForwardConstants.cuh"
#include "PrVeloUT.cuh"
#include "HitUtils.cuh"
#include "ParabolaFitting.cuh"
#include "SciFiEventModel.cuh"

/**
   Helper functions related to track properties
 */

// extrapolate x position from given state to z
__host__ __device__ inline float xFromVelo(const float z, MiniState velo_state)
{
  return velo_state.x + (z - velo_state.z) * velo_state.tx;
}

// extrapolate y position from given state to z
__host__ __device__ inline float yFromVelo(const float z, MiniState velo_state)
{
  return velo_state.y + (z - velo_state.z) * velo_state.ty;
}

__host__ __device__ inline float evalCubicParameterization(const float params[4], float z)
{
  float dz = z - SciFi::Tracking::zReference;
  return params[0] + (params[1] + (params[2] + params[3] * dz) * dz) * dz;
}

__host__ __device__ inline float straightLinePropagationFromReferencePlane(const float params[4], float z)
{
  float dz = z - SciFi::Tracking::zReference;
  return params[0] + params[1] * dz;
} 

__host__ __device__ inline float straightLinePropagationFromReferencePlane(const float x0, const float tx, float z)
{
  float dz = z - SciFi::Tracking::zReference;
  return x0 + tx * dz;
} 

__host__ __device__ void getTrackParameters(
  float xAtRef,
  MiniState velo_state,
  const SciFi::Tracking::Arrays* constArrays,
  float trackParams[SciFi::Tracking::nTrackParams]);

__host__ __device__ float calcqOverP(float bx, const SciFi::Tracking::Arrays* constArrays, MiniState velo_state);

__host__ __device__ float zMagnet(MiniState velo_state, const SciFi::Tracking::Arrays* constArrays);

__host__ __device__ float calcDxRef(float pt, MiniState velo_state);

__host__ __device__ float
trackToHitDistance(float trackParameters[SciFi::Tracking::nTrackParams], const SciFi::Hits& scifi_hits, int hit);

__host__ __device__ static inline bool lowerByQuality(SciFi::Tracking::Track t1, SciFi::Tracking::Track t2)
{
  return t1.quality < t2.quality;
}

__host__ __device__ float chi2XHit(const float parsX[4], const SciFi::Hits& scifi_hits, const int hit);

__host__ __device__ bool quadraticFitX(
  const SciFi::Hits& scifi_hits,
  float trackParameters[SciFi::Tracking::nTrackParams],
  int coordToFit[SciFi::Tracking::max_coordToFit],
  int& n_coordToFit,
  PlaneCounter& planeCounter,
  SciFi::Tracking::HitSearchCuts& pars_cur);

__host__ __device__ bool fitYProjection(
  const SciFi::Hits& scifi_hits,
  SciFi::Tracking::Track& track,
  int stereoHits[SciFi::Tracking::max_stereo_hits],
  int& n_stereoHits,
  PlaneCounter& planeCounter,
  MiniState velo_state,
  const SciFi::Tracking::Arrays* constArrays,
  SciFi::Tracking::HitSearchCuts& pars_cur);
