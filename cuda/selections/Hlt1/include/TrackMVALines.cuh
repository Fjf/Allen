#pragma once

#include "ParKalmanDefinitions.cuh"
#include "ParKalmanMath.cuh"
#include "PV_Definitions.cuh"
#include "VertexDefinitions.cuh"
#include "SystemOfUnits.h"

namespace TrackMVALines {

  // One track parameters.
  const float maxChi2Ndof = 10000.0; // Large for now until we better understand the parameterized Kalman fit quality.
  const float minPt = 1000.0f / Gaudi::Units::GeV;
  const float maxPt = 25000.0f / Gaudi::Units::GeV;
  const float minIPChi2 = 7.4f;
  const float param1 = 1.0f;
  const float param2 = 1.0f;
  const float param3 = 1.1f;
  const float alpha = 2000.0f;

  // Two track parameters.
  const float minComboPt = 2000.0f / Gaudi::Units::MeV;
  const float maxVertexChi2 = 10.0f;
  const float minMCor = 1000.0f / Gaudi::Units::MeV;
  const float minEta = 2.0f;
  const float maxEta = 5.0f;
  const int maxNTrksAssoc = 2; // Placeholder. To be replaced with MVA selection.
  const float minFDChi2 = 0.0f; // Placeholder. To be replaced with MVA selection.

  // Selections.
  __device__ bool OneTrackMVA(const ParKalmanFilter::FittedTrack& track);
  __device__ bool TwoTrackMVA(const VertexFit::TrackMVAVertex& vertex);  
}
