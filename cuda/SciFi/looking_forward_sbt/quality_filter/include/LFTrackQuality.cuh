#pragma once

#include "LookingForwardConstants.cuh"
#include "LookingForwardTools.cuh"
#include "SciFiEventModel.cuh"
#include "TMVA_Forward.cuh"
#include "TMVA_Forward_1.cuh"
#include "TMVA_Forward_2.cuh"
#include "TrackUtils.cuh"
#include "LFFitting.cuh"

__device__ float lf_track_quality (SciFi::TrackHits& track,
  const MiniState& velo_state,
  const float VeloUT_qOverP,
  const SciFi::Tracking::Arrays* constArrays,
  const SciFi::Tracking::TMVA* tmva1,
  const SciFi::Tracking::TMVA* tmva2,
  const SciFi::Hits& scifi_hits,
  const int event_offset);