#pragma once

#include <tuple>

// All includes of all algorithms
#include "PrefixSum.cuh"
#include "EstimateInputSize.cuh"
#include "MaskedVeloClustering.cuh"
#include "CalculatePhiAndSort.cuh"
#include "SearchByTriplet.cuh"
#include "ConsolidateVelo.cuh"
#include "UTCalculateNumberOfHits.cuh"
#include "UTPreDecode.cuh"
#include "UTFindPermutation.cuh"
#include "UTDecodeRawBanksInOrder.cuh"
#include "VeloEventModel.cuh"
#include "ConsolidateUT.cuh"
#include "SciFiCalculateClusterCountV6.cuh"
#include "SciFiPreDecodeV6.cuh"
#include "SciFiRawBankDecoderV6.cuh"
#include "SciFiCalculateClusterCountV5.cuh"
#include "SciFiPreDecodeV5.cuh"
#include "SciFiRawBankDecoderV5.cuh"
#include "SciFiCalculateClusterCountV4.cuh"
#include "SciFiDirectDecoderV4.cuh"
#include "SciFiPreDecodeV4.cuh"
#include "SciFiRawBankDecoderV4.cuh"
#include "ConsolidateSciFi.cuh"
#include "SearchWindows.cuh"
#include "CompassUT.cuh"
#include "VeloKalmanFilter.cuh"
#include "GetSeeds.cuh"
#include "FitSeeds.cuh"
#include "pv_beamline_extrapolate.cuh"
#include "pv_beamline_histo.cuh"
#include "pv_beamline_peak.cuh"
#include "pv_beamline_calculate_denom.cuh"
#include "pv_beamline_multi_fitter.cuh"
#include "pv_beamline_cleanup.cuh"
#include "VeloPVIP.cuh"
#include "IsMuon.cuh"
#include "MuonFeaturesExtraction.cuh"
#include "MuonCatboostEvaluator.cuh"
#include "ParKalmanFilter.cuh"
#include "ParKalmanVeloOnly.cuh"
#include "LFSearchInitialWindows.cuh"
#include "LFTripletSeeding.cuh"
#include "LFTripletKeepBest.cuh"
#include "LFExtendTracksX.cuh"
#include "LFExtendTracksUV.cuh"
#include "LFQualityFilter.cuh"
#include "LFQualityFilterX.cuh"
#include "LFQualityFilterLength.cuh"
#include "LFCalculateParametrization.cuh"
#include "LFLeastMeanSquareFit.cuh"
#include "MuonDecoding.cuh"
#include "MuonPreDecoding.cuh"
#include "MuonSortBySRQ.cuh"
#include "MuonAddCoordsCrossingMaps.cuh"
#include "KalmanPVIPChi2.cuh"
#include "VertexFitter.cuh"
#include "VertexDefinitions.cuh"
#include "TrackMVALines.cuh"
#include "RunHlt1.cuh"
#include "MuonSortByStation.cuh"
#include "PrepareRawBanks.cuh"
#include "VeloCopyTrackHitNumber.cuh"
#include "UTCopyTrackHitNumber.cuh"
#include "SciFiCopyTrackHitNumber.cuh"

#include "CpuInitEventList.h"
#include "CpuGlobalEventCut.h"
#include "CpuVeloPrefixSumNumberOfClusters.h"
#include "CpuVeloPrefixSumNumberOfTracks.h"
#include "CpuVeloPrefixSumNumberOfTrackHits.h"
#include "CpuUTPrefixSumNumberOfHits.h"
#include "CpuUTPrefixSumNumberOfTracks.h"
#include "CpuUTPrefixSumNumberOfTrackHits.h"
#include "CpuSciFiPrefixSumNumberOfHits.h"
#include "CpuSciFiPrefixSumNumberOfTracks.h"
#include "CpuSciFiPrefixSumNumberOfTrackHits.h"
#include "CpuMuonPrefixSumStorage.h"
#include "CpuMuonPrefixSumStation.h"
#include "CpuSVPrefixSumOffsets.h"

#define SEQUENCE_T(...) typedef std::tuple<__VA_ARGS__> configured_sequence_t;

// SEQUENCE must be defined at compile time.
// Values passed at compile time should match
// the name of the file in "sequences/<filename>.cuh":
//
// "cmake -DSEQUENCE=<sequence_name> .." matches "sequences/<sequence_name>.cuh"
//
// eg.
// "cmake -DSEQUENCE=DefaultSequence .." (or just "cmake ..") matches "sequences/DefaultSequence.cuh"
// "cmake -DSEQUENCE=Velo .." matches "sequences/Velo.cuh"

#include "ConfiguredSequence.h"
