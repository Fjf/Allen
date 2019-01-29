#pragma once

#include "BeamlinePVConstants.cuh"
#include "Common.h"
#include "Handler.cuh"
#include "Arguments.cuh"
#include "TrackBeamLineVertexFinder.cuh"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "patPV_Definitions.cuh"
#include <cstdint>

__global__ void
pv_beamline_histo(int* dev_atomics_storage, uint* dev_velo_track_hit_number, PVTrack* dev_pvtracks, float* dev_zhisto);

ALGORITHM(pv_beamline_histo, pv_beamline_histo_t,
  ARGUMENTS(
    dev_atomics_velo,
    dev_velo_track_hit_number,
    dev_pvtracks,
    dev_zhisto
))
