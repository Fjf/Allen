/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include "BeamCrossingLine.cuh"
#include "Event/ODIN.h"

// Explicit instantiation
INSTANTIATE_LINE(beam_crossing_line::beam_crossing_line_t, beam_crossing_line::Parameters)

__device__ bool beam_crossing_line::beam_crossing_line_t::select(
  const Parameters& parameters,
  std::tuple<const ODINData&> input)
{
  const unsigned bxt = to_integral(LHCb::ODIN {std::get<0>(input)}.bunchCrossingType());
  if (bxt == parameters.beam_crossing_type) return true;

  return false;
}
