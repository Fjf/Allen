#include "BinarySearch.cuh"
#include "VeloTools.cuh"
#include "CalculateWindows.cuh"
#include "SearchWindows.cuh"

//=============================================================================
// Reject tracks outside of acceptance or pointing to the beam pipe
//=============================================================================
__device__ bool velo_track_in_UTA_acceptance(const MiniState& state)
{
  const float xMidUT = state.x + state.tx * (UT::Constants::zMidUT - state.z);
  const float yMidUT = state.y + state.ty * (UT::Constants::zMidUT - state.z);

  if (xMidUT * xMidUT + yMidUT * yMidUT < UT::Constants::centralHoleSize * UT::Constants::centralHoleSize) return false;
  if ((fabsf(state.tx) > UT::Constants::maxXSlope) || (fabsf(state.ty) > UT::Constants::maxYSlope)) return false;

  if (
    UT::Constants::passTracks && fabsf(xMidUT) < UT::Constants::passHoleSize &&
    fabsf(yMidUT) < UT::Constants::passHoleSize) {
    return false;
  }

  return true;
}

//=========================================================================
// Check if hit is inside tolerance and refine by Y
//=========================================================================
__device__ void tol_refine(
  int& first_candidate,
  int& number_of_candidates,
  UT::ConstHits& ut_hits,
  const MiniState& velo_state,
  const float invNormfact,
  const float xTolNormFact,
  const float dxDy,
  const float y_tol,
  const float y_tol_slope)
{
  bool first_found = false;
  const auto const_first_candidate = first_candidate;
  int last_candidate = first_candidate;
  for (int candidate_i = 0; candidate_i < number_of_candidates; ++candidate_i) {
    const auto i = const_first_candidate + candidate_i;

    const auto zInit = ut_hits.zAtYEq0(i);
    const auto yApprox = velo_state.y + velo_state.ty * (zInit - velo_state.z);
    const auto xOnTrackProto = velo_state.x + velo_state.tx * (zInit - velo_state.z);
    const auto xx = ut_hits.xAt(i, yApprox, dxDy);
    const auto dx = xx - xOnTrackProto;

    if (
      dx >= -xTolNormFact && dx <= xTolNormFact &&
      !ut_hits.isNotYCompatible(i, yApprox, y_tol + y_tol_slope * fabsf(dx * invNormfact))) {
      // It is compatible
      if (!first_found) {
        first_found = true;
        first_candidate = i;
      }
      last_candidate = i;
    }
  }

  if (!first_found) {
    first_candidate = 0;
    number_of_candidates = 0;
  } else {
    number_of_candidates = last_candidate - first_candidate + 1;
  }
}

//=============================================================================
// Get the windows
//=============================================================================
__device__ std::tuple<int, int, int, int, int, int, int, int, int, int> calculate_windows(
  const int layer,
  const MiniState& velo_state,
  const float* fudge_factors,
  UT::ConstHits& ut_hits,
  const UT::HitOffsets& ut_hit_offsets,
  const float* ut_dxDy,
  const float* dev_unique_sector_xs,
  const uint* dev_unique_x_sector_layer_offsets,
  const float y_tol,
  const float y_tol_slope,
  const float min_pt,
  const float min_momentum)
{
  // -- This is hardcoded, so faster
  // -- If you ever change the Table in the magnet tool, this will be wrong
  const float absSlopeY = fabsf(velo_state.ty);
  const int index = (int) (absSlopeY * 100 + 0.5f);
  assert(3 + 4 * index < UTMagnetTool::N_dxLay_vals);
  const float normFact[4] {
    fudge_factors[4 * index], fudge_factors[1 + 4 * index], fudge_factors[2 + 4 * index], fudge_factors[3 + 4 * index]};

  // -- this 500 seems a little odd...
  // to do: change back!
  const float invTheta = min(500.0f, 1.0f / sqrtf(velo_state.tx * velo_state.tx + velo_state.ty * velo_state.ty));
  const float minMom = max(min_pt * invTheta, min_momentum);
  const float xTol = fabsf(1.0f / (UT::Constants::distToMomentum * minMom));
  // const float yTol     = UT::Constants::yTol + UT::Constants::yTolSlope * xTol;

  int layer_offset = ut_hit_offsets.layer_offset(layer);

  const float dx_dy = ut_dxDy[layer];
  const float z_at_layer = ut_hits.zAtYEq0(layer_offset);
  const float y_track = velo_state.y + velo_state.ty * (z_at_layer - velo_state.z);
  const float x_track = velo_state.x + velo_state.tx * (z_at_layer - velo_state.z);
  const float invNormFact = 1.0f / normFact[layer];
  const float xTolNormFact = xTol * invNormFact;

  // Second sector group search
  // const float tolerance_in_x = xTol * invNormFact;

  // Find sector group for lowerBoundX and upperBoundX
  const int first_sector_group_in_layer = dev_unique_x_sector_layer_offsets[layer];
  const int last_sector_group_in_layer = dev_unique_x_sector_layer_offsets[layer + 1];
  const int sector_group_size = last_sector_group_in_layer - first_sector_group_in_layer;

  const int local_sector_group =
    binary_search_leftmost(dev_unique_sector_xs + first_sector_group_in_layer, sector_group_size, x_track);
  int sector_group = first_sector_group_in_layer + local_sector_group;

  int first_candidate = 0, number_of_candidates = 0;
  int left_group_first_candidate = 0, left_group_number_of_candidates = 0;
  int left2_group_first_candidate = 0, left2_group_number_of_candidates = 0;
  int right_group_first_candidate = 0, right_group_number_of_candidates = 0;
  int right2_group_first_candidate = 0, right2_group_number_of_candidates = 0;

  // Get correct index position in array
  sector_group -= 1;
  // central sector group
  if ((sector_group + 1) < last_sector_group_in_layer && sector_group > first_sector_group_in_layer) {
    const auto sector_candidates = find_candidates_in_sector_group(
      ut_hits,
      ut_hit_offsets,
      velo_state,
      dev_unique_sector_xs,
      x_track,
      y_track,
      dx_dy,
      invNormFact,
      xTolNormFact,
      sector_group,
      y_tol,
      y_tol_slope);

    first_candidate = std::get<0>(sector_candidates);
    number_of_candidates = std::get<1>(sector_candidates);
  }

  // left sector group
  const int left_group = sector_group - 1;
  if ((left_group + 1) < last_sector_group_in_layer && left_group > first_sector_group_in_layer) {
    // Valid sector group to find compatible hits
    const auto left_group_candidates = find_candidates_in_sector_group(
      ut_hits,
      ut_hit_offsets,
      velo_state,
      dev_unique_sector_xs,
      x_track,
      y_track,
      dx_dy,
      invNormFact,
      xTolNormFact,
      left_group,
      y_tol,
      y_tol_slope);

    left_group_first_candidate = std::get<0>(left_group_candidates);
    left_group_number_of_candidates = std::get<1>(left_group_candidates);
  }

  // left-left sector group
  const int left2_group = sector_group - 2;
  if ((left2_group + 1) < last_sector_group_in_layer && left2_group > first_sector_group_in_layer) {
    // Valid sector group to find compatible hits
    const auto left2_group_candidates = find_candidates_in_sector_group(
      ut_hits,
      ut_hit_offsets,
      velo_state,
      dev_unique_sector_xs,
      x_track,
      y_track,
      dx_dy,
      invNormFact,
      xTolNormFact,
      left2_group,
      y_tol,
      y_tol_slope);

    left2_group_first_candidate = std::get<0>(left2_group_candidates);
    left2_group_number_of_candidates = std::get<1>(left2_group_candidates);
  }

  // right sector group
  const int right_group = sector_group + 1;
  if ((right_group + 1) < last_sector_group_in_layer && right_group > first_sector_group_in_layer) {
    // Valid sector group to find compatible hits
    const auto right_group_candidates = find_candidates_in_sector_group(
      ut_hits,
      ut_hit_offsets,
      velo_state,
      dev_unique_sector_xs,
      x_track,
      y_track,
      dx_dy,
      invNormFact,
      xTolNormFact,
      right_group,
      y_tol,
      y_tol_slope);

    right_group_first_candidate = std::get<0>(right_group_candidates);
    right_group_number_of_candidates = std::get<1>(right_group_candidates);
  }

  // right-right sector group
  const int right2_group = sector_group + 2;
  if ((right2_group + 1) < last_sector_group_in_layer && right2_group > first_sector_group_in_layer) {
    // Valid sector group to find compatible hits
    const auto right2_group_candidates = find_candidates_in_sector_group(
      ut_hits,
      ut_hit_offsets,
      velo_state,
      dev_unique_sector_xs,
      x_track,
      y_track,
      dx_dy,
      invNormFact,
      xTolNormFact,
      right2_group,
      y_tol,
      y_tol_slope );

    right2_group_first_candidate = std::get<0>(right2_group_candidates);
    right2_group_number_of_candidates = std::get<1>(right2_group_candidates);
  }

  return std::tuple<int, int, int, int, int, int, int, int, int, int> {first_candidate,
                                                                       number_of_candidates,
                                                                       left_group_first_candidate,
                                                                       left_group_number_of_candidates,
                                                                       right_group_first_candidate,
                                                                       right_group_number_of_candidates,
                                                                       left2_group_first_candidate,
                                                                       left2_group_number_of_candidates,
                                                                       right2_group_first_candidate,
                                                                       right2_group_number_of_candidates};
}

__device__ std::tuple<int, int> find_candidates_in_sector_group(
  UT::ConstHits& ut_hits,
  const UT::HitOffsets& ut_hit_offsets,
  const MiniState& velo_state,
  const float* dev_unique_sector_xs,
  const float x_track,
  const float y_track,
  const float dx_dy,
  const float invNormFact,
  const float xTolNormFact,
  const int sector_group,
  const float y_tol,
  const float y_tol_slope)
{
  const float x_at_left_sector = dev_unique_sector_xs[sector_group];
  const float x_at_right_sector = dev_unique_sector_xs[sector_group + 1];
  const float xx_at_left_sector = x_at_left_sector + y_track * dx_dy;
  const float xx_at_right_sector = x_at_right_sector + y_track * dx_dy;
  const float dx_max = max(xx_at_left_sector - x_track, xx_at_right_sector - x_track);

  const float tol = y_tol + y_tol_slope * fabsf(dx_max * invNormFact);
  const uint sector_group_offset = ut_hit_offsets.sector_group_offset(sector_group);

  int number_of_candidates = 0;

  // Find the first candidate (y_track - tol) employing a normal binary search
  int first_candidate = binary_search_leftmost(
    ut_hits.yEnd_p(sector_group_offset),
    ut_hit_offsets.sector_group_number_of_hits(sector_group),
    y_track - tol);

  // In case we found a first candidate
  if (first_candidate < static_cast<int>(ut_hit_offsets.sector_group_number_of_hits(sector_group))) {
    // Find the last candidate (y_track + tol)
    number_of_candidates = binary_search_leftmost(
      ut_hits.yBegin_p(sector_group_offset + first_candidate),
      ut_hit_offsets.sector_group_number_of_hits(sector_group) - first_candidate,
      y_track + tol);
  
    first_candidate += sector_group_offset;

    // If there is no last candidate, we are done
    if (number_of_candidates > 0) {
      // Refine the found candidate to fulfill some specific criteria
      tol_refine(
        first_candidate, number_of_candidates, ut_hits, velo_state, invNormFact, xTolNormFact, dx_dy, y_tol, y_tol_slope);
    }
  }

  return std::tuple<int, int> {first_candidate, number_of_candidates};
}
