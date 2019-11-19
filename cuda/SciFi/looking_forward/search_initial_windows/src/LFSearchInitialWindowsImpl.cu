#include "LFSearchInitialWindowsImpl.cuh"
#include "LookingForwardConstants.cuh"
#include "LookingForwardTools.cuh"
#include "BinarySearch.cuh"

__device__ void lf_search_initial_windows_impl(
  const SciFi::Hits& scifi_hits,
  const SciFi::HitCount& scifi_hit_count,
  const MiniState& UT_state,
  const LookingForward::Constants* looking_forward_constants,
  const float qop,
  const int side,
  int* initial_windows,
  const int number_of_tracks,
  const uint event_offset,
  bool* dev_process_track,
  const uint ut_track_index)
{
  int iZoneStartingPoint = (side > 0) ? LookingForward::number_of_x_layers : 0;
  uint16_t sizes = 0;

  for (int i = 0; i < LookingForward::number_of_x_layers; i++) {
    const auto iZone = iZoneStartingPoint + i;

    const auto stateInZone = LookingForward::propagate_state_from_velo_multi_par(
      UT_state, qop, looking_forward_constants->x_layers[i], looking_forward_constants);
    const float xInZone = stateInZone.x;

    const float xTol = 1.5f * (100.f + 1.4e6f * fabsf(qop));
    const float xMin = xInZone - xTol;
    const float xMax = xInZone + xTol;

    // const float xTol = 150.f + 2e6f * fabsf(qop);
    // const float xMin = xInZone - xTol - 100.f * signbit(qop);
    // const float xMax = xInZone + xTol + 100.f * (signbit(qop) ^ 0x01);

    // Get the hits within the bounds
    const int x_zone_offset_begin = scifi_hit_count.zone_offset(looking_forward_constants->xZones[iZone]);
    const int x_zone_size = scifi_hit_count.zone_number_of_hits(looking_forward_constants->xZones[iZone]);
    const int hits_within_bounds_start = binary_search_leftmost(scifi_hits.x0 + x_zone_offset_begin, x_zone_size, xMin);
    const int hits_within_bounds_xInZone = binary_search_leftmost(
      scifi_hits.x0 + x_zone_offset_begin + hits_within_bounds_start, x_zone_size - hits_within_bounds_start, xInZone);
    const int hits_within_bounds_size = binary_search_leftmost(
      scifi_hits.x0 + x_zone_offset_begin + hits_within_bounds_start, x_zone_size - hits_within_bounds_start, xMax);

    // Cap the central windows to a certain size
    const int central_window_begin =
      max(hits_within_bounds_xInZone - LookingForward::extreme_layers_window_size / 2, 0);
    const int central_window_size =
      min(central_window_begin + LookingForward::extreme_layers_window_size, hits_within_bounds_size) -
      central_window_begin;

    // Initialize windows
    initial_windows[i * 8 * number_of_tracks] =
      hits_within_bounds_start + x_zone_offset_begin - event_offset + central_window_begin;
    initial_windows[(i * 8 + 1) * number_of_tracks] = central_window_size;

    sizes |= (hits_within_bounds_size > 0) << i;

    // Skip making range but continue if the size is zero
    if (hits_within_bounds_size > 0) {
      // Now match the stereo hits
      const float zZone = looking_forward_constants->Zone_zPos_xlayers[i];
      const float this_uv_z = looking_forward_constants->Zone_zPos_uvlayers[i];
      const float dz = this_uv_z - zZone;
      const float xInUv = LookingForward::linear_propagation(xInZone, stateInZone.tx, dz);
      const float UvCorr =
        LookingForward::y_at_z(stateInZone, this_uv_z) * looking_forward_constants->Zone_dxdy_uvlayers[i % 2];
      const float xInUvCorr = xInUv - UvCorr;
      const float xMinUV = xInUvCorr - 800.f;
      const float xMaxUV = xInUvCorr + 800.f;

      // Get bounds in UV layers
      // do one search on the same side as the x module
      const int uv_zone_offset_begin = scifi_hit_count.zone_offset(looking_forward_constants->uvZones[iZone]);
      const int uv_zone_size = scifi_hit_count.zone_number_of_hits(looking_forward_constants->uvZones[iZone]);
      const int hits_within_uv_bounds =
        binary_search_leftmost(scifi_hits.x0 + uv_zone_offset_begin, uv_zone_size, xMinUV);
      const int hits_within_uv_bounds_size = binary_search_leftmost(
        scifi_hits.x0 + uv_zone_offset_begin + hits_within_uv_bounds, uv_zone_size - hits_within_uv_bounds, xMaxUV);

      initial_windows[(i * 8 + 2) * number_of_tracks] = hits_within_uv_bounds + uv_zone_offset_begin - event_offset;
      initial_windows[(i * 8 + 3) * number_of_tracks] = hits_within_uv_bounds_size;

      sizes |= (hits_within_uv_bounds_size > 0) << (8 + i);
    }
  }

  // Process track if:
  // * It can have a triplet 0,2,4 or 1,3,5
  // * It can have at least one hit in UV layers
  //   (1 or 2) and (5 or 6) and (9 or 10)
  const bool do_process =
    (((sizes & 0x01) && (sizes & 0x04) && (sizes & 0x10)) || ((sizes & 0x02) && (sizes & 0x08) && (sizes & 0x20))) &&
    ((sizes & 0x0100) || (sizes & 0x0200)) && ((sizes & 0x0400) || (sizes & 0x0800)) &&
    ((sizes & 0x1000) || (sizes & 0x2000));

  dev_process_track[ut_track_index] = do_process;
}
