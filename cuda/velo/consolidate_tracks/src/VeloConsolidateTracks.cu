#include "VeloConsolidateTracks.cuh"

using namespace velo_consolidate_tracks;

/**
 * @brief Calculates the parameters according to a root means square fit
 */
__device__ VeloState means_square_fit(Velo::Consolidated::Hits& consolidated_hits, const uint number_of_hits) {
  VeloState state;

  // Fit parameters
  float s0, sx, sz, sxz, sz2;
  float u0, uy, uz, uyz, uz2;
  s0 = sx = sz = sxz = sz2 = 0.0f;
  u0 = uy = uz = uyz = uz2 = 0.0f;

  // Iterate over hits
  for (unsigned short h = 0; h < number_of_hits; ++h) {
    const auto x = consolidated_hits.x[h];
    const auto y = consolidated_hits.y[h];
    const auto z = consolidated_hits.z[h];

    const auto wx = Velo::Tracking::param_w;
    const auto wx_t_x = wx * x;
    const auto wx_t_z = wx * z;
    s0 += wx;
    sx += wx_t_x;
    sz += wx_t_z;
    sxz += wx_t_x * z;
    sz2 += wx_t_z * z;

    const auto wy = Velo::Tracking::param_w;
    const auto wy_t_y = wy * y;
    const auto wy_t_z = wy * z;
    u0 += wy;
    uy += wy_t_y;
    uz += wy_t_z;
    uyz += wy_t_y * z;
    uz2 += wy_t_z * z;
  }

  // Calculate tx, ty and backward
  const auto dens = 1.0f / (sz2 * s0 - sz * sz);
  state.tx = (sxz * s0 - sx * sz) * dens;
  state.x = (sx * sz2 - sxz * sz) * dens;

  const auto denu = 1.0f / (uz2 * u0 - uz * uz);
  state.ty = (uyz * u0 - uy * uz) * denu;
  state.y = (uy * uz2 - uyz * uz) * denu;

  state.z = -(state.x * state.tx + state.y * state.ty) / (state.tx * state.tx + state.ty * state.ty);
  state.backward = state.z > consolidated_hits.z[0];

  state.x = state.x + state.tx * state.z;
  state.y = state.y + state.ty * state.z;

  return state;
}

template<typename T, typename F>
__device__ void populate(const Velo::TrackHits* track, const uint number_of_hits, T* __restrict__ a, const F& fn) {
  for (int i = 0; i < number_of_hits; ++i) {
    const auto hit_index = track->hits[i];
    a[i] = fn(hit_index);
  }
}

__global__ void velo_consolidate_tracks::velo_consolidate_tracks(Arguments arguments) {
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;

  const Velo::TrackHits* event_tracks = arguments.dev_tracks + event_number * Velo::Constants::max_tracks;
  const Velo::TrackletHits* three_hit_tracks =
    arguments.dev_three_hit_tracks_output + event_number * Velo::Constants::max_tracks;

  // Consolidated datatypes
  const Velo::Consolidated::Tracks velo_tracks {
    arguments.dev_atomics_velo, arguments.dev_velo_track_hit_number, event_number, number_of_events};
  Velo::Consolidated::States velo_states {const_cast<char*>(arguments.dev_velo_states.get()),
                                          velo_tracks.total_number_of_tracks()};

  const uint event_number_of_tracks = velo_tracks.number_of_tracks(event_number);
  const uint event_tracks_offset = velo_tracks.tracks_offset(event_number);

  const auto event_number_of_three_hit_tracks_filtered =
    arguments.dev_offsets_number_of_three_hit_tracks_filtered[event_number + 1] -
    arguments.dev_offsets_number_of_three_hit_tracks_filtered[event_number];
  const auto event_number_of_tracks_in_main_track_container =
    event_number_of_tracks - event_number_of_three_hit_tracks_filtered;

  // Pointers to data within event
  const uint total_estimated_number_of_clusters =
    arguments.dev_offsets_estimated_input_size[Velo::Constants::n_modules * number_of_events];
  const uint* module_hitStarts = arguments.dev_offsets_estimated_input_size + event_number * Velo::Constants::n_modules;
  const uint hit_offset = module_hitStarts[0];

  // TODO: Offset'ed container
  const auto velo_cluster_container = Velo::Clusters<const uint> {
    arguments.dev_sorted_velo_cluster_container.get() + hit_offset, total_estimated_number_of_clusters};

  for (uint i = threadIdx.x; i < event_number_of_tracks; i += blockDim.x) {
    Velo::Consolidated::Hits consolidated_hits = velo_tracks.get_hits(arguments.dev_velo_track_hits, i);

    Velo::TrackHits* track;
    uint number_of_hits;

    if (i < event_number_of_tracks_in_main_track_container) {
      track = const_cast<Velo::TrackHits*>(event_tracks) + i;
      number_of_hits = track->hitsNum;
    }
    else {
      track = const_cast<Velo::TrackHits*>(reinterpret_cast<const Velo::TrackHits*>(
        three_hit_tracks + i - event_number_of_tracks_in_main_track_container));
      number_of_hits = 3;
    }

    populate<float>(track, number_of_hits, consolidated_hits.x, [&velo_cluster_container](const uint hit_index) {
      return velo_cluster_container.x(hit_index);
    });
    populate<float>(track, number_of_hits, consolidated_hits.y, [&velo_cluster_container](const uint hit_index) {
      return velo_cluster_container.y(hit_index);
    });
    populate<float>(track, number_of_hits, consolidated_hits.z, [&velo_cluster_container](const uint hit_index) {
      return velo_cluster_container.z(hit_index);
    });
    populate<uint32_t>(
      track, number_of_hits, consolidated_hits.LHCbID, [&velo_cluster_container](const uint hit_index) {
        return velo_cluster_container.id(hit_index);
      });

    // Calculate and store fit in consolidated container
    VeloState beam_state = means_square_fit(consolidated_hits, number_of_hits);
    velo_states.set(event_tracks_offset + i, beam_state);
  }
}