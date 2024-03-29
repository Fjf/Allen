/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "VeloConsolidateTracks.cuh"

INSTANTIATE_ALGORITHM(velo_consolidate_tracks::velo_consolidate_tracks_t)

/**
 * @brief Creates VELO views for hits, track, tracks and multieventtracks.
 */
__global__ void create_velo_views(velo_consolidate_tracks::Parameters parameters)
{
  const unsigned number_of_events = parameters.dev_number_of_events[0];
  const unsigned event_number = blockIdx.x;

  const auto event_tracks_offset = parameters.dev_offsets_all_velo_tracks[event_number];
  const auto event_number_of_tracks = parameters.dev_offsets_all_velo_tracks[event_number + 1] - event_tracks_offset;

  for (unsigned track_index = threadIdx.x; track_index < event_number_of_tracks; track_index += blockDim.x) {
    new (parameters.dev_velo_track_view + event_tracks_offset + track_index)
      Allen::Views::Velo::Consolidated::Track {parameters.dev_velo_hits_view,
                                               parameters.dev_offsets_all_velo_tracks,
                                               parameters.dev_offsets_velo_track_hit_number,
                                               track_index,
                                               event_number};
  }

  if (threadIdx.x == 0) {
    new (parameters.dev_velo_hits_view + event_number)
      Allen::Views::Velo::Consolidated::Hits {parameters.dev_velo_track_hits,
                                              parameters.dev_offsets_all_velo_tracks,
                                              parameters.dev_offsets_velo_track_hit_number,
                                              event_number,
                                              number_of_events};

    new (parameters.dev_velo_tracks_view + event_number) Allen::Views::Velo::Consolidated::Tracks {
      parameters.dev_velo_track_view, parameters.dev_offsets_all_velo_tracks, event_number};
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    new (parameters.dev_velo_multi_event_tracks_view)
      Allen::Views::Velo::Consolidated::MultiEventTracks {parameters.dev_velo_tracks_view, number_of_events};

    parameters.dev_imec_velo_tracks[0] = parameters.dev_velo_multi_event_tracks_view;
  }
}

void velo_consolidate_tracks::velo_consolidate_tracks_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const auto total_number_of_velo_tracks = first<host_number_of_reconstructed_velo_tracks_t>(arguments) +
                                           first<host_number_of_three_hit_tracks_filtered_t>(arguments);

  set_size<dev_velo_track_hits_t>(
    arguments, first<host_accumulated_number_of_hits_in_velo_tracks_t>(arguments) * Velo::Clusters::element_size);
  set_size<dev_accepted_velo_tracks_t>(arguments, total_number_of_velo_tracks);
  set_size<dev_velo_hits_view_t>(arguments, first<host_number_of_events_t>(arguments));
  set_size<dev_velo_track_view_t>(arguments, total_number_of_velo_tracks);
  set_size<dev_velo_tracks_view_t>(arguments, first<host_number_of_events_t>(arguments));
  set_size<dev_velo_multi_event_tracks_view_t>(arguments, 1);
  set_size<dev_imec_velo_tracks_t>(arguments, 1);
  set_size<dev_imec_velo_tracks_t>(arguments, 1);
}

void velo_consolidate_tracks::velo_consolidate_tracks_t::init()
{
#ifndef ALLEN_STANDALONE
  m_velo_tracks = new Gaudi::Accumulators::Counter<>(this, "n_velo_tracks");

  histogram_n_velo_tracks = new gaudi_monitoring::Lockable_Histogram<> {
    {this, "n_velo_tracks_event", "n_velo_tracks_event", {100, 0, 500}}, {}};
#endif
}

void velo_consolidate_tracks::velo_consolidate_tracks_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  // Set all found tracks to accepted
  Allen::memset_async<dev_accepted_velo_tracks_t>(arguments, 1, context);

  // Initialize dev_velo_multi_event_tracks_view_t to avoid invalid std::function destructor
  Allen::memset_async<dev_velo_multi_event_tracks_view_t>(arguments, 0, context);
  Allen::memset_async<dev_velo_tracks_view_t>(arguments, 0, context);

  auto dev_number_of_tracks_histo = make_device_buffer<unsigned>(arguments, 100u);
  auto dev_tracks_counter = make_device_buffer<unsigned>(arguments, 1u);
  Allen::memset_async(
    dev_number_of_tracks_histo.data(), 0, dev_number_of_tracks_histo.size() * sizeof(unsigned), context);
  Allen::memset_async(dev_tracks_counter.data(), 0, dev_tracks_counter.size() * sizeof(unsigned), context);

  global_function(velo_consolidate_tracks)(size<dev_event_list_t>(arguments), property<block_dim_t>(), context)(
    arguments, dev_number_of_tracks_histo.get(), dev_tracks_counter.get());

  global_function(create_velo_views)(first<host_number_of_events_t>(arguments), 256, context)(arguments);

#ifndef ALLEN_STANDALONE
  // Monitoring
  gaudi_monitoring::fill(
    arguments,
    context,
    std::tuple {std::tuple {dev_number_of_tracks_histo.get(), histogram_n_velo_tracks, 0, 500},
                std::tuple {dev_tracks_counter.get(), m_velo_tracks}});
#endif
}

template<typename F>
__device__ void populate(const Velo::TrackHits* track, const unsigned number_of_hits, const F& assign)
{
  for (unsigned i = 0; i < number_of_hits; ++i) {
    const auto hit_index = track->hits[i];
    assign(i, hit_index);
  }
}

__global__ void velo_consolidate_tracks::velo_consolidate_tracks(
  velo_consolidate_tracks::Parameters parameters,
  gsl::span<unsigned> dev_number_of_tracks_histo,
  gsl::span<unsigned> dev_tracks_counter)
{
  const unsigned number_of_events = parameters.dev_number_of_events[0];
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  const auto tracks_offset = Velo::track_offset(parameters.dev_offsets_estimated_input_size, event_number);
  const Velo::TrackHits* event_tracks = parameters.dev_tracks + tracks_offset;
  const Velo::TrackletHits* three_hit_tracks = parameters.dev_three_hit_tracks_output + tracks_offset;

  Velo::Consolidated::Tracks velo_tracks {parameters.dev_offsets_all_velo_tracks,
                                          parameters.dev_offsets_velo_track_hit_number,
                                          event_number,
                                          number_of_events};
  const unsigned event_total_number_of_tracks = velo_tracks.number_of_tracks(event_number);
  const auto event_number_of_three_hit_tracks_filtered =
    parameters.dev_offsets_number_of_three_hit_tracks_filtered[event_number + 1] -
    parameters.dev_offsets_number_of_three_hit_tracks_filtered[event_number];
  const auto event_number_of_tracks_in_main_track_container =
    event_total_number_of_tracks - event_number_of_three_hit_tracks_filtered;

  if (event_total_number_of_tracks < 500) {
    unsigned bin = std::floor(event_total_number_of_tracks / 5);
    dev_number_of_tracks_histo[bin]++;
  }
  dev_tracks_counter[0] += event_total_number_of_tracks;

  // Pointers to data within event
  const unsigned total_estimated_number_of_clusters =
    parameters.dev_offsets_estimated_input_size[Velo::Constants::n_module_pairs * number_of_events];
  const unsigned* module_hitStarts =
    parameters.dev_offsets_estimated_input_size + event_number * Velo::Constants::n_module_pairs;
  const unsigned hit_offset = module_hitStarts[0];

  // Offset'ed container
  const auto velo_cluster_container =
    Velo::ConstClusters {parameters.dev_sorted_velo_cluster_container, total_estimated_number_of_clusters, hit_offset};

  for (unsigned i = threadIdx.x; i < event_total_number_of_tracks; i += blockDim.x) {
    __syncthreads();

    Velo::Consolidated::Hits consolidated_hits = velo_tracks.get_hits(parameters.dev_velo_track_hits, i);

    Velo::TrackHits* track;
    unsigned number_of_hits;

    if (i < event_number_of_tracks_in_main_track_container) {
      track = const_cast<Velo::TrackHits*>(event_tracks) + i;
      number_of_hits = track->hitsNum;
    }
    else {
      track = const_cast<Velo::TrackHits*>(reinterpret_cast<const Velo::TrackHits*>(
        three_hit_tracks + i - event_number_of_tracks_in_main_track_container));
      number_of_hits = 3;
    }

    // Populate hits in a coalesced manner, taking into account
    // the underlying container.
    populate(
      track, number_of_hits, [&velo_cluster_container, &consolidated_hits](const unsigned i, const unsigned hit_index) {
        consolidated_hits.set_x(i, velo_cluster_container.x(hit_index));
        consolidated_hits.set_y(i, velo_cluster_container.y(hit_index));
        consolidated_hits.set_z(i, velo_cluster_container.z(hit_index));
      });

    populate(
      track, number_of_hits, [&velo_cluster_container, &consolidated_hits](const unsigned i, const unsigned hit_index) {
        consolidated_hits.set_id(i, velo_cluster_container.id(hit_index));
      });
  }
}

void velo_consolidate_tracks::lhcb_id_container_checks::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  const unsigned number_of_events = first<Parameters::host_number_of_events_t>(arguments);

  // Create velo hits views
  const auto dev_velo_track_hits = make_host_buffer<Parameters::dev_velo_track_hits_t>(arguments, context);
  const auto dev_offsets_all_velo_tracks =
    make_host_buffer<Parameters::dev_offsets_all_velo_tracks_t>(arguments, context);
  const auto dev_offsets_velo_track_hit_number =
    make_host_buffer<Parameters::dev_offsets_velo_track_hit_number_t>(arguments, context);
  std::vector<Allen::Views::Velo::Consolidated::Hits> velo_hits_view;
  for (unsigned event_number = 0; event_number < number_of_events; ++event_number) {
    velo_hits_view.emplace_back(Allen::Views::Velo::Consolidated::Hits {dev_velo_track_hits.data(),
                                                                        dev_offsets_all_velo_tracks.data(),
                                                                        dev_offsets_velo_track_hit_number.data(),
                                                                        event_number,
                                                                        number_of_events});
  }

  // Create velo track views
  std::vector<Allen::Views::Velo::Consolidated::Track> velo_track_view;
  for (unsigned event_number = 0; event_number < number_of_events; ++event_number) {
    const auto event_tracks_offset = dev_offsets_all_velo_tracks[event_number];
    const auto event_number_of_tracks = dev_offsets_all_velo_tracks[event_number + 1] - event_tracks_offset;

    for (unsigned track_index = 0; track_index < event_number_of_tracks; ++track_index) {
      velo_track_view.emplace_back(Allen::Views::Velo::Consolidated::Track {velo_hits_view.data(),
                                                                            dev_offsets_all_velo_tracks.data(),
                                                                            dev_offsets_velo_track_hit_number.data(),
                                                                            track_index,
                                                                            event_number});
    }
  }

  // Create velo tracks view
  std::vector<Allen::Views::Velo::Consolidated::Tracks> velo_tracks_view;
  for (unsigned event_number = 0; event_number < number_of_events; ++event_number) {
    velo_tracks_view.emplace_back(Allen::Views::Velo::Consolidated::Tracks {
      velo_track_view.data(), dev_offsets_all_velo_tracks.data(), event_number});
  }

  // Create velo multi event tracks view
  std::vector<Allen::Views::Velo::Consolidated::MultiEventTracks> velo_multi_event_tracks_view;
  velo_multi_event_tracks_view.emplace_back(
    Allen::Views::Velo::Consolidated::MultiEventTracks {velo_tracks_view.data(), number_of_events});
  const Allen::Views::Velo::Consolidated::MultiEventTracks* multiev_id_cont = velo_multi_event_tracks_view.data();

  // Conditions to check
  bool equal_number_of_tracks_and_sequences = true;
  bool lhcb_ids_never_zero = true;
  bool lhcb_ids_have_velo_preamble = true;

  for (unsigned event_number = 0; event_number < velo_multi_event_tracks_view[0].number_of_events(); ++event_number) {
    const auto& tracks = velo_multi_event_tracks_view[0].container(event_number);
    const auto& id_cont = multiev_id_cont->container(event_number);
    equal_number_of_tracks_and_sequences &= tracks.size() == id_cont.number_of_id_sequences();

    for (unsigned sequence_index = 0; sequence_index < tracks.size(); ++sequence_index) {
      const auto& track = tracks.track(sequence_index);
      for (unsigned id_index = 0; id_index < track.number_of_ids(); ++id_index) {
        lhcb_ids_never_zero &= track.id(id_index) != 0;
        lhcb_ids_have_velo_preamble &= lhcb_id::is_velo(track.id(id_index));
      }
    }
  }

  require(
    equal_number_of_tracks_and_sequences,
    "Require that the number of tracks equals the number of LHCb ID sequences for all events");
  require(lhcb_ids_never_zero, "Require that LHCb IDs are never zero");
  require(lhcb_ids_have_velo_preamble, "Require that LHCb IDs of VELO container have VELO preamble");
}
