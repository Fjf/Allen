#pragma once

#include <cstdint>
#include <cfloat>
#include "ClusteringDefinitions.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "DeviceAlgorithm.cuh"

namespace velo_search_by_triplet {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    HOST_INPUT(host_total_number_of_velo_clusters_t, uint);
    DEVICE_INPUT(dev_sorted_velo_cluster_container_t, char) dev_sorted_velo_cluster_container;
    DEVICE_INPUT(dev_offsets_estimated_input_size_t, uint) dev_offsets_estimated_input_size;
    DEVICE_INPUT(dev_module_cluster_num_t, uint) dev_module_cluster_num;
    DEVICE_INPUT(dev_hit_phi_t, int16_t) dev_hit_phi;
    DEVICE_OUTPUT(dev_tracks_t, Velo::TrackHits) dev_tracks;
    DEVICE_OUTPUT(dev_tracklets_t, Velo::TrackletHits) dev_tracklets;
    DEVICE_OUTPUT(dev_tracks_to_follow_t, uint) dev_tracks_to_follow;
    DEVICE_OUTPUT(dev_three_hit_tracks_t, Velo::TrackletHits) dev_three_hit_tracks;
    DEVICE_OUTPUT(dev_hit_used_t, bool) dev_hit_used;
    DEVICE_OUTPUT(dev_atomics_velo_t, uint) dev_atomics_velo;
    DEVICE_OUTPUT(dev_rel_indices_t, unsigned short) dev_rel_indices;
    DEVICE_OUTPUT(dev_number_of_velo_tracks_t, uint) dev_number_of_velo_tracks;

    // Forward tolerance in phi
    PROPERTY(forward_phi_tolerance_t, float, "forward_phi_tolerance", "forwarding tolerance") forward_phi_tolerance;
    PROPERTY(seeding_phi_tolerance_t, float, "seeding_phi_tolerance", "seeding tolerance") seeding_phi_tolerance;

    // Max scatter for forming triplets (seeding) and forwarding
    PROPERTY(max_scatter_forwarding_t, float, "max_scatter_forwarding", "scatter forwarding")
    max_scatter_forwarding;
    PROPERTY(max_scatter_seeding_t, float, "max_scatter_seeding", "scatter seeding") max_scatter_seeding;

    // Maximum number of skipped modules allowed for a track
    // before storing it
    PROPERTY(max_skipped_modules_t, uint, "max_skipped_modules", "skipped modules") max_skipped_modules;

    // Block dimensions of kernel
    PROPERTY(block_dim_t, DeviceDimensions, "block_dim", "block dimensions");
  };

  __global__ void velo_search_by_triplet(Parameters, const VeloGeometry*);

  template<typename T, char... S>
  struct velo_search_by_triplet_t : public DeviceAlgorithm, Parameters {
    constexpr static auto name = Name<S...>::s;
    decltype(global_function(velo_search_by_triplet)) function {velo_search_by_triplet};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const
    {
      set_size<dev_tracks_t>(
        arguments, value<host_number_of_selected_events_t>(arguments) * Velo::Constants::max_tracks);
      set_size<dev_tracklets_t>(
        arguments, value<host_number_of_selected_events_t>(arguments) * Velo::Constants::max_tracks_to_follow);
      set_size<dev_tracks_to_follow_t>(
        arguments, value<host_number_of_selected_events_t>(arguments) * Velo::Constants::max_tracks_to_follow);
      set_size<dev_three_hit_tracks_t>(
        arguments, value<host_number_of_selected_events_t>(arguments) * Velo::Constants::max_three_hit_tracks);
      set_size<dev_hit_used_t>(arguments, value<host_total_number_of_velo_clusters_t>(arguments));
      set_size<dev_atomics_velo_t>(arguments, value<host_number_of_selected_events_t>(arguments) * Velo::num_atomics);
      set_size<dev_number_of_velo_tracks_t>(arguments, value<host_number_of_selected_events_t>(arguments));
      set_size<dev_rel_indices_t>(
        arguments, value<host_number_of_selected_events_t>(arguments) * Velo::Constants::max_numhits_in_module);
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions&,
      const Constants& constants,
      HostBuffers&,
      cudaStream_t& cuda_stream,
      cudaEvent_t&) const
    {
      initialize<dev_atomics_velo_t>(arguments, 0, cuda_stream);
      initialize<dev_hit_used_t>(arguments, 0, cuda_stream);
      initialize<dev_number_of_velo_tracks_t>(arguments, 0, cuda_stream);

      function(dim3(value<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(
        Parameters {begin<dev_sorted_velo_cluster_container_t>(arguments),
                    begin<dev_offsets_estimated_input_size_t>(arguments),
                    begin<dev_module_cluster_num_t>(arguments),
                    begin<dev_hit_phi_t>(arguments),
                    begin<dev_tracks_t>(arguments),
                    begin<dev_tracklets_t>(arguments),
                    begin<dev_tracks_to_follow_t>(arguments),
                    begin<dev_three_hit_tracks_t>(arguments),
                    begin<dev_hit_used_t>(arguments),
                    begin<dev_atomics_velo_t>(arguments),
                    begin<dev_rel_indices_t>(arguments),
                    begin<dev_number_of_velo_tracks_t>(arguments),
                    property<forward_phi_tolerance_t>(),
                    property<seeding_phi_tolerance_t>(),
                    property<max_scatter_forwarding_t>(),
                    property<max_scatter_seeding_t>(),
                    property<max_skipped_modules_t>()},
        constants.dev_velo_geometry);
    }

  private:
    Property<forward_phi_tolerance_t> m_forward_tol {this, 0.052f};
    Property<seeding_phi_tolerance_t> m_seeding_tol {this, 0.052f};
    Property<max_scatter_forwarding_t> m_scat {this, 0.1f};
    Property<max_scatter_seeding_t> m_seed {this, 0.1f};
    Property<max_skipped_modules_t> m_skip {this, 1u};
    Property<block_dim_t> m_block_dim {this, {{32, 1, 1}}};
  };
} // namespace velo_search_by_triplet
