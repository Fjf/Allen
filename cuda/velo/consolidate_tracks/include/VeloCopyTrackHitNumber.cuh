#include "VeloEventModel.cuh"
#include "DeviceAlgorithm.cuh"

namespace velo_copy_track_hit_number {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, uint);
    HOST_INPUT(host_number_of_three_hit_tracks_filtered_t, uint);
    DEVICE_INPUT(dev_tracks_t, Velo::TrackHits) dev_tracks;
    DEVICE_INPUT(dev_offsets_velo_tracks_t, uint) dev_offsets_velo_tracks;
    DEVICE_INPUT(dev_offsets_number_of_three_hit_tracks_filtered_t, uint) dev_offsets_number_of_three_hit_tracks_filtered;
    DEVICE_OUTPUT(dev_velo_track_hit_number_t, uint) dev_velo_track_hit_number;
    DEVICE_OUTPUT(dev_offsets_all_velo_tracks_t, uint) dev_offsets_all_velo_tracks;
  };

  __global__ void velo_copy_track_hit_number(Parameters);

  template<typename T, char... S>
  struct velo_copy_track_hit_number_t : public DeviceAlgorithm, Parameters {
    constexpr static auto name = Name<S...>::s;
    decltype(global_function(velo_copy_track_hit_number)) function {velo_copy_track_hit_number};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const HostBuffers& host_buffers) const {
      set_size<dev_velo_track_hit_number_t>(arguments, value<host_number_of_reconstructed_velo_tracks_t>(arguments)
        + value<host_number_of_three_hit_tracks_filtered_t>(arguments));

      // Note: Size is "+ 1" due to it storing offsets.
      set_size<dev_offsets_all_velo_tracks_t>(arguments, value<host_number_of_selected_events_t>(arguments) + 1);
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      cudaStream_t& cuda_stream,
      cudaEvent_t& cuda_generic_event) const {
      cudaCheck(cudaMemsetAsync(
        offset<dev_offsets_all_velo_tracks_t>(arguments),
        0,
        sizeof(uint), // Note: Only the first element needs to be initialized here.
        cuda_stream));

      function(dim3(value<host_number_of_selected_events_t>(arguments)), block_dimension(), cuda_stream)(
        Parameters{
          offset<dev_tracks_t>(arguments),
          offset<dev_offsets_velo_tracks_t>(arguments),
          offset<dev_offsets_number_of_three_hit_tracks_filtered_t>(arguments),
          offset<dev_velo_track_hit_number_t>(arguments),
          offset<dev_offsets_all_velo_tracks_t>(arguments)
        });
    }
  };
} // namespace velo_copy_track_hit_number