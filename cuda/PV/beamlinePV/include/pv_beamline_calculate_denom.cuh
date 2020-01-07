#pragma once

#include "BeamlinePVConstants.cuh"
#include "Common.h"
#include "DeviceAlgorithm.cuh"
#include "TrackBeamLineVertexFinder.cuh"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "FloatOperations.cuh"
#include <cstdint>

namespace pv_beamline_calculate_denom {
  struct Arguments {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, uint);
    DEVICE_INPUT(dev_atomics_velo_t, uint) dev_atomics_velo;
    DEVICE_INPUT(dev_velo_track_hit_number_t, uint) dev_velo_track_hit_number;
    DEVICE_INPUT(dev_pvtracks_t, PVTrack) dev_pvtracks;
    DEVICE_INPUT(dev_zpeaks_t, float) dev_zpeaks;
    DEVICE_INPUT(dev_number_of_zpeaks_t, uint) dev_number_of_zpeaks;
    DEVICE_OUTPUT(dev_pvtracks_denom_t, float) dev_pvtracks_denom;
  };

  __global__ void pv_beamline_calculate_denom(Arguments arguments);

  template<typename T>
  struct pv_beamline_calculate_denom_t : public DeviceAlgorithm, Arguments {
    constexpr static auto name {"pv_beamline_calculate_denom_t"};
    decltype(global_function(pv_beamline_calculate_denom)) function {pv_beamline_calculate_denom};

    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const HostBuffers& host_buffers) const {
      set_size<dev_pvtracks_denom_t>(arguments, value<host_number_of_reconstructed_velo_tracks_t>(arguments));
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      cudaStream_t& cuda_stream,
      cudaEvent_t& cuda_generic_event) const {
      function.invoke(dim3(value<host_number_of_selected_events_t>(arguments)), block_dimension(), cuda_stream)(
        Arguments{
          offset<dev_atomics_velo_t>(arguments),
          offset<dev_velo_track_hit_number_t>(arguments),
          offset<dev_pvtracks_t>(arguments),
          offset<dev_pvtracks_denom_t>(arguments),
          offset<dev_zpeaks_t>(arguments),
          offset<dev_number_of_zpeaks_t>(arguments)});
    }
  };
}