#pragma once

#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "DeviceAlgorithm.cuh"

namespace scifi_direct_decoder_v4 {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    DEVICE_INPUT(dev_scifi_raw_input_t, char) dev_scifi_raw_input;
    DEVICE_INPUT(dev_scifi_raw_input_offsets_t, uint) dev_scifi_raw_input_offsets;
    DEVICE_INPUT(dev_scifi_hit_offsets_t, uint) dev_scifi_hit_count;
    DEVICE_OUTPUT(dev_scifi_hits_t, char) dev_scifi_hits;
    DEVICE_INPUT(dev_event_list_t, uint) dev_event_list;
    PROPERTY(block_dim_t, DeviceDimensions, "block_dim", "block dimensions");
  };

  __global__ void scifi_direct_decoder_v4(Parameters, const char* scifi_geometry);

  __global__ void scifi_direct_decoder_v4_mep(Parameters, const char* scifi_geometry);

  template<typename T, char... S>
  struct scifi_direct_decoder_v4_t : public DeviceAlgorithm, Parameters {
    constexpr static auto name = Name<S...>::s;
    decltype(global_function(scifi_direct_decoder_v4)) function {scifi_direct_decoder_v4};
    decltype(global_function(scifi_direct_decoder_v4_mep)) function_mep {scifi_direct_decoder_v4_mep};

    void set_arguments_size(
      ArgumentRefManager<T>,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const
    {}

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers&,
      cudaStream_t& cuda_stream,
      cudaEvent_t&) const
    {
      const auto parameters = Parameters {begin<dev_scifi_raw_input_t>(arguments),
                                          begin<dev_scifi_raw_input_offsets_t>(arguments),
                                          begin<dev_scifi_hit_offsets_t>(arguments),
                                          begin<dev_scifi_hits_t>(arguments),
                                          begin<dev_event_list_t>(arguments)};

      if (runtime_options.mep_layout) {
        function_mep(dim3(value<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(
          parameters, constants.dev_scifi_geometry);
      }
      else {
        function(dim3(value<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(
          parameters, constants.dev_scifi_geometry);
      }
    }

  private:
    Property<block_dim_t> m_block_dim {this, {{2, 16, 1}}};
  };
} // namespace scifi_direct_decoder_v4
