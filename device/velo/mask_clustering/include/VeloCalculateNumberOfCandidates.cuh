#pragma once

#include "DeviceAlgorithm.cuh"
#include "ClusteringDefinitions.cuh"
#include <boost/pfr/precise/core.hpp>

namespace velo_calculate_number_of_candidates {
  struct Parameters {
    HOST_INPUT(host_number_of_selected_events_t, uint);
    DEVICE_INPUT(dev_event_list_t, uint) dev_event_list;
    DEVICE_INPUT(dev_velo_raw_input_t, char) dev_velo_raw_input;
    DEVICE_INPUT(dev_velo_raw_input_offsets_t, uint) dev_velo_raw_input_offsets;
    DEVICE_OUTPUT(dev_number_of_candidates_t, uint) dev_number_of_candidates;
    PROPERTY(block_dim_x_t, uint, "block_dim_x", "block dimension X");
  };

  // Global function
  __global__ void velo_calculate_number_of_candidates(Parameters parameters, const uint number_of_events);
  __global__ void velo_calculate_number_of_candidates_mep(Parameters parameters, const uint number_of_events);

  // Algorithm
  template<typename T, char... S>
  struct velo_calculate_number_of_candidates_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentRefManager<T> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const
    {
      boost::pfr::tuple_element<0, Parameters>::type a;

      if (logger::verbosity() >= logger::debug) {
        debug_cout << "# of events = " << value<host_number_of_selected_events_t>(arguments) << std::endl;
      }
      set_size<dev_number_of_candidates_t>(arguments, value<host_number_of_selected_events_t>(arguments));
    }

    void operator()(
      const ArgumentRefManager<T>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants&,
      HostBuffers&,
      cudaStream_t& cuda_stream,
      cudaEvent_t&) const
    {
      // Enough blocks to cover all events
      const auto grid_size = dim3(
        (value<host_number_of_selected_events_t>(arguments) + property<block_dim_x_t>() - 1) / property<block_dim_x_t>());

      // Invoke kernel
      const auto parameters = Parameters {begin<dev_event_list_t>(arguments),
                                          begin<dev_velo_raw_input_t>(arguments),
                                          begin<dev_velo_raw_input_offsets_t>(arguments),
                                          begin<dev_number_of_candidates_t>(arguments)};

      using function_t = decltype(global_function(velo_calculate_number_of_candidates));
      function_t function = runtime_options.mep_layout ? global_function(velo_calculate_number_of_candidates_mep) : global_function(velo_calculate_number_of_candidates);
      function(grid_size, dim3(property<block_dim_x_t>().get()), cuda_stream)(
        parameters, value<host_number_of_selected_events_t>(arguments));
    }

  private:
    Property<block_dim_x_t> m_block_dim_x {this, 256};
  };
} // namespace velo_calculate_number_of_candidates
