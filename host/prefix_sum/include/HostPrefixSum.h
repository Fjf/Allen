/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#pragma once

#include "BackendCommon.h"
#include "AlgorithmTypes.cuh"
#include "GenericContainerContracts.h"

namespace host_prefix_sum {
  struct Parameters {
    HOST_OUTPUT(host_total_sum_holder_t, unsigned) host_total_sum_holder;
    DEVICE_INPUT(dev_input_buffer_t, unsigned) dev_input_buffer;
    HOST_OUTPUT(host_output_buffer_t, unsigned) host_output_buffer;
    DEVICE_OUTPUT(dev_output_buffer_t, unsigned) dev_output_buffer;
  };

  /**
   * @brief Implementation of prefix sum on the host.
   */
  void host_prefix_sum_impl(
    unsigned* host_prefix_sum_buffer,
    const size_t input_number_of_elements,
    unsigned* host_total_sum_holder = nullptr);

  struct host_prefix_sum_t : public HostAlgorithm, Parameters {
    using contracts = std::tuple<
      Allen::contract::is_monotonically_increasing<host_output_buffer_t, Parameters, Allen::contract::Postcondition>,
      Allen::contract::
        are_equal<host_output_buffer_t, dev_output_buffer_t, Parameters, Allen::contract::Postcondition>>;

    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions&,
      const Constants&,
      const HostBuffers&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      HostBuffers& host_buffers,
      const Allen::Context&) const;
  };
} // namespace host_prefix_sum
