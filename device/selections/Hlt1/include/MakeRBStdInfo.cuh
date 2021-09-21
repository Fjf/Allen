/*****************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "DeviceAlgorithm.cuh"

namespace make_rb_stdinfo {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_stdinfo_bank_size_t, unsigned) host_stdinfo_bank_size;
    DEVICE_INPUT(dev_number_of_active_lines_t, unsigned) dev_number_of_active_lines;
    DEVICE_INPUT(dev_rb_stdinfo_bank_offsets_t, unsigned) dev_rb_stdinfo_bank_offsets;
    DEVICE_INPUT(dev_sel_count_t, unsigned) dev_sel_count;
    DEVICE_INPUT(dev_sel_sv_count_t, unsigned) dev_sel_sv_count;
    DEVICE_INPUT(dev_sel_track_count_t, unsigned) dev_sel_track_count;
    DEVICE_INPUT(dev_sel_list_t, unsigned) dev_sel_list;
    DEVICE_OUTPUT(dev_rb_stdinfo_t, unsigned) dev_rb_stdinfo;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
  };

  __global__ void make_stdinfo_bank(Parameters, const unsigned number_of_events);

  struct make_rb_stdinfo_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
      ArgumentReferences<Parameters> arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const HostBuffers& host_buffers) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      HostBuffers& host_buffers,
      const Allen::Context& context) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{64, 1, 1}}};
  };

} // namespace make_rb_stdinfo