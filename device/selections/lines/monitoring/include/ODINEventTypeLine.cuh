/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#pragma once

#include "SelectionAlgorithm.cuh"
#include "ODINLine.cuh"
#include "Event/ODIN.h"

namespace odin_event_type_line {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_mep_layout_t, unsigned) dev_mep_layout;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_odin_raw_input_t, char) dev_odin_raw_input;
    DEVICE_INPUT(dev_odin_raw_input_offsets_t, unsigned) dev_odin_raw_input_offsets;
    DEVICE_OUTPUT(dev_decisions_t, bool) dev_decisions;
    DEVICE_OUTPUT(dev_decisions_offsets_t, unsigned) dev_decisions_offsets;
    HOST_OUTPUT(host_post_scaler_t, float) host_post_scaler;
    HOST_OUTPUT(host_post_scaler_hash_t, uint32_t) host_post_scaler_hash;
    PROPERTY(pre_scaler_t, "pre_scaler", "Pre-scaling factor", float) pre_scaler;
    PROPERTY(post_scaler_t, "post_scaler", "Post-scaling factor", float) post_scaler;
    PROPERTY(pre_scaler_hash_string_t, "pre_scaler_hash_string", "Pre-scaling hash string", std::string);
    PROPERTY(post_scaler_hash_string_t, "post_scaler_hash_string", "Post-scaling hash string", std::string);
    PROPERTY(odin_event_type_t, "odin_event_type", "ODIN event type", unsigned) odin_event_type;
  };

  struct odin_event_type_line_t : public SelectionAlgorithm, Parameters, ODINLine<odin_event_type_line_t, Parameters> {
    __device__ static bool select(const Parameters& parameters, std::tuple<const unsigned*> input);

  private:
    Property<pre_scaler_t> m_pre_scaler {this, 1.f};
    Property<post_scaler_t> m_post_scaler {this, 0.5f};
    Property<pre_scaler_hash_string_t> m_pre_scaler_hash_string {this, ""};
    Property<post_scaler_hash_string_t> m_post_scaler_hash_string {this, ""};
    Property<odin_event_type_t> m_odin_event_type {this, LHCb::ODIN::Lumi};
  };
} // namespace odin_event_type_line
