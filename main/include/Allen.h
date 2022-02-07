/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <map>
#include <string>

#include <ZeroMQ/IZeroMQSvc.h>
#include <Dumpers/IUpdater.h>
#include "InputProvider.h"

struct Constants;

void register_consumers(Allen::NonEventData::IUpdater* updater, Constants& constants);

int allen(
  std::map<std::string, std::string> options,
  Allen::NonEventData::IUpdater* updater,
  IInputProvider* input_provider,
  IZeroMQSvc* zmqSvc,
  std::string_view control_connection
  );

namespace {
  constexpr size_t n_write = 1;
  constexpr size_t n_input = 1;
  constexpr size_t n_io = n_input + n_write;
  constexpr size_t n_mon = 1;
  constexpr size_t max_stream_threads = 1024;
} // namespace
