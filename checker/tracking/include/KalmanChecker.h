#pragma once

#include <functional>
#include <set>
#include <string>
#include <vector>
#include "Logger.h"
#include "MCAssociator.h"
#include "MCEvent.h"
#include "CheckerTypes.h"

#ifdef WITH_ROOT
#include "TTree.h"
#include "TFile.h"
#endif

void checkKalmanTracks(
  const uint start_event_offset,
  const std::vector<Checker::Tracks>& tracks,
  const MCEvents selected_mc_events);
