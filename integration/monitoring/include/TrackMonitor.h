/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "BufferMonitor.h"

struct HostBuffersManager;

struct TrackMonitor : public BufferMonitor {
#ifdef WITH_ROOT
  TrackMonitor(MonitorManager* manager, HostBuffersManager* buffers_manager, int timeStep = 30, int offset = 0) :
    BufferMonitor(manager, "kalmanTracks", timeStep, offset), m_buffers_manager(buffers_manager)
  {
    init();
  };
#else
  TrackMonitor(MonitorManager* manager, HostBuffersManager*, int timeStep = 30, int offset = 0) :
    BufferMonitor(manager, "kalmanTracks", timeStep, offset)
  {
    init();
  };
#endif

  virtual ~TrackMonitor() = default;

  void fill(unsigned i_buf, bool useWallTime = true) override;

private:
  void init();

#ifdef WITH_ROOT
  HostBuffersManager* m_buffers_manager;
#endif
};
