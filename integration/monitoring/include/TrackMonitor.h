#pragma once

#include "BufferMonitor.h"

struct HostBuffersManager;

struct TrackMonitor : public BufferMonitor {
  TrackMonitor(HostBuffersManager* buffers_manager, int timeStep = 30, int offset = 0) :
    BufferMonitor("kalmanTracks", timeStep, offset), m_buffers_manager(buffers_manager)
  {
    init();
  };

  virtual ~TrackMonitor() = default;

  void fill(uint i_buf, bool useWallTime = true) override;

private:
  void init();

  HostBuffersManager* m_buffers_manager;
};
