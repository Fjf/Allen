/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "HostBuffersManager.cuh"
#include "HostBuffers.cuh"
#include "Logger.h"

void HostBuffersManager::init(size_t nBuffers, size_t host_memory_size)
{
  m_host_memory_size = host_memory_size;
  host_buffers.reserve(nBuffers);
  for (size_t i = 0; i < nBuffers; ++i) {
    host_buffers.push_back(new HostBuffers());
    m_persistent_stores.push_back(new Allen::Store::PersistentStore(host_memory_size, 64));
    buffer_statuses.push_back(BufferStatus::Empty);
    empty_buffers.push(i);
  }
}

size_t HostBuffersManager::assignBufferToFill()
{
  if (empty_buffers.empty()) {
    warning_cout << "No empty buffers available" << std::endl;
    warning_cout << "Adding new buffers" << std::endl;
    host_buffers.push_back(new HostBuffers());
    m_persistent_stores.push_back(new Allen::Store::PersistentStore(m_host_memory_size, 64));
    buffer_statuses.push_back(BufferStatus::Filling);
    return host_buffers.size() - 1;
  }

  auto b = empty_buffers.front();
  empty_buffers.pop();

  buffer_statuses[b] = BufferStatus::Filling;
  return b;
}

size_t HostBuffersManager::assignBufferToProcess()
{
  // FIXME required until nvcc supports C++17
  // ideally, this fuction would return a std::optional<size_t>
  if (filled_buffers.empty()) return SIZE_MAX;

  auto b = filled_buffers.front();
  filled_buffers.pop();

  buffer_statuses[b] = BufferStatus::Processing;
  return b;
}

void HostBuffersManager::returnBufferFilled(size_t b)
{
  buffer_statuses[b] = BufferStatus::Filled;
  filled_buffers.push(b);
}

void HostBuffersManager::returnBufferUnfilled(size_t b)
{
  buffer_statuses[b] = BufferStatus::Empty;
  empty_buffers.push(b);
}

void HostBuffersManager::returnBufferProcessed(size_t b)
{
  // buffer must be both processed (monitoring) and written (I/O)
  // if I/O is already finished then mark "empty"
  // otherwise, mark "processed" and wait for I/O
  if (buffer_statuses[b] == BufferStatus::Written) {
    buffer_statuses[b] = BufferStatus::Empty;
    empty_buffers.push(b);
  }
  else {
    buffer_statuses[b] = BufferStatus::Processed;
  }
}

void HostBuffersManager::returnBufferWritten(size_t b)
{
  // buffer must be both processed (monitoring) and written (I/O)
  // if monitoring is already finished then mark "empty"
  // otherwise, mark "written" and wait for I/O
  if (buffer_statuses[b] == BufferStatus::Processed) {
    buffer_statuses[b] = BufferStatus::Empty;
    empty_buffers.push(b);
  }
  else {
    buffer_statuses[b] = BufferStatus::Written;
  }
}

// TODO: Inject a single number onto the store
void HostBuffersManager::writeSingleEventPassthrough(const size_t/*b*/)
{
  // if (b >= m_persistent_stores.size()) {
  //   error_cout << "Buffer index " << b << " is larger than the number of available buffers: " << m_persistent_stores.size()
  //              << std::endl;
  //   return;
  // }
  // auto store = m_persistent_stores[b];

  // "host_init_number_of_events__host_number_of_events_t"

  // buf->host_number_of_events = 1u;
}

void HostBuffersManager::printStatus() const
{
  info_cout << host_buffers.size() << " buffers; " << empty_buffers.size() << " empty; " << filled_buffers.size()
            << " filled." << std::endl;
}
