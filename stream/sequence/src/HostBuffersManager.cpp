/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "HostBuffersManager.cuh"
#include "HostBuffers.cuh"
#include "Logger.h"

void HostBuffersManager::init(size_t nBuffers)
{
  host_buffers.reserve(nBuffers);
  for (size_t i = 0; i < nBuffers; ++i) {
    host_buffers.push_back(new HostBuffers());
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

void HostBuffersManager::writeSingleEventPassthrough(const size_t b)
{
  if (b >= host_buffers.size()) {
    error_cout << "Buffer index " << b << " is larger than the number of available buffers: " << host_buffers.size()
               << std::endl;
    return;
  }
  auto buf = host_buffers[b];

  buf->host_number_of_events = 1u;
  buf->host_passing_event_list[0] = true;
}

std::tuple<
  gsl::span<bool>,
  gsl::span<uint32_t>,
  gsl::span<uint32_t>,
  gsl::span<uint32_t>,
  gsl::span<unsigned>,
  gsl::span<uint32_t>,
  gsl::span<unsigned>>
HostBuffersManager::getBufferOutputData(size_t b)
{
  if (b > host_buffers.size()) return {};

  HostBuffers* buf = host_buffers.at(b);
  return {buf->host_passing_event_list,
          buf->host_dec_reports,
          buf->host_routingbits,
          buf->host_sel_reports,
          buf->host_sel_report_offsets,
          buf->host_lumi_summaries,
          buf->host_lumi_summary_offsets};
}

void HostBuffersManager::printStatus() const
{
  info_cout << host_buffers.size() << " buffers; " << empty_buffers.size() << " empty; " << filled_buffers.size()
            << " filled." << std::endl;
}
