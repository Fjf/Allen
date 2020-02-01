#pragma once
#include <vector>

#include <Logger.h>
#include <BankTypes.h>
#include <Timer.h>

#include <zmq/zmq.hpp>

struct IInputProvider;

class OutputHandler {
public:
  OutputHandler(IInputProvider const* input_provider, size_t eps) :
    m_input_provider {input_provider}, m_sizes(eps)
  {}

  virtual ~OutputHandler() {}

  bool output_selected_events(
    size_t const slice_index,
    size_t const event_offset,
    gsl::span<unsigned int const> const selected_events,
    gsl::span<uint32_t const> const dec_reports);

  virtual zmq::socket_t* client_socket() { return nullptr; }

  virtual void handle() {}

protected:
  virtual std::tuple<size_t, gsl::span<char>> buffer(size_t buffer_size) = 0;

  virtual bool write_buffer(size_t id) = 0;

  IInputProvider const* m_input_provider = nullptr;
  std::vector<size_t> m_sizes;
  std::array<uint32_t, 4> m_trigger_mask = {~0u, ~0u, ~0u, ~0u};
};
