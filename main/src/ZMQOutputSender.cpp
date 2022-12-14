/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <limits.h>
#include <unistd.h>

#include <string>

#include <ZMQOutputSender.h>
#include <InputProvider.h>
#include <ZeroMQ/IZeroMQSvc.h>
#include <zmq_compat.h>
#include <Logger.h>

#ifdef __APPLE__
#define HOST_NAME_MAX 64
#endif

namespace {
  using namespace std::string_literals;
  using namespace zmq;
} // namespace

namespace Utils {
  std::string hostname()
  {
    char hname[HOST_NAME_MAX];
    std::string hn;
    if (!gethostname(hname, sizeof(hname))) {
      hn = std::string {hname};
      auto pos = hn.find('.');
      if (pos != std::string::npos) {
        hn = hn.substr(0, pos);
      }
    }
    return hn;
  }
} // namespace Utils

ZMQOutputSender::ZMQOutputSender(
  IInputProvider const* input_provider,
  std::string const receiver_connection,
  size_t const output_batch_size,
  size_t const n_lines,
  IZeroMQSvc* zmqSvc,
  bool const checksum) :
  OutputHandler {input_provider, receiver_connection, 1u, output_batch_size, n_lines, checksum},
  m_zmq {zmqSvc}
{
  auto const pos = receiver_connection.rfind(":");
  auto const receiver = receiver_connection.substr(0, pos);

  m_request = m_zmq->socket(zmq::REQ);
  zmq::setsockopt(*m_request, zmq::LINGER, 0);
  m_request->connect(receiver_connection.c_str());
  m_id = "Allen_"s + Utils::hostname() + "_" + std::to_string(::getpid());
  m_zmq->send(*m_request, "PORT", send_flags::sndmore);
  m_zmq->send(*m_request, m_id);

  // Wait for reply for a second
  zmq::pollitem_t items[] = {{*m_request, 0, zmq::POLLIN, 0}};
  zmq::poll(&items[0], 1, 1000);
  if (items[0].revents & zmq::POLLIN) {
    auto port = m_zmq->receive<std::string>(*m_request);
    std::string connection = receiver + ":" + port;
    m_socket = m_zmq->socket(zmq::PAIR);
    zmq::setsockopt(*m_socket, zmq::LINGER, 0);
    m_socket->connect(connection.c_str());
    info_cout << "Connected MDF output socket to " << connection << "\n";
    m_connected = true;
  }
  else {
    m_connected = false;
    throw std::runtime_error {"Failed to connect to receiver on "s + receiver_connection};
  }
}

ZMQOutputSender::~ZMQOutputSender()
{
  if (m_connected && m_request) {
    m_zmq->send(*m_request, "CLIENT_EXIT", send_flags::sndmore);
    m_zmq->send(*m_request, m_id);
    zmq::pollitem_t items[] = {{*m_request, 0, zmq::POLLIN, 0}};
    zmq::poll(&items[0], 1, 500);
    if (items[0].revents & zmq::POLLIN) {
      m_zmq->receive<std::string>(*m_request);
    }
  }
}

zmq::socket_t* ZMQOutputSender::client_socket() const { return (m_connected && m_socket) ? &(*m_socket) : nullptr; }

void ZMQOutputSender::handle()
{
  auto msg = m_zmq->receive<std::string>(*m_socket);
  if (msg == "RECEIVER_STOP") {
    info_cout << "MDF receiver is exiting\n";
    m_zmq->send(*m_socket, "OK");
    m_connected = false;
  }
  else {
    error_cout << "Received unknown message from output receiver: " << msg << "\n";
  }
}

gsl::span<char> ZMQOutputSender::buffer(size_t, size_t buffer_size, size_t)
{
  m_buffer.rebuild(buffer_size);
  return gsl::span {static_cast<char*>(m_buffer.data()), static_cast<events_size>(buffer_size)};
}

bool ZMQOutputSender::write_buffer(size_t)
{
  if (m_connected) {
    m_zmq->send(*m_socket, "EVENT", send_flags::sndmore);
    m_zmq->send(*m_socket, m_buffer);
  }

  return true;
}
