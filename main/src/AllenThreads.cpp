/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <any>
#include <string>
#include <thread>
#include <pthread.h>

#include <zmq_compat.h>
#include <ZeroMQ/IZeroMQSvc.h>
#include <AllenThreads.h>

#include <OutputHandler.h>
#include <HostBuffersManager.cuh>
#include <MonitorManager.h>
#include <CheckerInvoker.h>
#include <ROOTService.h>
#include <MCRaw.h>
#include <InputProvider.h>
#include <Stream.h>
#include <Tools.h>

#include "MonitoringAggregator.h"
#include "MonitoringPrinter.h"

namespace {
  using namespace zmq;
  using namespace std::string_literals;
} // namespace

void set_current_thread_name(const std::string& thread_name)
{
#ifdef __linux__
  pthread_setname_np(pthread_self(), thread_name.c_str());
#else
  pthread_setname_np(thread_name.c_str());
#endif
}

std::string connection(const size_t id, std::string suffix)
{
  auto con = std::string {"inproc://control_"} + std::to_string(id);
  if (!suffix.empty()) {
    con += "_" + suffix;
  }
  return con;
}

zmq::socket_t make_control(size_t thread_id, IZeroMQSvc* zmqSvc, std::string suffix = std::string {})
{

  auto make_socket = [thread_id, &suffix, zmqSvc] {
    zmq::socket_t control = zmqSvc->socket(zmq::PAIR);
    zmq::setsockopt(control, zmq::LINGER, 0);
    auto con = connection(thread_id, suffix);
    try {
      control.connect(con.c_str());
    } catch (const zmq::error_t& e) {
      error_cout << "failed to connect connection " << con << "\n";
      throw e;
    }
    return control;
  };

  auto control = make_socket();
  zmq::pollitem_t items[] = {{control, 0, zmq::POLLIN, 0}};

  bool connected = false;
  unsigned int tries = 5;
  while (!connected && tries > 0) {
    zmqSvc->poll(&items[0], 1, 500);
    if (items[0].revents & zmq::POLLIN) {
      auto msg = zmqSvc->receive<std::string>(control);
      assert(msg == "STATUS");
      zmqSvc->send(control, "READY", send_flags::sndmore);
      zmqSvc->send(control, thread_id);
      connected = true;
    }
    else {
      control = make_socket();
      items[0] = {control, 0, zmq::POLLIN, 0};
      --tries;
    }
  }

  if (!connected) {
    auto msg = "Failed to connect control socket for thread "s + std::to_string(thread_id);
    throw std::runtime_error {msg};
  }

  return control;
}

void run_output(
  const size_t thread_id,
  const size_t output_id,
  IZeroMQSvc* zmqSvc,
  OutputHandler* output_handler,
  HostBuffersManager* buffer_manager)
{
  // Set thread name for easier debugging
  auto thread_name = std::string {"output_"} + std::to_string(output_id);
  set_current_thread_name(thread_name);

  auto* client_socket = output_handler ? output_handler->client_socket() : nullptr;

  std::vector<zmq::pollitem_t> items(client_socket ? 2 : 1);
  if (client_socket) {
    items[1] = {*client_socket, 0, zmq::POLLIN, 0};
  }

  // Create a control socket and connect it.
  zmq::socket_t control = make_control(thread_id, zmqSvc);
  items[0] = {control, 0, zmq::POLLIN, 0};

  while (true) {

    // Check if there are messages
    zmqSvc->poll(&items[0], items.size(), -1);

    if (client_socket && (items[1].revents & zmq::POLLIN)) {
      output_handler->handle();
    }

    if (items[0].revents & zmq::POLLIN) {
      bool more = false;
      auto msg = zmqSvc->receive<std::string>(control, &more);
      if (msg == "DONE") {
        break;
      }
      else if (msg == "WRITE") {
        auto slc_idx = zmqSvc->receive<size_t>(control);
        auto first_evt = zmqSvc->receive<size_t>(control);
        auto buf_idx = zmqSvc->receive<size_t>(control);
        bool success = true;
        size_t n_written = 0;

        if (output_handler != nullptr) {
          std::tie(success, n_written) = output_handler->output_selected_events(
            output_id, slc_idx, first_evt, *buffer_manager->get_persistent_store(buf_idx));
        }

        zmqSvc->send(control, "WRITTEN", send_flags::sndmore);
        zmqSvc->send(control, slc_idx, send_flags::sndmore);
        zmqSvc->send(control, first_evt, send_flags::sndmore);
        zmqSvc->send(control, buf_idx, send_flags::sndmore);
        zmqSvc->send(control, success, send_flags::sndmore);
        zmqSvc->send(control, n_written);
      }
      else {
        error_cout << "Output threads got unknown message: " << msg << "\n";
        while (more) {
          zmqSvc->receive<zmq::message_t>(control, &more);
        }
      }
    }
  }
}

/**
 * @brief      Request slices from the input provider and report
 *             them to the main thread; run from a separate thread
 *
 * @param      thread ID of this I/O thread
 * @param      IInputProvider instance
 *
 * @return     void
 */
void run_slices(const size_t thread_id, IZeroMQSvc* zmqSvc, IInputProvider* input_provider)
{
  // Set thread name for easier debugging
  auto thread_name = std::string {"slices_"} + std::to_string(thread_id);
  set_current_thread_name(thread_name);

  // Create a control socket and connect it.
  zmq::socket_t control = make_control(thread_id, zmqSvc);

  zmq::pollitem_t items[] = {{control, 0, zmq::POLLIN, 0}};

  int timeout = -1;
  uint current_run_number = 0;
  while (true) {

    // Check if there are messages without blocking
    zmqSvc->poll(&items[0], 1, timeout);

    if (items[0].revents & zmq::POLLIN) {
      auto msg = zmqSvc->receive<std::string>(control);
      if (msg == "DONE") {
        break;
      }
      else if (msg == "START") {
        timeout = 0;
      }
    }

    // Get a slice and inform the main thread that it is available
    // NOTE: the argument specifies the timeout in ms, not the number of events.
    auto [good, done, timed_out, slice_index, n_filled, a] = input_provider->get_slice(1000);
    // Report errors or good slices that contain events
    if (!timed_out && good && n_filled != 0) {
      // If run number has change then report this first
      if (a.has_value()) {
        auto odin_data = std::any_cast<gsl::span<unsigned const>>(a);
        LHCb::ODIN odin {odin_data};
        if (odin.runNumber() == 0) {
          info_cout << "ODIN run number 0, skipping \n";
          continue;
        }
        else if (odin.runNumber() != current_run_number) {
          current_run_number = odin.runNumber();
          zmqSvc->send(control, "RUN", send_flags::sndmore);
          zmqSvc->send(control, odin.data);
        }
      }
      zmqSvc->send(control, "SLICE", send_flags::sndmore);
      zmqSvc->send(control, slice_index, send_flags::sndmore);
      zmqSvc->send(control, n_filled);
    }
    else if (!good) {
      zmqSvc->send(control, "ERROR");
      break;
    }
    if (done) {
      zmqSvc->send(control, "DONE");
      timeout = -1;
    }
  }
}

/**
 * @brief      Process events on GPU streams; run from a separate thread
 *
 * @param      thread ID
 * @param      GPU stream ID
 * @param      GPU device id
 * @param
 * @param      CUDA device
 *
 * @return     return type
 */
void run_stream(
  size_t const thread_id,
  size_t const stream_id,
  int device_id,
  Stream* stream,
  std::shared_ptr<IInputProvider> input_provider,
  IZeroMQSvc* zmqSvc,
  CheckerInvoker* checker_invoker,
  ROOTService* root_service,
  unsigned n_reps,
  bool mep_layout,
  uint inject_mem_fail,
  [[maybe_unused]] bool prefer_shared)
{
  Allen::set_device(device_id, stream_id);

  // Set thread name for easier debugging
  auto thread_name = std::string {"stream_"} + std::to_string(stream_id);
  set_current_thread_name(thread_name);

#if defined(TARGET_DEVICE_CUDA)
  if (prefer_shared) {
    cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  }
#endif

  zmq::socket_t control = make_control(thread_id, zmqSvc);

  zmq::pollitem_t items[] = {
    {control, 0, ZMQ_POLLIN, 0},
  };

  while (true) {

    zmqSvc->poll(&items[0], 1, -1);

    std::string command;
    std::optional<size_t> idx;
    size_t buf;
    size_t first;
    size_t last;
    if (items[0].revents & zmq::POLLIN) {
      command = zmqSvc->receive<std::string>(control);
      if (command == "DONE") {
        break;
      }
      else if (command != "PROCESS") {
        error_cout << "processor " << stream_id << " received bad command: " << command << "\n";
      }
      else {
        idx = zmqSvc->receive<size_t>(control);
        first = zmqSvc->receive<size_t>(control);
        last = zmqSvc->receive<size_t>(control);
        buf = zmqSvc->receive<size_t>(control);
      }
    }

    if (idx) {
      // Run the stream
      auto status = stream->run(
        buf,
        {input_provider,
         *idx,
         {static_cast<unsigned>(first), static_cast<unsigned>(last)},
         n_reps,
         mep_layout,
         inject_mem_fail,
         checker_invoker,
         root_service});

      if (status == Allen::error::errorMemoryAllocation) {
        zmqSvc->send(control, "SPLIT", send_flags::sndmore);
        zmqSvc->send(control, *idx, send_flags::sndmore);
        zmqSvc->send(control, first, send_flags::sndmore);
        zmqSvc->send(control, last, send_flags::sndmore);
        zmqSvc->send(control, buf);
      }
      else if (status == Allen::error::success) {
        // signal that we're done
        zmqSvc->send(control, "PROCESSED", send_flags::sndmore);
        zmqSvc->send(control, *idx, send_flags::sndmore);
        zmqSvc->send(control, first, send_flags::sndmore);
        zmqSvc->send(control, buf);
      }
    }
  }
}

/**
 * @brief      Receive filled HostBuffers from GPU
 *             threads and produce rate histograms
 *
 * @param      thread ID of this monitoring thread
 * @param      manager for the monitor objects
 * @param      index of the monitor objects to use for this thread
 *
 * @return     void
 */
void run_monitoring(const size_t mon_id, IZeroMQSvc* zmqSvc, MonitorManager* monitor_manager, unsigned i_monitor)
{
  // Set thread name for easier debugging
  auto thread_name = std::string {"monitoring_"} + std::to_string(mon_id);
  set_current_thread_name(thread_name);

  zmq::socket_t control = make_control(mon_id, zmqSvc);
  zmq::pollitem_t items[] = {{control, 0, zmq::POLLIN, 0}};

  while (true) {
    // Check if there are messages
    zmqSvc->poll(&items[0], 1, -1);

    std::optional<size_t> buf_idx;
    if (items[0].revents & zmq::POLLIN) {
      auto msg = zmqSvc->receive<std::string>(control);
      if (msg == "DONE") {
        break;
      }
      else if (msg != "MONITOR") {
        error_cout << "monitor thread " << mon_id << " received bad command: " << msg << "\n";
      }
      else {
        buf_idx = zmqSvc->receive<size_t>(control);
      }
    }

    if (buf_idx) {
      monitor_manager->fill(i_monitor, *buf_idx);
      zmqSvc->send(control, "MONITORED", send_flags::sndmore);
      zmqSvc->send(control, *buf_idx, send_flags::sndmore);
      zmqSvc->send(control, i_monitor);
    }
  }
}

void run_aggregation(
  const size_t thread_id,
  IZeroMQSvc* zmqSvc,
  MonitoringAggregator* aggregator,
  MonitoringPrinter* printer)
{
  // Set thread name for easier debugging
  auto thread_name = std::string {"aggregation_"} + std::to_string(thread_id);
  set_current_thread_name(thread_name);

  zmq::socket_t control = make_control(thread_id, zmqSvc);
  zmq::pollitem_t items[] = {{control, 0, zmq::POLLIN, 0}};

  Timer t;
  bool started = false;

  while (true) {
    // Run timer to attempt to keep aggregation out of sync with any sinks
    t.restart();

    // Check for signal to start, explictly aggregate or terminate and
    // otherwise continue processing
    if (zmqSvc->poll(&items[0], 1, started ? 0 : -1) > 0) {
      if (items[0].revents & zmq::POLLIN) {
        auto msg = zmqSvc->receive<std::string>(control);
        if (msg == "START") {
          if (aggregator) aggregator->start();
          zmqSvc->send(control, true);
          started = true;
        }
        else if (msg == "AGGREGATE") {
          // Run the aggregator and printer
          if (aggregator) aggregator->process();
          if (printer) printer->process();
          zmqSvc->send(control, true);
        }
        else if (msg == "DONE") {
          started = false;
          break;
        }
      }
    }
    else {
      // Run the aggregator and printer
      if (aggregator) aggregator->process();
      if (printer) printer->process();
    }

    // Sleep
    long millis = static_cast<long>(1000 * t.get_elapsed_time());
    std::this_thread::sleep_for(std::chrono::milliseconds(997 - millis));
    t.stop();
  }

  if (printer) printer->process(true);
}
