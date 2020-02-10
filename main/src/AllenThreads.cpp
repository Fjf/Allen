#include <string>
#include <thread>

#include <ZeroMQ/IZeroMQSvc.h>
#include <AllenThreads.h>

#include <OutputHandler.h>
#include <HostBuffersManager.cuh>
#include <MonitorManager.h>
#include <CheckerInvoker.h>
#include <InputProvider.h>
#include <StreamWrapper.cuh>
#include <Tools.h>

namespace {
  using namespace zmq;
}

std::string connection(const size_t id, std::string suffix)
{
  auto con = std::string {"inproc://control_"} + std::to_string(id);
  if (!suffix.empty()) {
    con += "_" + suffix;
  }
  return con;
}

void run_output(
  const size_t thread_id,
  IZeroMQSvc* zmqSvc,
  OutputHandler* output_handler,
  HostBuffersManager* buffer_manager)
{
  // Create a control socket and connect it.
  zmq::socket_t control = zmqSvc->socket(zmq::PAIR);
  zmq::setsockopt(control, zmq::LINGER, 0);

  auto con = connection(thread_id);
  try {
    control.connect(con.c_str());
  } catch (const zmq::error_t& e) {
    error_cout << "failed to connect connection " << con << "\n";
    throw e;
  }

  auto* client_socket = output_handler ? output_handler->client_socket() : nullptr;

  std::vector<zmq::pollitem_t> items(client_socket ? 2 : 1);
  items[0] = {control, 0, zmq::POLLIN, 0};
  if (client_socket) {
    items[1] = {*client_socket, 0, zmq::POLLIN, 0};
  }


  while (true) {

    // Check if there are messages
    zmq::poll(&items[0], items.size(), -1);

    if (client_socket && (items[1].revents & zmq::POLLIN)) {
      output_handler->handle();
    }

    if (items[0].revents & zmq::POLLIN) {
      auto msg = zmqSvc->receive<std::string>(control);
      if (msg == "DONE") {
        break;
      }
      else if (msg == "WRITE") {
        auto slc_idx = zmqSvc->receive<size_t>(control);
        auto first_evt = zmqSvc->receive<size_t>(control);
        auto buf_idx = zmqSvc->receive<size_t>(control);

        bool success = true;
        auto [passing_event_list, dec_reports] = buffer_manager->getBufferOutputData(buf_idx);
        if (output_handler != nullptr) {
          success =
            output_handler->output_selected_events(slc_idx, first_evt, passing_event_list, dec_reports);
        }

        zmqSvc->send(control, "WRITTEN", send_flags::sndmore);
        zmqSvc->send(control, slc_idx, send_flags::sndmore);
        zmqSvc->send(control, first_evt, send_flags::sndmore);
        zmqSvc->send(control, buf_idx, send_flags::sndmore);
        zmqSvc->send(control, success, send_flags::sndmore);
        zmqSvc->send(control, static_cast<size_t>(passing_event_list.size()));
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
void run_slices(
  const size_t thread_id,
  IZeroMQSvc* zmqSvc,
  IInputProvider* input_provider)
{

  // Create a control socket and connect it.
  zmq::socket_t control = zmqSvc->socket(zmq::PAIR);
  zmq::setsockopt(control, zmq::LINGER, 0);

  auto con = connection(thread_id);
  try {
    control.connect(con.c_str());
  } catch (const zmq::error_t& e) {
    error_cout << "failed to connect connection " << con << "\n";
    throw e;
  }

  zmq::pollitem_t items[] = {{control, 0, zmq::POLLIN, 0}};

  int timeout = -1;
  while (true) {

    // Check if there are messages without blocking
    zmq::poll(&items[0], 1, timeout);

    if (items[0].revents & zmq::POLLIN) {
      auto msg = zmqSvc->receive<std::string>(control);
      if (msg == "DONE") {
        break;
      } else if (msg == "START") {
        timeout = 0;
      }
    }

    // Get a slice and inform the main thread that it is available
    // NOTE: the argument specifies the timeout in ms, not the number of events.
    auto [good, done, timed_out, slice_index, n_filled] = input_provider->get_slice(1000);
    // Report errors or good slices that contain events
    if (!timed_out && good && n_filled != 0) {
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
  StreamWrapper* wrapper,
  IInputProvider const* input_provider,
  IZeroMQSvc* zmqSvc,
  CheckerInvoker* checker_invoker,
  uint n_reps,
  bool do_check,
  bool cpu_offload,
  bool mep_layout,
  std::string folder_name_imported_forward_tracks)
{
  auto make_control = [thread_id, zmqSvc](std::string suffix = std::string {}) {
    zmq::socket_t control = zmqSvc->socket(zmq::PAIR);
    zmq::setsockopt(control, zmq::LINGER, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds {50});
    auto con = connection(thread_id, suffix);
    try {
      control.connect(con.c_str());
    } catch (const zmq::error_t& e) {
      error_cout << "failed to connect connection " << con << "\n";
      throw e;
    }
    return control;
  };

  zmq::socket_t control = make_control();
  std::optional<zmq::socket_t> check_control;
  if (do_check) {
    check_control = make_control("check");
  }

  auto [device_set, device_name] = set_device(device_id, stream_id);

  zmq::pollitem_t items[] = {
    {control, 0, ZMQ_POLLIN, 0},
  };

  // Indicate to the main thread that we are ready to process
  zmqSvc->send(control, "READY", send_flags::sndmore);
  zmqSvc->send(control, device_set);

  while (true) {

    // Wait until we need to process
    std::optional<int> n;
    do {
      try {
        n = zmq::poll(&items[0], 1, -1);
      } catch (const zmq::error_t& err) {
        if (err.num() == EINTR) {
          continue;
        }
        else {
          warning_cout << "processor caught exception." << err.what() << "\n";
        }
      }
    } while (!n);

    n.reset();

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
      auto status = wrapper->run_stream(
        stream_id,
        buf,
        {input_provider->banks(BankTypes::VP, *idx),
         input_provider->banks(BankTypes::UT, *idx),
         input_provider->banks(BankTypes::FT, *idx),
         input_provider->banks(BankTypes::MUON, *idx),
         {static_cast<uint>(first), static_cast<uint>(last)},
         n_reps,
         do_check,
         cpu_offload,
         mep_layout});

      if (status == cudaErrorMemoryAllocation) {
        zmqSvc->send(control, "SPLIT", send_flags::sndmore);
        zmqSvc->send(control, *idx, send_flags::sndmore);
        zmqSvc->send(control, first, send_flags::sndmore);
        zmqSvc->send(control, last, send_flags::sndmore);
        zmqSvc->send(control, buf);
      }
      else if (status == cudaSuccess) {
        // signal that we're done
        zmqSvc->send(control, "PROCESSED", send_flags::sndmore);
        zmqSvc->send(control, *idx, send_flags::sndmore);
        zmqSvc->send(control, first, send_flags::sndmore);
        zmqSvc->send(control, buf);
        if (do_check && check_control) {
          // Get list of events that are in the slice
          auto const& events = input_provider->event_ids(*idx, first, last);

          // synchronise to avoid threading issues with
          // CheckerInvoker. The main thread will send the folder to
          // only one stream at a time and will block until it receives
          // the message that informs it the checker is done.
          auto mc_folder = zmqSvc->receive<std::string>(*check_control);
          auto mask = wrapper->reconstructed_events(stream_id);
          auto mc_events = checker_invoker->load(mc_folder, events, mask);

          if (mc_events.empty()) {
            zmqSvc->send(*check_control, false);
          }
          else {
            // Run the checker
            std::vector<Checker::Tracks> forward_tracks;
            if (!folder_name_imported_forward_tracks.empty()) {
              std::vector<char> events_tracks;
              std::vector<uint> event_tracks_offsets;
              read_folder(
                folder_name_imported_forward_tracks, events, mask, events_tracks, event_tracks_offsets, true);
              forward_tracks =
                read_forward_tracks(events_tracks.data(), event_tracks_offsets.data(), events.size());
            }

            wrapper->run_monte_carlo_test(stream_id, *checker_invoker, mc_events, forward_tracks);
            zmqSvc->send(*check_control, true);
          }
        }
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
void run_monitoring(const size_t mon_id, IZeroMQSvc* zmqSvc, MonitorManager* monitor_manager, uint i_monitor)
{

  // Create a control socket and connect it.
  zmq::socket_t control = zmqSvc->socket(zmq::PAIR);
  zmq::setsockopt(control, zmq::LINGER, 0);

  auto con = connection(mon_id);
  try {
    control.connect(con.c_str());
  } catch (const zmq::error_t& e) {
    error_cout << "failed to connect connection " << con << "\n";
    throw e;
  }

  zmq::pollitem_t items[] = {{control, 0, zmq::POLLIN, 0}};

  while (true) {

    // Wait until we need to process
    std::optional<int> n;
    do {
      try {
        n = zmq::poll(&items[0], 1, -1);
      } catch (const zmq::error_t& err) {
        if (err.num() == EINTR) {
          continue;
        }
        else {
          warning_cout << "processor caught exception." << err.what() << "\n";
        }
      }
    } while (!n);

    n.reset();

    // Check if there are messages
    zmq::poll(&items[0], 1, 0);

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
