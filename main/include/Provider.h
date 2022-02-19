/*****************************************************************************\
 * (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <memory>
#include <map>
#include <string>

#include "BankTypes.h"
#include "InputProvider.h"
#include "OutputHandler.h"

class IZeroMQSvc;

namespace {
  constexpr size_t n_write = 1;
  constexpr size_t n_input = 1;
  constexpr size_t n_io = n_input + n_write;
  constexpr size_t n_mon = 1;
  constexpr size_t max_stream_threads = 1024;
} // namespace

namespace Allen {
  struct IOConf {
    bool async_io = false;
    unsigned number_of_slices = 0;
    unsigned number_of_repetitions = 0;
    unsigned n_io_reps = 0;
  };

  std::unordered_set<BankTypes> configured_bank_types(std::string const& json_file);

  std::tuple<std::string, bool> sequence_conf(std::map<std::string, std::string> const& options);

  std::shared_ptr<IInputProvider> make_provider(std::map<std::string, std::string> const& options);

  std::unique_ptr<OutputHandler> output_handler(
    IInputProvider* input_provider,
    IZeroMQSvc* zmq_svc,
    std::map<std::string, std::string> const& options);

  Allen::IOConf io_configuration(
    unsigned number_of_slices,
    unsigned number_of_repetitions,
    unsigned number_of_threads,
    bool quiet = false);
} // namespace Allen