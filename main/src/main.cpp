/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
/**
 *      CUDA HLT1
 *
 *      author  -  GPU working group
 *      e-mail  -  lhcb-parallelization@cern.ch
 *
 *      Started development on February, 2018
 *      CERN
 */
#include <getopt.h>
#include <cstring>
#include <map>
#include <string>
#include <iostream>
#include <vector>
#include <Allen.h>
#include <Updater.h>
#include <ProgramOptions.h>
#include <Logger.h>
#include <Timer.h>
#include <ZeroMQ/IZeroMQSvc.h>
#include <zmq/svc.h>
#include <Provider.h>

#ifdef DEBUG
#include <fenv.h>
#endif

int main(int argc, char* argv[])
{
#ifdef DEBUG
  feenableexcept(FE_INVALID | FE_DIVBYZERO);
#endif

  const auto program_options = allen_program_options();

  // Options object that will be passed to Allen
  std::map<std::string, std::string> allen_options;

  // Create long_options from program_options
  std::vector<option> long_options;
  std::string accepted_single_letter_options = "h";
  for (const auto& po : program_options) {
    for (const auto& opt : po.options) {
      if (opt.length() > 1) {
        long_options.push_back(option {opt.c_str(), required_argument, nullptr, 0});
      }
      else {
        accepted_single_letter_options += opt + ":";
      }
    }
  }
  long_options.push_back({nullptr, 0, nullptr, 0});

  int option_index = 0;
  signed char c;
  while ((c = getopt_long(argc, argv, accepted_single_letter_options.c_str(), long_options.data(), &option_index)) !=
         -1) {
    switch (c) {
    case 0:
      for (const auto& po : program_options) {
        for (const auto& opt : po.options) {
          if (std::string(long_options[option_index].name) == opt) {
            if (optarg) {
              allen_options[opt] = optarg;
            }
            else {
              allen_options[opt] = "1";
            }
          }
        }
      }
      break;
    default:
      bool found_opt = false;
      for (const auto& po : program_options) {
        for (const auto& opt : po.options) {
          if (std::string {(char) c} == opt) {
            if (optarg) {
              allen_options[std::string {(char) c}] = optarg;
            }
            else {
              allen_options[std::string {(char) c}] = "1";
            }
            found_opt = true;
          }
        }
      }
      if (!found_opt) {
        // If we reach this point, it is not supported
        print_usage(argv, program_options);
        return -1;
      }
      break;
    }
  }

  // Iterate all options with default values and put those in
  // if they were not specified
  for (const auto& po : program_options) {
    bool initialized = false;
    for (const auto& opt : po.options) {
      const auto it = allen_options.find(opt);
      if (it != allen_options.end()) {
        initialized = true;
      }
    }

    if (po.options[0] == "sequence" && !initialized) {
      error_cout << "--sequence option must be set. See usage below.\n";
      print_usage(argv, program_options);
      return -1;
    }

    if (!initialized && po.default_value != "") {
      allen_options[po.options[0]] = po.default_value;
    }
  }

  auto zmqSvc = makeZmqSvc();

  Allen::NonEventData::Updater updater {allen_options};
  auto input_provider = Allen::make_provider(allen_options);
  auto output_handler = Allen::output_handler(input_provider.get(), zmqSvc, allen_options);
  if (!input_provider) return -1;
  return allen(std::move(allen_options), &updater, std::move(input_provider), output_handler.get(), zmqSvc, "");
}
