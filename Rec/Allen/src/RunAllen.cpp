/***************************************************************************** \
 * (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/

/**
 * Call Allen to run at one event at a time
 *
 * author Dorothea vom Bruch
 *
 */
#include <boost/algorithm/string.hpp>

#include "RunAllen.h"
#include "ROOTService.h"

DECLARE_COMPONENT(RunAllen)

namespace {
  std::string resolveEnvVars(std::string s)
  {
    std::regex envExpr {"\\$\\{([A-Za-z0-9_]+)\\}"};
    std::smatch m;
    while (std::regex_search(s, m, envExpr)) {
      std::string rep;
      System::getEnv(m[1].str(), rep);
      s = s.replace(m[1].first - 2, m[1].second + 1, rep);
    }
    return s;
  }
} // namespace

RunAllen::RunAllen(const std::string& name, ISvcLocator* pSvcLocator) :
  MultiTransformerFilter(
    name,
    pSvcLocator,
    // Inputs
    {KeyValue {"AllenRawInput", "Allen/Raw/Input"}, KeyValue {"ODINLocation", LHCb::ODINLocation::Default}},
    // Outputs
    {KeyValue {"AllenOutput", "Allen/Out/HostBuffers"}, KeyValue {"DecReportsLocation", "Allen/Out/DecReports"}})
{}

StatusCode RunAllen::initialize()
{
  auto sc = MultiTransformerFilter::initialize();
  if (sc.isFailure()) return sc;
  if (msgLevel(MSG::DEBUG)) debug() << "==> Initialize" << endmsg;

  /* initialize Allen */

  // Get updater service and register all consumers
  auto svc = service(m_updaterName);
  if (!svc) {
    error() << "Failed get updater " << m_updaterName.value() << endmsg;
    return StatusCode::FAILURE;
  }
  auto* updater = dynamic_cast<Allen::NonEventData::IUpdater*>(svc.get());
  if (!updater) {
    error() << "Failed cast updater " << m_updaterName.value() << " to Allen::NonEventData::IUpdater " << endmsg;
    return StatusCode::FAILURE;
  }

  // get constants
  const std::string params_path = resolveEnvVars(m_paramDir);

  std::vector<float> muon_field_of_interest_params;
  read_muon_field_of_interest(muon_field_of_interest_params, params_path + "/field_of_interest_params.bin");

  m_constants.reserve_and_initialize(muon_field_of_interest_params, params_path + "/params_kalman_FT6x2/");

  std::unique_ptr<CatboostModelReader> muon_catboost_model_reader =
    std::make_unique<CatboostModelReader>(params_path + "/muon_catboost_model.json");
  m_constants.initialize_muon_catboost_model_constants(
    muon_catboost_model_reader->n_trees(),
    muon_catboost_model_reader->tree_depths(),
    muon_catboost_model_reader->tree_offsets(),
    muon_catboost_model_reader->leaf_values(),
    muon_catboost_model_reader->leaf_offsets(),
    muon_catboost_model_reader->split_border(),
    muon_catboost_model_reader->split_feature());

  std::unique_ptr<CatboostModelReader> two_track_catboost_model_reader =
    std::make_unique<CatboostModelReader>(params_path + "/two_track_catboost_model_small.json");
  m_constants.initialize_two_track_catboost_model_constants(
    two_track_catboost_model_reader->n_trees(),
    two_track_catboost_model_reader->tree_depths(),
    two_track_catboost_model_reader->tree_offsets(),
    two_track_catboost_model_reader->leaf_values(),
    two_track_catboost_model_reader->leaf_offsets(),
    two_track_catboost_model_reader->split_border(),
    two_track_catboost_model_reader->split_feature());

  // Allen Consumers
  register_consumers(updater, m_constants);

  // Run all registered producers and consumers
  updater->update(0);

  // Read configuration
  std::string conf_file = resolveEnvVars(m_json);
  ConfigurationReader configuration_reader(conf_file);

  // Initialize stream
  const bool print_memory_usage = false;
  const unsigned start_event_offset = 0;
  const size_t reserve_mb = 10; // to do: how much do we need maximally for one event?
  const unsigned required_memory_alignment =
    64; // 64 bytes is equivalent to 512-bit alignment (currently widest vectors)

  m_stream_wrapper.reset(new StreamWrapper());
  m_stream_wrapper->initialize_streams(
    m_number_of_streams,
    print_memory_usage,
    start_event_offset,
    reserve_mb,
    reserve_mb, // host memory same as "device"
    required_memory_alignment,
    m_constants,
    configuration_reader.params());

  // Initialize host buffers (where Allen output is stored)
  m_host_buffers_manager.reset(
    new HostBuffersManager(m_n_buffers, 2, m_line_names.size(), m_do_check, m_stream_wrapper->errorevent_line));
  m_stream_wrapper->initialize_streams_host_buffers_manager(m_host_buffers_manager.get());

  // Initialize input provider
  const size_t number_of_slices = 1;
  const size_t events_per_slice = 1;
  const size_t n_events = 1;
  m_tes_input_provider.reset(
    new TESProvider<BankTypes::VP, BankTypes::UT, BankTypes::FT, BankTypes::MUON, BankTypes::ODIN>(
      number_of_slices, events_per_slice, n_events));

  // Get HLT1 selection names from configuration and initialize rate counters
  auto selection_params = configuration_reader.params("gather_selections");
  if (selection_params.empty()) {
    error() << "Failed to obtain parameters of gather_selections from " << m_json << endmsg;
    return StatusCode::FAILURE;
  }
  auto selection_names = selection_params.find("names_of_active_lines");
  if (selection_names == selection_params.end()) {
    error() << "Failed to obtain names_of_active_lines from gather_selections " << endmsg;
    return StatusCode::FAILURE;
  }
  boost::split(m_line_names, selection_names->second, boost::is_any_of(","));
  if (m_line_names.empty()) {
    error() << "Failed to obtain any line names from " << selection_names->second << endmsg;
    return StatusCode::FAILURE;
  }
  else {
    for (auto line_name : m_line_names) {
      debug() << line_name << " ";
    }
    debug() << endmsg;
  }

  m_hlt1_line_rates.reserve(m_line_names.size());
  for (unsigned i = 0; i < m_line_names.size(); ++i) {
    const std::string name = m_line_names[i] + "Decision";
    m_hlt1_line_rates.emplace_back(this, "Selected by " + name);
  }

  // Set verbosity level
  logger::setVerbosity(6 - this->msgLevel());

  return StatusCode::SUCCESS;
}

/** Calls Allen for one event
 */
std::tuple<bool, HostBuffers, LHCb::HltDecReports> RunAllen::operator()(
  const std::array<std::tuple<std::vector<char>, int>, LHCb::RawBank::types().size()>& allen_banks,
  const LHCb::ODIN&) const
{
  int rv = m_tes_input_provider->set_banks(allen_banks, m_bankTypes);
  if (rv > 0) {
    error() << "Error in reading dumped raw banks" << endmsg;
  }

  // initialize RuntimeOptions
  const unsigned event_start = 0;
  const unsigned event_end = 1;
  const size_t slice_index = 0;
  const bool mep_layout = false;
  const uint inject_mem_fail = 0;
  auto root_service = std::make_unique<ROOTService>();
  RuntimeOptions runtime_options {m_tes_input_provider.get(),
                                  slice_index,
                                  {event_start, event_end},
                                  m_number_of_repetitions,
                                  m_do_check,
                                  m_cpu_offload,
                                  mep_layout,
                                  inject_mem_fail,
                                  nullptr,
                                  root_service.get()};

  const unsigned buf_idx = m_n_buffers - 1;
  const unsigned stream_index = m_number_of_streams - 1;
  Allen::error cuda_rv = m_stream_wrapper->run_stream(stream_index, buf_idx, runtime_options);
  if (cuda_rv != Allen::error::success) {
    error() << "Allen exited with errorCode " << rv << endmsg;
    // how to exit a filter with failure?
  }
  bool filter = true;
  HostBuffers* buffer = m_host_buffers_manager->getBuffers(buf_idx);
  if (m_filterGEC.value()) {
    filter = static_cast<bool>(buffer->host_number_of_selected_events);
  }
  else if (m_filter_hlt1.value()) {
    filter = buffer->host_passing_event_list[0];
  }
  // Get line decisions from DecReports
  // First two words contain the TCK and taskID, then one word per HLT1 line
  LHCb::HltDecReports reports {};
  reports.setConfiguredTCK(m_tck);
  reports.reserve(buffer->host_number_of_lines);
  uint32_t dec_mask = HltDecReport::decReportMasks::decisionMask;
  for (unsigned int i = 0; i < buffer->host_number_of_lines; i++) {
    const uint32_t line_report = buffer->host_dec_reports[2 + i];
    const bool dec = line_report & dec_mask;
    const std::string modified_name = m_line_names[i] + "Decision";
    m_hlt1_line_rates[i].buffer() += int(dec);
    // Note: the line index in a DecReport cannot be zero -> start at 1
    const int dec_rep_index = i + 1;
    if (msgLevel(MSG::VERBOSE)) {
      verbose() << "Adding Allen line " << dec_rep_index << " with name " << modified_name
                << " to HltDecReport with decision " << int(dec) << endmsg;
    }
    reports.insert(modified_name, {dec, 0, 0, 0, dec_rep_index}).ignore(/* AUTOMATICALLY ADDED FOR gaudi/Gaudi!763 */);
  }
  if (msgLevel(MSG::DEBUG)) debug() << "Event selected by Allen: " << unsigned(filter) << endmsg;
  return std::make_tuple(filter, *buffer, reports);
}
