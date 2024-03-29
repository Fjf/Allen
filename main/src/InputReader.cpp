/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <gsl/gsl>
#include <InputReader.h>
#include <boost/algorithm/string.hpp>
#include "InputTools.h"
#include "Tools.h"

namespace {
  using std::make_pair;
}

Reader::Reader(const std::string& folder_name) : folder_name(folder_name)
{
  if (!exists_test(folder_name)) {
    throw StrException("Folder " + folder_name + " does not exist.");
  }
}

std::vector<char> GeometryReader::read_geometry(const std::string& filename) const
{
  std::vector<char> geometry;
  ::read_geometry(folder_name + "/" + filename, geometry);
  return geometry;
}

CatboostModelReader::CatboostModelReader(const std::string& file_name)
{
  if (!exists_test(file_name)) {
    throw StrException("Catboost model file " + file_name + " does not exist.");
  }
  std::ifstream i(file_name);
  nlohmann::json j;
  i >> j;
  m_num_features = j["features_info"]["float_features"].size();
  m_num_trees = j["oblivious_trees"].size();
  m_tree_offsets.push_back(0);
  m_leaf_offsets.push_back(0);
  for (nlohmann::json::iterator it = j["oblivious_trees"].begin(); it != j["oblivious_trees"].end(); ++it) {
    nlohmann::json tree(*it);
    std::vector<float> tree_split_borders;
    std::vector<int> tree_split_features;
    m_leaf_values.insert(std::end(m_leaf_values), std::begin(tree["leaf_values"]), std::end(tree["leaf_values"]));
    m_tree_depths.push_back(tree["splits"].size());
    m_tree_offsets.push_back(m_tree_offsets.back() + m_tree_depths.back());
    m_leaf_offsets.push_back(m_leaf_offsets.back() + (1 << m_tree_depths.back()));
    for (nlohmann::json::iterator it_spl = tree["splits"].begin(); it_spl != tree["splits"].end(); ++it_spl) {
      nlohmann::json split(*it_spl);
      tree_split_borders.push_back(split["border"]);
      tree_split_features.push_back(split["float_feature_index"]);
    }
    m_split_border.insert(std::end(m_split_border), std::begin(tree_split_borders), std::end(tree_split_borders));
    m_split_feature.insert(std::end(m_split_feature), std::begin(tree_split_features), std::end(tree_split_features));
  }
}

TwoTrackMVAModelReader::TwoTrackMVAModelReader(const std::string& file_name)
{
  if (!exists_test(file_name)) {
    throw StrException("Two Track MVA model file " + file_name + " does not exist.");
  }
  std::ifstream i(file_name);
  nlohmann::json j;
  i >> j;

  std::map<int, int> layer_sizes {};
  std::map<int, std::vector<float>> biases {};
  std::map<int, std::vector<float>> weights {};

  layer_sizes[0] = 4; // input size hard coded
  for (auto el = j.begin(); el != j.end(); ++el) {
    // map is sorted
    std::vector<std::string> tokens;
    boost::split(tokens, el.key(), boost::is_any_of("."));
    if (el.key().find("bias") != std::string::npos) {
      int layer_n = std::stoi(tokens[tokens.size() - 2]) + 1;
      layer_sizes[layer_n] = el.value().size();
      biases[layer_n] = std::vector<float> {};
      for (auto el_bias : el.value()) {
        biases[layer_n].push_back(el_bias);
      }
    }
    else if (el.key().find("weight") != std::string::npos) {
      int layer_n = std::stoi(tokens[tokens.size() - 2]) + 1;
      weights[layer_n] = std::vector<float> {};
      for (auto weight_row : el.value()) {
        for (auto weight_el : weight_row) {
          weights[layer_n].push_back(weight_el);
        }
      }
    }
  }
  for (auto el : layer_sizes) {
    m_layer_sizes.push_back(el.second);
  }

  for (auto el : weights) {
    for (auto el_w : el.second) {
      m_weights.push_back(el_w);
    }
  }
  for (auto el : biases) {
    for (auto el_b : el.second) {
      m_biases.push_back(el_b);
    }
  }

  // hardcode monotone constraints for now
  // monotone constraints define in which features we want to be monotonic:
  // 0 -> -lambda <= df/dx <= lambda
  // 1 -> 0 <= df/dx <= 2*lambda
  // -1 -> -2*lambda <= df/dx <= 0
  m_monotone_constraints = std::vector<float> {1, 1, 0, 1};
  m_lambda = j["sigmanet.sigma"][0];
  m_nominal_cut = j["nominal_cut"];
  m_n_layers = m_layer_sizes.size();
}

ConfigurationReader::ConfigurationReader(std::string_view configuration)
{
  nlohmann::json j = nlohmann::json::parse(configuration);
  for (auto& el : j.items()) {
    std::string component = el.key();
    if (component == "sequence") {
      m_sequence = {};
      for (auto& el2 : el.value().items()) {
        if (el2.key() == "configured_algorithms") {
          m_sequence[el2.key()] = el2.value();
          m_configured_sequence.configured_algorithms = ParsedSequence::to_configured<ConfiguredAlgorithm>(
            el2.value().get<ParsedSequence::configured_algorithm_parse_t>());
        }
        else if (el2.key() == "configured_arguments") {
          m_sequence[el2.key()] = el2.value();
          m_configured_sequence.configured_arguments = ParsedSequence::to_configured<ConfiguredArgument>(
            el2.value().get<ParsedSequence::configured_argument_parse_t>());
        }
        else if (el2.key() == "configured_sequence_arguments") {
          m_sequence[el2.key()] = el2.value();
          m_configured_sequence.configured_algorithm_arguments =
            ParsedSequence::to_configured<ConfiguredAlgorithmArguments>(
              el2.value().get<ParsedSequence::configured_algorithm_argument_parse_t>());
        }
        else if (el2.key() == "argument_dependencies") {
          m_sequence[el2.key()] = el2.value();
          m_configured_sequence.argument_dependencies =
            el2.value().get<ParsedSequence::argument_dependencies_parse_t>();
        }
      }
    }
    else {
      for (auto& el2 : el.value().items()) {
        std::string property = el2.key();
        m_params[component][property] = el2.value();
      }
    }
  }

  if (logger::verbosity() >= logger::verbose) {
    for (auto it = m_params.begin(); it != m_params.end(); ++it) {
      for (auto it2 = (*it).second.begin(); it2 != (*it).second.end(); ++it2) {
        verbose_cout << (*it).first << ":" << (*it2).first << ":" << (*it2).second << std::endl;
      }
    }
  }
}

std::map<std::string, nlohmann::json> ConfigurationReader::get_sequence() const { return m_sequence; }

void ConfigurationReader::save(std::string file_name)
{
  using json_float = nlohmann::basic_json<std::map, std::vector, std::string, bool, std::int32_t, std::uint32_t, float>;
  json_float j;
  for (auto [alg, props] : m_params) {
    for (auto [k, v] : props) {
      j[alg][k] = v;
    }
  }
  std::ofstream o(file_name);
  o << std::setw(4) << j;
  o.close();
}

std::unordered_set<BankTypes> ConfigurationReader::configured_bank_types() const
{
  // Bank types
  std::unordered_set<BankTypes> bank_types = {BankTypes::ODIN};

  std::vector<std::string> provider_algorithms;
  for (const auto& alg : m_configured_sequence.configured_algorithms) {
    if (alg.scope == "ProviderAlgorithm") {
      provider_algorithms.push_back(alg.name);
    }
  }

  for (const auto& provider_alg : provider_algorithms) {
    const auto props = m_params.at(provider_alg);
    auto it_type = props.find("bank_type");
    auto it_empty = props.find("empty");
    if (it_type != props.end()) {
      auto type = it_type->second;
      auto const bt = ::bank_type(type);
      if (bt == BankTypes::Unknown) {
        error_cout << "Unknown bank type " << type << " requested.\n";
      }
      else if (it_empty == props.end() || !it_empty->second.get<bool>()) {
        bank_types.emplace(bt);
      }
    }
  }

  return bank_types;
}
