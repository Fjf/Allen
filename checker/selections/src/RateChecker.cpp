/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "RateChecker.h"
#include "SelectionsEventModel.cuh"

double binomial_error(int n, int k) { return 1. / n * std::sqrt(1. * k * (1. - 1. * k / n)); }

void RateChecker::accumulate(
  const std::vector<std::string>& names_of_lines,
  const std::vector<Allen::bool_as_char_t<bool>>& selections,
  const std::vector<unsigned>& selections_offsets,
  const unsigned number_of_events)
{
  std::lock_guard<std::mutex> guard(m_mutex);
  const auto number_of_lines = names_of_lines.size();
  if (!m_counters.size()) {
    m_line_names = names_of_lines;
    m_counters = std::vector<unsigned>(number_of_lines, 0);
  }

  Selections::ConstSelections sels {
    reinterpret_cast<const bool*>(selections.data()), selections_offsets.data(), number_of_events};

  for (auto i = 0u; i < number_of_events; ++i) {
    bool any_line_fired = false;
    for (auto j = 0u; j < number_of_lines; ++j) {
      auto decs = sels.get_span(j, i);
      for (auto k = 0u; k < decs.size(); ++k) {
        if (decs[k]) {
          ++m_counters[j];
          any_line_fired = true;
          break;
        }
      }
    }
    if (any_line_fired) {
      ++m_tot;
    }
  }
}

void RateChecker::report(const size_t requested_events) const
{
  // Assume 30 MHz input rate.
  const double in_rate = 30000.0;
  size_t longest_string = 10;
  for (const auto& line_name : m_line_names) {
    if (line_name.length() > longest_string) {
      longest_string = line_name.length();
    }
  }

  for (unsigned i_line = 0; i_line < m_line_names.size(); i_line++) {
    std::printf("%s:", m_line_names[i_line].c_str());
    for (unsigned i = 0; i < longest_string - m_line_names[i_line].length(); ++i) {
      std::printf(" ");
    }

    std::printf(
      " %6i/%6lu, (%8.2f +/- %8.2f) kHz\n",
      m_counters[i_line],
      requested_events,
      1. * m_counters[i_line] / requested_events * in_rate,
      binomial_error(requested_events, m_counters[i_line]) * in_rate);
  }

  std::printf("Inclusive:");
  for (unsigned i = 0; i < longest_string - 9; ++i) {
    std::printf(" ");
  }

  std::printf(
    " %6i/%6lu, (%8.2f +/- %8.2f) kHz\n",
    m_tot,
    requested_events,
    1. * m_tot / requested_events * in_rate,
    binomial_error(requested_events, m_tot) * in_rate);

  std::printf("\n");
}