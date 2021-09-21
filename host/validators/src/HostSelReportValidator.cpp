/*****************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "HostSelReportValidator.h"
#include "SelReportChecker.h"

void host_sel_report_validator::host_sel_report_validator_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  HostBuffers&,
  const Allen::Context&) const
{
  const auto sel_reports = make_vector<dev_sel_reports_t>(arguments);
  const auto sel_report_offsets = make_vector<dev_sel_report_offsets_t>(arguments);
  const char* line_names_char = data<host_names_of_lines_t>(arguments);
  const std::string line_names_str = line_names_char;
  std::vector<std::string> line_names;
  std::stringstream data(line_names_str);
  std::string line_name;
  while (std::getline(data, line_name, ',')) {
    line_names.push_back(line_name);
  }

  auto& checker = runtime_options.checker_invoker->checker<SelReportChecker>(name());
  checker.accumulate(
    line_names, sel_reports.data(), sel_report_offsets.data(), first<host_number_of_events_t>(arguments));
}