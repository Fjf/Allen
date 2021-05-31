/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <Common.h>
#include <InputTools.h>
#include <PV_Definitions.cuh>
#include <patPV_Definitions.cuh>
#include <CheckerTypes.h>
#include <CheckerInvoker.h>
#include <MCEvent.h>
#include <MCVertex.h>
#include <algorithm>
#include <mutex>
#include <Argument.cuh>

// configuration for PV checker -> check values
static constexpr int nTracksToBeRecble = 4;
static constexpr double dzIsolated = 10.; // mm
static constexpr bool matchByTracks = false;

class PVCheckerHistos;

class PVChecker : public Checker::BaseChecker {
public:
  PVChecker(CheckerInvoker const* invoker, std::string const& root_file, std::string const& name);

  virtual ~PVChecker();

  void accumulate(
    MCEvents const& mc_events,
    gsl::span<const PV::Vertex> rec_vertex,
    gsl::span<const unsigned> number_of_vertex,
    gsl::span<const mask_t> event_list);

  void report(size_t n_events) const override;

private:
  std::mutex m_mutex;
  PVCheckerHistos* m_histos = nullptr;

  size_t passed = 0;

  // counters for efficiencies/fake rate
  int sum_nMCPV = 0;
  int sum_nRecMCPV = 0;
  int sum_nMCPV_isol = 0;
  int sum_nRecMCPV_isol = 0;
  int sum_nMCPV_close = 0;
  int sum_nRecMCPV_close = 0;
  int sum_nFalsePV = 0;
  int sum_nFalsePV_real = 0;
  int sum_clones = 0;
  int sum_norm_clones = 0;
};

void match_mc_vertex_by_distance(int ipv, std::vector<RecPVInfo>& rinfo, std::vector<MCPVInfo>& mcpvvec);

void printRat(std::string mes, int a, int b);

std::vector<MCPVInfo>::iterator closestMCPV(std::vector<MCPVInfo>& rblemcpv, std::vector<MCPVInfo>::iterator& itmc);
