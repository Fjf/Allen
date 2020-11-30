/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <tuple>
#include "../../device/selections/lines/include/D2KKLine.cuh"
#include "../../device/selections/lines/include/ODINEventTypeLine.cuh"
#include "../../device/selections/lines/include/TrackMVALine.cuh"
#include "../../device/selections/lines/include/LowPtMuonLine.cuh"
#include "../../device/selections/lines/include/DiMuonMassLine.cuh"
#include "../../device/selections/lines/include/DiMuonSoftLine.cuh"
#include "../../device/selections/lines/include/TrackMuonMVALine.cuh"
#include "../../device/selections/lines/include/D2PiPiLine.cuh"
#include "../../device/selections/lines/include/D2KPiLine.cuh"
#include "../../device/selections/lines/include/VeloMicroBiasLine.cuh"
#include "../../device/selections/lines/include/LowPtDiMuonLine.cuh"
#include "../../device/selections/lines/include/BeamCrossingLine.cuh"
#include "../../device/selections/lines/include/SingleHighPtMuonLine.cuh"
#include "../../device/selections/lines/include/PassthroughLine.cuh"
#include "../../device/selections/lines/include/TwoTrackMVALine.cuh"

struct Hlt1TrackMVA__dev_decisions_t : track_mva_line::Parameters::dev_decisions_t {
  using type = track_mva_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TrackMVA__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1TwoTrackMVA__dev_decisions_t : two_track_mva_line::Parameters::dev_decisions_t {
  using type = two_track_mva_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TwoTrackMVA__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1NoBeam__dev_decisions_t : beam_crossing_line::Parameters::dev_decisions_t {
  using type = beam_crossing_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1NoBeam__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BeamOne__dev_decisions_t : beam_crossing_line::Parameters::dev_decisions_t {
  using type = beam_crossing_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BeamOne__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BeamTwo__dev_decisions_t : beam_crossing_line::Parameters::dev_decisions_t {
  using type = beam_crossing_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BeamTwo__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BothBeams__dev_decisions_t : beam_crossing_line::Parameters::dev_decisions_t {
  using type = beam_crossing_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BothBeams__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1VeloMicroBias__dev_decisions_t : velo_micro_bias_line::Parameters::dev_decisions_t {
  using type = velo_micro_bias_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1VeloMicroBias__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1ODINLumi__dev_decisions_t : odin_event_type_line::Parameters::dev_decisions_t {
  using type = odin_event_type_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1ODINLumi__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1ODINNoBias__dev_decisions_t : odin_event_type_line::Parameters::dev_decisions_t {
  using type = odin_event_type_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1ODINNoBias__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1SingleHighPtMuon__dev_decisions_t : single_high_pt_muon_line::Parameters::dev_decisions_t {
  using type = single_high_pt_muon_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1SingleHighPtMuon__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1LowPtMuon__dev_decisions_t : low_pt_muon_line::Parameters::dev_decisions_t {
  using type = low_pt_muon_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1LowPtMuon__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2KK__dev_decisions_t : d2kk_line::Parameters::dev_decisions_t {
  using type = d2kk_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2KK__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2KPi__dev_decisions_t : d2kpi_line::Parameters::dev_decisions_t {
  using type = d2kpi_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2KPi__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2PiPi__dev_decisions_t : d2pipi_line::Parameters::dev_decisions_t {
  using type = d2pipi_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2PiPi__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonHighMass__dev_decisions_t : di_muon_mass_line::Parameters::dev_decisions_t {
  using type = di_muon_mass_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonHighMass__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonLowMass__dev_decisions_t : di_muon_mass_line::Parameters::dev_decisions_t {
  using type = di_muon_mass_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonLowMass__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonSoft__dev_decisions_t : di_muon_soft_line::Parameters::dev_decisions_t {
  using type = di_muon_soft_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonSoft__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1LowPtDiMuon__dev_decisions_t : low_pt_di_muon_line::Parameters::dev_decisions_t {
  using type = low_pt_di_muon_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1LowPtDiMuon__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1TrackMuonMVA__dev_decisions_t : track_muon_mva_line::Parameters::dev_decisions_t {
  using type = track_muon_mva_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TrackMuonMVA__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1GECPassthrough__dev_decisions_t : passthrough_line::Parameters::dev_decisions_t {
  using type = passthrough_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1GECPassthrough__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1Passthrough__dev_decisions_t : passthrough_line::Parameters::dev_decisions_t {
  using type = passthrough_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1Passthrough__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1TrackMVA__dev_decisions_offsets_t : track_mva_line::Parameters::dev_decisions_offsets_t {
  using type = track_mva_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TrackMVA__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1TwoTrackMVA__dev_decisions_offsets_t : two_track_mva_line::Parameters::dev_decisions_offsets_t {
  using type = two_track_mva_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TwoTrackMVA__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1NoBeam__dev_decisions_offsets_t : beam_crossing_line::Parameters::dev_decisions_offsets_t {
  using type = beam_crossing_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1NoBeam__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BeamOne__dev_decisions_offsets_t : beam_crossing_line::Parameters::dev_decisions_offsets_t {
  using type = beam_crossing_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BeamOne__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BeamTwo__dev_decisions_offsets_t : beam_crossing_line::Parameters::dev_decisions_offsets_t {
  using type = beam_crossing_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BeamTwo__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BothBeams__dev_decisions_offsets_t : beam_crossing_line::Parameters::dev_decisions_offsets_t {
  using type = beam_crossing_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BothBeams__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1VeloMicroBias__dev_decisions_offsets_t : velo_micro_bias_line::Parameters::dev_decisions_offsets_t {
  using type = velo_micro_bias_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1VeloMicroBias__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1ODINLumi__dev_decisions_offsets_t : odin_event_type_line::Parameters::dev_decisions_offsets_t {
  using type = odin_event_type_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1ODINLumi__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1ODINNoBias__dev_decisions_offsets_t : odin_event_type_line::Parameters::dev_decisions_offsets_t {
  using type = odin_event_type_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1ODINNoBias__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1SingleHighPtMuon__dev_decisions_offsets_t : single_high_pt_muon_line::Parameters::dev_decisions_offsets_t {
  using type = single_high_pt_muon_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1SingleHighPtMuon__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1LowPtMuon__dev_decisions_offsets_t : low_pt_muon_line::Parameters::dev_decisions_offsets_t {
  using type = low_pt_muon_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1LowPtMuon__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2KK__dev_decisions_offsets_t : d2kk_line::Parameters::dev_decisions_offsets_t {
  using type = d2kk_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2KK__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2KPi__dev_decisions_offsets_t : d2kpi_line::Parameters::dev_decisions_offsets_t {
  using type = d2kpi_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2KPi__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2PiPi__dev_decisions_offsets_t : d2pipi_line::Parameters::dev_decisions_offsets_t {
  using type = d2pipi_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2PiPi__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonHighMass__dev_decisions_offsets_t : di_muon_mass_line::Parameters::dev_decisions_offsets_t {
  using type = di_muon_mass_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonHighMass__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonLowMass__dev_decisions_offsets_t : di_muon_mass_line::Parameters::dev_decisions_offsets_t {
  using type = di_muon_mass_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonLowMass__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonSoft__dev_decisions_offsets_t : di_muon_soft_line::Parameters::dev_decisions_offsets_t {
  using type = di_muon_soft_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonSoft__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1LowPtDiMuon__dev_decisions_offsets_t : low_pt_di_muon_line::Parameters::dev_decisions_offsets_t {
  using type = low_pt_di_muon_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1LowPtDiMuon__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1TrackMuonMVA__dev_decisions_offsets_t : track_muon_mva_line::Parameters::dev_decisions_offsets_t {
  using type = track_muon_mva_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TrackMuonMVA__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1GECPassthrough__dev_decisions_offsets_t : passthrough_line::Parameters::dev_decisions_offsets_t {
  using type = passthrough_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1GECPassthrough__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1Passthrough__dev_decisions_offsets_t : passthrough_line::Parameters::dev_decisions_offsets_t {
  using type = passthrough_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1Passthrough__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1TrackMVA__host_post_scaler_t : track_mva_line::Parameters::host_post_scaler_t {
  using type = track_mva_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TrackMVA__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1TwoTrackMVA__host_post_scaler_t : two_track_mva_line::Parameters::host_post_scaler_t {
  using type = two_track_mva_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TwoTrackMVA__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1NoBeam__host_post_scaler_t : beam_crossing_line::Parameters::host_post_scaler_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1NoBeam__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BeamOne__host_post_scaler_t : beam_crossing_line::Parameters::host_post_scaler_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BeamOne__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BeamTwo__host_post_scaler_t : beam_crossing_line::Parameters::host_post_scaler_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BeamTwo__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BothBeams__host_post_scaler_t : beam_crossing_line::Parameters::host_post_scaler_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BothBeams__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1VeloMicroBias__host_post_scaler_t : velo_micro_bias_line::Parameters::host_post_scaler_t {
  using type = velo_micro_bias_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1VeloMicroBias__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1ODINLumi__host_post_scaler_t : odin_event_type_line::Parameters::host_post_scaler_t {
  using type = odin_event_type_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1ODINLumi__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1ODINNoBias__host_post_scaler_t : odin_event_type_line::Parameters::host_post_scaler_t {
  using type = odin_event_type_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1ODINNoBias__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1SingleHighPtMuon__host_post_scaler_t : single_high_pt_muon_line::Parameters::host_post_scaler_t {
  using type = single_high_pt_muon_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1SingleHighPtMuon__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1LowPtMuon__host_post_scaler_t : low_pt_muon_line::Parameters::host_post_scaler_t {
  using type = low_pt_muon_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1LowPtMuon__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2KK__host_post_scaler_t : d2kk_line::Parameters::host_post_scaler_t {
  using type = d2kk_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2KK__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2KPi__host_post_scaler_t : d2kpi_line::Parameters::host_post_scaler_t {
  using type = d2kpi_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2KPi__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2PiPi__host_post_scaler_t : d2pipi_line::Parameters::host_post_scaler_t {
  using type = d2pipi_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2PiPi__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonHighMass__host_post_scaler_t : di_muon_mass_line::Parameters::host_post_scaler_t {
  using type = di_muon_mass_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonHighMass__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonLowMass__host_post_scaler_t : di_muon_mass_line::Parameters::host_post_scaler_t {
  using type = di_muon_mass_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonLowMass__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonSoft__host_post_scaler_t : di_muon_soft_line::Parameters::host_post_scaler_t {
  using type = di_muon_soft_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonSoft__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1LowPtDiMuon__host_post_scaler_t : low_pt_di_muon_line::Parameters::host_post_scaler_t {
  using type = low_pt_di_muon_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1LowPtDiMuon__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1TrackMuonMVA__host_post_scaler_t : track_muon_mva_line::Parameters::host_post_scaler_t {
  using type = track_muon_mva_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TrackMuonMVA__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1GECPassthrough__host_post_scaler_t : passthrough_line::Parameters::host_post_scaler_t {
  using type = passthrough_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1GECPassthrough__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1Passthrough__host_post_scaler_t : passthrough_line::Parameters::host_post_scaler_t {
  using type = passthrough_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1Passthrough__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1TrackMVA__host_post_scaler_hash_t : track_mva_line::Parameters::host_post_scaler_hash_t {
  using type = track_mva_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TrackMVA__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1TwoTrackMVA__host_post_scaler_hash_t : two_track_mva_line::Parameters::host_post_scaler_hash_t {
  using type = two_track_mva_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TwoTrackMVA__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1NoBeam__host_post_scaler_hash_t : beam_crossing_line::Parameters::host_post_scaler_hash_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1NoBeam__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BeamOne__host_post_scaler_hash_t : beam_crossing_line::Parameters::host_post_scaler_hash_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BeamOne__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BeamTwo__host_post_scaler_hash_t : beam_crossing_line::Parameters::host_post_scaler_hash_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BeamTwo__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1BothBeams__host_post_scaler_hash_t : beam_crossing_line::Parameters::host_post_scaler_hash_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1BothBeams__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1VeloMicroBias__host_post_scaler_hash_t : velo_micro_bias_line::Parameters::host_post_scaler_hash_t {
  using type = velo_micro_bias_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1VeloMicroBias__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1ODINLumi__host_post_scaler_hash_t : odin_event_type_line::Parameters::host_post_scaler_hash_t {
  using type = odin_event_type_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1ODINLumi__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1ODINNoBias__host_post_scaler_hash_t : odin_event_type_line::Parameters::host_post_scaler_hash_t {
  using type = odin_event_type_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1ODINNoBias__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1SingleHighPtMuon__host_post_scaler_hash_t : single_high_pt_muon_line::Parameters::host_post_scaler_hash_t {
  using type = single_high_pt_muon_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1SingleHighPtMuon__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1LowPtMuon__host_post_scaler_hash_t : low_pt_muon_line::Parameters::host_post_scaler_hash_t {
  using type = low_pt_muon_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1LowPtMuon__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2KK__host_post_scaler_hash_t : d2kk_line::Parameters::host_post_scaler_hash_t {
  using type = d2kk_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2KK__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2KPi__host_post_scaler_hash_t : d2kpi_line::Parameters::host_post_scaler_hash_t {
  using type = d2kpi_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2KPi__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1D2PiPi__host_post_scaler_hash_t : d2pipi_line::Parameters::host_post_scaler_hash_t {
  using type = d2pipi_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1D2PiPi__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonHighMass__host_post_scaler_hash_t : di_muon_mass_line::Parameters::host_post_scaler_hash_t {
  using type = di_muon_mass_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonHighMass__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonLowMass__host_post_scaler_hash_t : di_muon_mass_line::Parameters::host_post_scaler_hash_t {
  using type = di_muon_mass_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonLowMass__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1DiMuonSoft__host_post_scaler_hash_t : di_muon_soft_line::Parameters::host_post_scaler_hash_t {
  using type = di_muon_soft_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1DiMuonSoft__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1LowPtDiMuon__host_post_scaler_hash_t : low_pt_di_muon_line::Parameters::host_post_scaler_hash_t {
  using type = low_pt_di_muon_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1LowPtDiMuon__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1TrackMuonMVA__host_post_scaler_hash_t : track_muon_mva_line::Parameters::host_post_scaler_hash_t {
  using type = track_muon_mva_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1TrackMuonMVA__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1GECPassthrough__host_post_scaler_hash_t : passthrough_line::Parameters::host_post_scaler_hash_t {
  using type = passthrough_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1GECPassthrough__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct Hlt1Passthrough__host_post_scaler_hash_t : passthrough_line::Parameters::host_post_scaler_hash_t {
  using type = passthrough_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "Hlt1Passthrough__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};

namespace gather_selections {
  namespace dev_input_selections_t {
    using tuple_t = std::tuple<
      Hlt1TrackMVA__dev_decisions_t,
      Hlt1TwoTrackMVA__dev_decisions_t,
      Hlt1NoBeam__dev_decisions_t,
      Hlt1BeamOne__dev_decisions_t,
      Hlt1BeamTwo__dev_decisions_t,
      Hlt1BothBeams__dev_decisions_t,
      Hlt1VeloMicroBias__dev_decisions_t,
      Hlt1ODINLumi__dev_decisions_t,
      Hlt1ODINNoBias__dev_decisions_t,
      Hlt1SingleHighPtMuon__dev_decisions_t,
      Hlt1LowPtMuon__dev_decisions_t,
      Hlt1D2KK__dev_decisions_t,
      Hlt1D2KPi__dev_decisions_t,
      Hlt1D2PiPi__dev_decisions_t,
      Hlt1DiMuonHighMass__dev_decisions_t,
      Hlt1DiMuonLowMass__dev_decisions_t,
      Hlt1DiMuonSoft__dev_decisions_t,
      Hlt1LowPtDiMuon__dev_decisions_t,
      Hlt1TrackMuonMVA__dev_decisions_t,
      Hlt1GECPassthrough__dev_decisions_t,
      Hlt1Passthrough__dev_decisions_t>;
  }
} // namespace gather_selections
namespace gather_selections {
  namespace dev_input_selections_offsets_t {
    using tuple_t = std::tuple<
      Hlt1TrackMVA__dev_decisions_offsets_t,
      Hlt1TwoTrackMVA__dev_decisions_offsets_t,
      Hlt1NoBeam__dev_decisions_offsets_t,
      Hlt1BeamOne__dev_decisions_offsets_t,
      Hlt1BeamTwo__dev_decisions_offsets_t,
      Hlt1BothBeams__dev_decisions_offsets_t,
      Hlt1VeloMicroBias__dev_decisions_offsets_t,
      Hlt1ODINLumi__dev_decisions_offsets_t,
      Hlt1ODINNoBias__dev_decisions_offsets_t,
      Hlt1SingleHighPtMuon__dev_decisions_offsets_t,
      Hlt1LowPtMuon__dev_decisions_offsets_t,
      Hlt1D2KK__dev_decisions_offsets_t,
      Hlt1D2KPi__dev_decisions_offsets_t,
      Hlt1D2PiPi__dev_decisions_offsets_t,
      Hlt1DiMuonHighMass__dev_decisions_offsets_t,
      Hlt1DiMuonLowMass__dev_decisions_offsets_t,
      Hlt1DiMuonSoft__dev_decisions_offsets_t,
      Hlt1LowPtDiMuon__dev_decisions_offsets_t,
      Hlt1TrackMuonMVA__dev_decisions_offsets_t,
      Hlt1GECPassthrough__dev_decisions_offsets_t,
      Hlt1Passthrough__dev_decisions_offsets_t>;
  }
} // namespace gather_selections
namespace gather_selections {
  namespace host_input_post_scale_factors_t {
    using tuple_t = std::tuple<
      Hlt1TrackMVA__host_post_scaler_t,
      Hlt1TwoTrackMVA__host_post_scaler_t,
      Hlt1NoBeam__host_post_scaler_t,
      Hlt1BeamOne__host_post_scaler_t,
      Hlt1BeamTwo__host_post_scaler_t,
      Hlt1BothBeams__host_post_scaler_t,
      Hlt1VeloMicroBias__host_post_scaler_t,
      Hlt1ODINLumi__host_post_scaler_t,
      Hlt1ODINNoBias__host_post_scaler_t,
      Hlt1SingleHighPtMuon__host_post_scaler_t,
      Hlt1LowPtMuon__host_post_scaler_t,
      Hlt1D2KK__host_post_scaler_t,
      Hlt1D2KPi__host_post_scaler_t,
      Hlt1D2PiPi__host_post_scaler_t,
      Hlt1DiMuonHighMass__host_post_scaler_t,
      Hlt1DiMuonLowMass__host_post_scaler_t,
      Hlt1DiMuonSoft__host_post_scaler_t,
      Hlt1LowPtDiMuon__host_post_scaler_t,
      Hlt1TrackMuonMVA__host_post_scaler_t,
      Hlt1GECPassthrough__host_post_scaler_t,
      Hlt1Passthrough__host_post_scaler_t>;
  }
} // namespace gather_selections
namespace gather_selections {
  namespace host_input_post_scale_hashes_t {
    using tuple_t = std::tuple<
      Hlt1TrackMVA__host_post_scaler_hash_t,
      Hlt1TwoTrackMVA__host_post_scaler_hash_t,
      Hlt1NoBeam__host_post_scaler_hash_t,
      Hlt1BeamOne__host_post_scaler_hash_t,
      Hlt1BeamTwo__host_post_scaler_hash_t,
      Hlt1BothBeams__host_post_scaler_hash_t,
      Hlt1VeloMicroBias__host_post_scaler_hash_t,
      Hlt1ODINLumi__host_post_scaler_hash_t,
      Hlt1ODINNoBias__host_post_scaler_hash_t,
      Hlt1SingleHighPtMuon__host_post_scaler_hash_t,
      Hlt1LowPtMuon__host_post_scaler_hash_t,
      Hlt1D2KK__host_post_scaler_hash_t,
      Hlt1D2KPi__host_post_scaler_hash_t,
      Hlt1D2PiPi__host_post_scaler_hash_t,
      Hlt1DiMuonHighMass__host_post_scaler_hash_t,
      Hlt1DiMuonLowMass__host_post_scaler_hash_t,
      Hlt1DiMuonSoft__host_post_scaler_hash_t,
      Hlt1LowPtDiMuon__host_post_scaler_hash_t,
      Hlt1TrackMuonMVA__host_post_scaler_hash_t,
      Hlt1GECPassthrough__host_post_scaler_hash_t,
      Hlt1Passthrough__host_post_scaler_hash_t>;
  }
} // namespace gather_selections