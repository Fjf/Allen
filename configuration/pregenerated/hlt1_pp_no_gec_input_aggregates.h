/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#pragma once

#include <tuple>
#include "../../device/selections/lines/include/D2KKLine.cuh"
#include "../../device/selections/lines/include/D2PiPiLine.cuh"
#include "../../device/selections/lines/include/D2KPiLine.cuh"
#include "../../device/selections/lines/include/SingleHighPtMuonLine.cuh"
#include "../../device/selections/lines/include/TrackMVALine.cuh"
#include "../../device/selections/lines/include/TrackMuonMVALine.cuh"
#include "../../device/selections/lines/include/DiMuonMassLine.cuh"
#include "../../device/selections/lines/include/VeloMicroBiasLine.cuh"
#include "../../device/selections/lines/include/DiMuonSoftLine.cuh"
#include "../../device/selections/lines/include/LowPtMuonLine.cuh"
#include "../../device/selections/lines/include/LowPtDiMuonLine.cuh"
#include "../../device/selections/lines/include/ODINEventTypeLine.cuh"
#include "../../device/selections/lines/include/BeamCrossingLine.cuh"
#include "../../device/selections/lines/include/PassthroughLine.cuh"
#include "../../device/selections/lines/include/TwoTrackMVALine.cuh"

struct track_mva_line__dev_decisions_t : track_mva_line::Parameters::dev_decisions_t {
  using type = track_mva_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "track_mva_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct two_track_mva_line__dev_decisions_t : two_track_mva_line::Parameters::dev_decisions_t {
  using type = two_track_mva_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "two_track_mva_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct no_beam_line__dev_decisions_t : beam_crossing_line::Parameters::dev_decisions_t {
  using type = beam_crossing_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "no_beam_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct beam_one_line__dev_decisions_t : beam_crossing_line::Parameters::dev_decisions_t {
  using type = beam_crossing_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "beam_one_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct beam_two_line__dev_decisions_t : beam_crossing_line::Parameters::dev_decisions_t {
  using type = beam_crossing_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "beam_two_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct both_beams_line__dev_decisions_t : beam_crossing_line::Parameters::dev_decisions_t {
  using type = beam_crossing_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "both_beams_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_micro_bias_line__dev_decisions_t : velo_micro_bias_line::Parameters::dev_decisions_t {
  using type = velo_micro_bias_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_micro_bias_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct odin_lumi_line__dev_decisions_t : odin_event_type_line::Parameters::dev_decisions_t {
  using type = odin_event_type_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "odin_lumi_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct odin_no_bias__dev_decisions_t : odin_event_type_line::Parameters::dev_decisions_t {
  using type = odin_event_type_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "odin_no_bias__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct single_high_pt_muon_line__dev_decisions_t : single_high_pt_muon_line::Parameters::dev_decisions_t {
  using type = single_high_pt_muon_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "single_high_pt_muon_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct low_pt_muon_line__dev_decisions_t : low_pt_muon_line::Parameters::dev_decisions_t {
  using type = low_pt_muon_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "low_pt_muon_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2kk_line__dev_decisions_t : d2kk_line::Parameters::dev_decisions_t {
  using type = d2kk_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2kk_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2kpi_line__dev_decisions_t : d2kpi_line::Parameters::dev_decisions_t {
  using type = d2kpi_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2kpi_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2pipi_line__dev_decisions_t : d2pipi_line::Parameters::dev_decisions_t {
  using type = d2pipi_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2pipi_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_high_mass_line__dev_decisions_t : di_muon_mass_line::Parameters::dev_decisions_t {
  using type = di_muon_mass_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_high_mass_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_low_mass_line__dev_decisions_t : di_muon_mass_line::Parameters::dev_decisions_t {
  using type = di_muon_mass_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_low_mass_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_soft_line__dev_decisions_t : di_muon_soft_line::Parameters::dev_decisions_t {
  using type = di_muon_soft_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_soft_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct low_pt_di_muon_line__dev_decisions_t : low_pt_di_muon_line::Parameters::dev_decisions_t {
  using type = low_pt_di_muon_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "low_pt_di_muon_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct track_muon_mva_line__dev_decisions_t : track_muon_mva_line::Parameters::dev_decisions_t {
  using type = track_muon_mva_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "track_muon_mva_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct gec_passthrough_line__dev_decisions_t : passthrough_line::Parameters::dev_decisions_t {
  using type = passthrough_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "gec_passthrough_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct passthrough_line__dev_decisions_t : passthrough_line::Parameters::dev_decisions_t {
  using type = passthrough_line::Parameters::dev_decisions_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "passthrough_line__dev_decisions_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct track_mva_line__dev_decisions_offsets_t : track_mva_line::Parameters::dev_decisions_offsets_t {
  using type = track_mva_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "track_mva_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct two_track_mva_line__dev_decisions_offsets_t : two_track_mva_line::Parameters::dev_decisions_offsets_t {
  using type = two_track_mva_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "two_track_mva_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct no_beam_line__dev_decisions_offsets_t : beam_crossing_line::Parameters::dev_decisions_offsets_t {
  using type = beam_crossing_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "no_beam_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct beam_one_line__dev_decisions_offsets_t : beam_crossing_line::Parameters::dev_decisions_offsets_t {
  using type = beam_crossing_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "beam_one_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct beam_two_line__dev_decisions_offsets_t : beam_crossing_line::Parameters::dev_decisions_offsets_t {
  using type = beam_crossing_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "beam_two_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct both_beams_line__dev_decisions_offsets_t : beam_crossing_line::Parameters::dev_decisions_offsets_t {
  using type = beam_crossing_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "both_beams_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_micro_bias_line__dev_decisions_offsets_t : velo_micro_bias_line::Parameters::dev_decisions_offsets_t {
  using type = velo_micro_bias_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_micro_bias_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct odin_lumi_line__dev_decisions_offsets_t : odin_event_type_line::Parameters::dev_decisions_offsets_t {
  using type = odin_event_type_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "odin_lumi_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct odin_no_bias__dev_decisions_offsets_t : odin_event_type_line::Parameters::dev_decisions_offsets_t {
  using type = odin_event_type_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "odin_no_bias__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct single_high_pt_muon_line__dev_decisions_offsets_t
  : single_high_pt_muon_line::Parameters::dev_decisions_offsets_t {
  using type = single_high_pt_muon_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "single_high_pt_muon_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct low_pt_muon_line__dev_decisions_offsets_t : low_pt_muon_line::Parameters::dev_decisions_offsets_t {
  using type = low_pt_muon_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "low_pt_muon_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2kk_line__dev_decisions_offsets_t : d2kk_line::Parameters::dev_decisions_offsets_t {
  using type = d2kk_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2kk_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2kpi_line__dev_decisions_offsets_t : d2kpi_line::Parameters::dev_decisions_offsets_t {
  using type = d2kpi_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2kpi_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2pipi_line__dev_decisions_offsets_t : d2pipi_line::Parameters::dev_decisions_offsets_t {
  using type = d2pipi_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2pipi_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_high_mass_line__dev_decisions_offsets_t : di_muon_mass_line::Parameters::dev_decisions_offsets_t {
  using type = di_muon_mass_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_high_mass_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_low_mass_line__dev_decisions_offsets_t : di_muon_mass_line::Parameters::dev_decisions_offsets_t {
  using type = di_muon_mass_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_low_mass_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_soft_line__dev_decisions_offsets_t : di_muon_soft_line::Parameters::dev_decisions_offsets_t {
  using type = di_muon_soft_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_soft_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct low_pt_di_muon_line__dev_decisions_offsets_t : low_pt_di_muon_line::Parameters::dev_decisions_offsets_t {
  using type = low_pt_di_muon_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "low_pt_di_muon_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct track_muon_mva_line__dev_decisions_offsets_t : track_muon_mva_line::Parameters::dev_decisions_offsets_t {
  using type = track_muon_mva_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "track_muon_mva_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct gec_passthrough_line__dev_decisions_offsets_t : passthrough_line::Parameters::dev_decisions_offsets_t {
  using type = passthrough_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "gec_passthrough_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct passthrough_line__dev_decisions_offsets_t : passthrough_line::Parameters::dev_decisions_offsets_t {
  using type = passthrough_line::Parameters::dev_decisions_offsets_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "passthrough_line__dev_decisions_offsets_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct track_mva_line__host_post_scaler_t : track_mva_line::Parameters::host_post_scaler_t {
  using type = track_mva_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "track_mva_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct two_track_mva_line__host_post_scaler_t : two_track_mva_line::Parameters::host_post_scaler_t {
  using type = two_track_mva_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "two_track_mva_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct no_beam_line__host_post_scaler_t : beam_crossing_line::Parameters::host_post_scaler_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "no_beam_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct beam_one_line__host_post_scaler_t : beam_crossing_line::Parameters::host_post_scaler_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "beam_one_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct beam_two_line__host_post_scaler_t : beam_crossing_line::Parameters::host_post_scaler_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "beam_two_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct both_beams_line__host_post_scaler_t : beam_crossing_line::Parameters::host_post_scaler_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "both_beams_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_micro_bias_line__host_post_scaler_t : velo_micro_bias_line::Parameters::host_post_scaler_t {
  using type = velo_micro_bias_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_micro_bias_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct odin_lumi_line__host_post_scaler_t : odin_event_type_line::Parameters::host_post_scaler_t {
  using type = odin_event_type_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "odin_lumi_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct odin_no_bias__host_post_scaler_t : odin_event_type_line::Parameters::host_post_scaler_t {
  using type = odin_event_type_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "odin_no_bias__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct single_high_pt_muon_line__host_post_scaler_t : single_high_pt_muon_line::Parameters::host_post_scaler_t {
  using type = single_high_pt_muon_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "single_high_pt_muon_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct low_pt_muon_line__host_post_scaler_t : low_pt_muon_line::Parameters::host_post_scaler_t {
  using type = low_pt_muon_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "low_pt_muon_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2kk_line__host_post_scaler_t : d2kk_line::Parameters::host_post_scaler_t {
  using type = d2kk_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2kk_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2kpi_line__host_post_scaler_t : d2kpi_line::Parameters::host_post_scaler_t {
  using type = d2kpi_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2kpi_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2pipi_line__host_post_scaler_t : d2pipi_line::Parameters::host_post_scaler_t {
  using type = d2pipi_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2pipi_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_high_mass_line__host_post_scaler_t : di_muon_mass_line::Parameters::host_post_scaler_t {
  using type = di_muon_mass_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_high_mass_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_low_mass_line__host_post_scaler_t : di_muon_mass_line::Parameters::host_post_scaler_t {
  using type = di_muon_mass_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_low_mass_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_soft_line__host_post_scaler_t : di_muon_soft_line::Parameters::host_post_scaler_t {
  using type = di_muon_soft_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_soft_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct low_pt_di_muon_line__host_post_scaler_t : low_pt_di_muon_line::Parameters::host_post_scaler_t {
  using type = low_pt_di_muon_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "low_pt_di_muon_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct track_muon_mva_line__host_post_scaler_t : track_muon_mva_line::Parameters::host_post_scaler_t {
  using type = track_muon_mva_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "track_muon_mva_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct gec_passthrough_line__host_post_scaler_t : passthrough_line::Parameters::host_post_scaler_t {
  using type = passthrough_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "gec_passthrough_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct passthrough_line__host_post_scaler_t : passthrough_line::Parameters::host_post_scaler_t {
  using type = passthrough_line::Parameters::host_post_scaler_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "passthrough_line__host_post_scaler_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct track_mva_line__host_post_scaler_hash_t : track_mva_line::Parameters::host_post_scaler_hash_t {
  using type = track_mva_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "track_mva_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct two_track_mva_line__host_post_scaler_hash_t : two_track_mva_line::Parameters::host_post_scaler_hash_t {
  using type = two_track_mva_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "two_track_mva_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct no_beam_line__host_post_scaler_hash_t : beam_crossing_line::Parameters::host_post_scaler_hash_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "no_beam_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct beam_one_line__host_post_scaler_hash_t : beam_crossing_line::Parameters::host_post_scaler_hash_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "beam_one_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct beam_two_line__host_post_scaler_hash_t : beam_crossing_line::Parameters::host_post_scaler_hash_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "beam_two_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct both_beams_line__host_post_scaler_hash_t : beam_crossing_line::Parameters::host_post_scaler_hash_t {
  using type = beam_crossing_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "both_beams_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct velo_micro_bias_line__host_post_scaler_hash_t : velo_micro_bias_line::Parameters::host_post_scaler_hash_t {
  using type = velo_micro_bias_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "velo_micro_bias_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct odin_lumi_line__host_post_scaler_hash_t : odin_event_type_line::Parameters::host_post_scaler_hash_t {
  using type = odin_event_type_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "odin_lumi_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct odin_no_bias__host_post_scaler_hash_t : odin_event_type_line::Parameters::host_post_scaler_hash_t {
  using type = odin_event_type_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "odin_no_bias__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct single_high_pt_muon_line__host_post_scaler_hash_t
  : single_high_pt_muon_line::Parameters::host_post_scaler_hash_t {
  using type = single_high_pt_muon_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "single_high_pt_muon_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct low_pt_muon_line__host_post_scaler_hash_t : low_pt_muon_line::Parameters::host_post_scaler_hash_t {
  using type = low_pt_muon_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "low_pt_muon_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2kk_line__host_post_scaler_hash_t : d2kk_line::Parameters::host_post_scaler_hash_t {
  using type = d2kk_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2kk_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2kpi_line__host_post_scaler_hash_t : d2kpi_line::Parameters::host_post_scaler_hash_t {
  using type = d2kpi_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2kpi_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct d2pipi_line__host_post_scaler_hash_t : d2pipi_line::Parameters::host_post_scaler_hash_t {
  using type = d2pipi_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "d2pipi_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_high_mass_line__host_post_scaler_hash_t : di_muon_mass_line::Parameters::host_post_scaler_hash_t {
  using type = di_muon_mass_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_high_mass_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_low_mass_line__host_post_scaler_hash_t : di_muon_mass_line::Parameters::host_post_scaler_hash_t {
  using type = di_muon_mass_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_low_mass_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct di_muon_soft_line__host_post_scaler_hash_t : di_muon_soft_line::Parameters::host_post_scaler_hash_t {
  using type = di_muon_soft_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "di_muon_soft_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct low_pt_di_muon_line__host_post_scaler_hash_t : low_pt_di_muon_line::Parameters::host_post_scaler_hash_t {
  using type = low_pt_di_muon_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "low_pt_di_muon_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct track_muon_mva_line__host_post_scaler_hash_t : track_muon_mva_line::Parameters::host_post_scaler_hash_t {
  using type = track_muon_mva_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "track_muon_mva_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct gec_passthrough_line__host_post_scaler_hash_t : passthrough_line::Parameters::host_post_scaler_hash_t {
  using type = passthrough_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "gec_passthrough_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};
struct passthrough_line__host_post_scaler_hash_t : passthrough_line::Parameters::host_post_scaler_hash_t {
  using type = passthrough_line::Parameters::host_post_scaler_hash_t::type;
  void set_size(size_t size) override { m_size = size; }
  size_t size() const override { return m_size; }
  std::string name() const override { return "passthrough_line__host_post_scaler_hash_t"; }
  void set_offset(char* offset) override { m_offset = offset; }
  char* offset() const override { return m_offset; }

private:
  size_t m_size = 0;
  char* m_offset = nullptr;
};

namespace gather_selections {
  namespace dev_input_selections_t {
    using tuple_t = std::tuple<
      track_mva_line__dev_decisions_t,
      two_track_mva_line__dev_decisions_t,
      no_beam_line__dev_decisions_t,
      beam_one_line__dev_decisions_t,
      beam_two_line__dev_decisions_t,
      both_beams_line__dev_decisions_t,
      velo_micro_bias_line__dev_decisions_t,
      odin_lumi_line__dev_decisions_t,
      odin_no_bias__dev_decisions_t,
      single_high_pt_muon_line__dev_decisions_t,
      low_pt_muon_line__dev_decisions_t,
      d2kk_line__dev_decisions_t,
      d2kpi_line__dev_decisions_t,
      d2pipi_line__dev_decisions_t,
      di_muon_high_mass_line__dev_decisions_t,
      di_muon_low_mass_line__dev_decisions_t,
      di_muon_soft_line__dev_decisions_t,
      low_pt_di_muon_line__dev_decisions_t,
      track_muon_mva_line__dev_decisions_t,
      gec_passthrough_line__dev_decisions_t,
      passthrough_line__dev_decisions_t>;
  }
} // namespace gather_selections
namespace gather_selections {
  namespace dev_input_selections_offsets_t {
    using tuple_t = std::tuple<
      track_mva_line__dev_decisions_offsets_t,
      two_track_mva_line__dev_decisions_offsets_t,
      no_beam_line__dev_decisions_offsets_t,
      beam_one_line__dev_decisions_offsets_t,
      beam_two_line__dev_decisions_offsets_t,
      both_beams_line__dev_decisions_offsets_t,
      velo_micro_bias_line__dev_decisions_offsets_t,
      odin_lumi_line__dev_decisions_offsets_t,
      odin_no_bias__dev_decisions_offsets_t,
      single_high_pt_muon_line__dev_decisions_offsets_t,
      low_pt_muon_line__dev_decisions_offsets_t,
      d2kk_line__dev_decisions_offsets_t,
      d2kpi_line__dev_decisions_offsets_t,
      d2pipi_line__dev_decisions_offsets_t,
      di_muon_high_mass_line__dev_decisions_offsets_t,
      di_muon_low_mass_line__dev_decisions_offsets_t,
      di_muon_soft_line__dev_decisions_offsets_t,
      low_pt_di_muon_line__dev_decisions_offsets_t,
      track_muon_mva_line__dev_decisions_offsets_t,
      gec_passthrough_line__dev_decisions_offsets_t,
      passthrough_line__dev_decisions_offsets_t>;
  }
} // namespace gather_selections
namespace gather_selections {
  namespace host_input_post_scale_factors_t {
    using tuple_t = std::tuple<
      track_mva_line__host_post_scaler_t,
      two_track_mva_line__host_post_scaler_t,
      no_beam_line__host_post_scaler_t,
      beam_one_line__host_post_scaler_t,
      beam_two_line__host_post_scaler_t,
      both_beams_line__host_post_scaler_t,
      velo_micro_bias_line__host_post_scaler_t,
      odin_lumi_line__host_post_scaler_t,
      odin_no_bias__host_post_scaler_t,
      single_high_pt_muon_line__host_post_scaler_t,
      low_pt_muon_line__host_post_scaler_t,
      d2kk_line__host_post_scaler_t,
      d2kpi_line__host_post_scaler_t,
      d2pipi_line__host_post_scaler_t,
      di_muon_high_mass_line__host_post_scaler_t,
      di_muon_low_mass_line__host_post_scaler_t,
      di_muon_soft_line__host_post_scaler_t,
      low_pt_di_muon_line__host_post_scaler_t,
      track_muon_mva_line__host_post_scaler_t,
      gec_passthrough_line__host_post_scaler_t,
      passthrough_line__host_post_scaler_t>;
  }
} // namespace gather_selections
namespace gather_selections {
  namespace host_input_post_scale_hashes_t {
    using tuple_t = std::tuple<
      track_mva_line__host_post_scaler_hash_t,
      two_track_mva_line__host_post_scaler_hash_t,
      no_beam_line__host_post_scaler_hash_t,
      beam_one_line__host_post_scaler_hash_t,
      beam_two_line__host_post_scaler_hash_t,
      both_beams_line__host_post_scaler_hash_t,
      velo_micro_bias_line__host_post_scaler_hash_t,
      odin_lumi_line__host_post_scaler_hash_t,
      odin_no_bias__host_post_scaler_hash_t,
      single_high_pt_muon_line__host_post_scaler_hash_t,
      low_pt_muon_line__host_post_scaler_hash_t,
      d2kk_line__host_post_scaler_hash_t,
      d2kpi_line__host_post_scaler_hash_t,
      d2pipi_line__host_post_scaler_hash_t,
      di_muon_high_mass_line__host_post_scaler_hash_t,
      di_muon_low_mass_line__host_post_scaler_hash_t,
      di_muon_soft_line__host_post_scaler_hash_t,
      low_pt_di_muon_line__host_post_scaler_hash_t,
      track_muon_mva_line__host_post_scaler_hash_t,
      gec_passthrough_line__host_post_scaler_hash_t,
      passthrough_line__host_post_scaler_hash_t>;
  }
} // namespace gather_selections
