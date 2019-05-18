#pragma once

#include <Constants.cuh>
#include <Dumpers/IUpdater.h>
#include <gsl-lite.hpp>

namespace Consumers {

  struct RawGeometry final : public Allen::NonEventData::Consumer {
  public:

    RawGeometry(char*& dev_geometry);

    void consume(std::vector<char> const& data) override;

  private:
    std::reference_wrapper<char*> m_dev_geometry;
    size_t m_size = 0;
  };

  struct BasicGeometry final : public Allen::NonEventData::Consumer {
  public:

    BasicGeometry(gsl::span<char>& dev_geometry);

    void consume(std::vector<char> const& data) override;

  private:
    std::reference_wrapper<gsl::span<char>> m_dev_geometry;
  };

  struct UTGeometry final : public Allen::NonEventData::Consumer {
  public:

    UTGeometry(Constants& constants);

    void consume(std::vector<char> const& data) override;

  private:

    void initialize(const std::vector<char>& data);
    std::reference_wrapper<Constants> m_constants;

  };

  struct UTLookupTables final : public Allen::NonEventData::Consumer {
  public:

    UTLookupTables(PrUTMagnetTool*& tool);

    void consume(std::vector<char> const& data) override;

  private:

    std::reference_wrapper<PrUTMagnetTool*> m_tool;
    size_t m_size = 0;
  };

  struct SciFiGeometry final : public Allen::NonEventData::Consumer {
  public:

    SciFiGeometry(std::vector<char>& host_geometry, char*& dev_geometry);

    void consume(std::vector<char> const& data) override;

  private:
    std::reference_wrapper<std::vector<char>> m_host_geometry;
    std::reference_wrapper<char*> m_dev_geometry;
  };

  struct Beamline final : public Allen::NonEventData::Consumer {
  public:

    Beamline(float*& );

    void consume(std::vector<char> const& data) override;

  private:

    std::reference_wrapper<float*> m_dev_beamline;
    const size_t m_size = 2 * sizeof(float);

  };

  struct MagneticField final : public Allen::NonEventData::Consumer {
  public:

    MagneticField(float*&);

    void consume(std::vector<char> const& data) override;

  private:

    std::reference_wrapper<float*> m_dev_magnet_polarity;
    const size_t m_size = sizeof(float);

  };

  struct MuonGeometry final : public Allen::NonEventData::Consumer {
  public:
    static constexpr n_preamble_blocks = 5;

    MuonGeometry(std::vector<char>& host_geometry_raw, char*& dev_geometry_raw, Muon::MuonGeometry*& muon_geometry);

    void consume(std::vector<char> const& data) override;

  private:
    std::reference_wrapper<std::vector<char>> m_host_geometry_raw;
    std::reference_wrapper<char*> m_dev_geometry_raw;
    std::reference_wrapper<Muon::MuonGeometry*> m_muon_geometry;
    size_t m_size = 0;
  };

  struct MuonLookupTables final : public Allen::NonEventData::Consumer {
  public:
    //3 stations, each has gridX, gridY, sizeX, sizeY, offset, and 4 blocks of coordinates (1 block per station)
    static constexpr n_data_blocks = 27;

    MuonLookupTables(std::vector<char>& host_muon_tables_raw, char*& dev_muon_tables_raw, Muon::MuonTables*& muon_tables);

    void consume(std::vector<char> const& data) override;

  private:
    std::reference_wrapper<std::vector<char>> m_host_muon_tables_raw;
    std::reference_wrapper<char*> m_dev_muon_tables_raw;
    std::reference_wrapper<Muon::MuonTables*> m_muon_tables;
    size_t m_size = 0;
  };
}
