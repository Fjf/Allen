/*****************************************************************************\
* (c) Copyright 2000-2019 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <array>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>
#include "DumpMuonGeometry.h"
#include "fmt/format.h"
#include <Dumpers/Utils.h>

DECLARE_COMPONENT(DumpMuonGeometry)

StatusCode DumpMuonGeometry::registerConditions(IUpdateManagerSvc* updMgrSvc)
{
  const auto& det = detector();
  m_daqHelper.initSvc(detSvc(), msgSvc());

  for (auto* station : det.childIDetectorElements()) {
    std::string_view name = station->name();
    if (name.find("/MF") != std::string_view::npos) continue; // skip muon filters
    auto path = fmt::format("{}/{}/Cabling", DeMuonLocation::Cabling, name.substr(name.size() - 2));
    info() << "Registering " << path << endmsg;
    updMgrSvc->registerCondition(&m_daqHelper, path, &MuonDAQHelper::updateLUT);
  }
  return updMgrSvc->update(&m_daqHelper);
}

DumpUtils::Dumps DumpMuonGeometry::dumpGeometry() const
{

  // Detector and mat geometry
  const auto& det = detector();
  const int nStations = det.stations();
  assert(nStations == 4);

  std::array<float, 4> innerX {}, innerY {}, outerX {}, outerY {}, stationZ {};

  DumpUtils::Writer output {};
  for (int s = 0; s != nStations; ++s) {
    innerX[s] = det.getInnerX(s);
    innerY[s] = det.getInnerY(s);
    outerX[s] = det.getOuterX(s);
    outerY[s] = det.getOuterY(s);
    stationZ[s] = det.getStationZ(s);
  }

  output.write(
    innerX.size(),
    innerX,
    innerY.size(),
    innerY,
    outerX.size(),
    outerX,
    outerY.size(),
    outerY,
    stationZ.size(),
    stationZ);

  std::vector<unsigned int> nTiles(m_daqHelper.TotTellNumber(), 0);
  for (auto tell1 = 0u; tell1 < m_daqHelper.TotTellNumber(); ++tell1) {
    nTiles[tell1] = m_daqHelper.getADDInTell1(tell1).size();
  }

  output.write(nTiles.size());
  info() << "Number of tiles in tell1 table: " << std::accumulate(nTiles.begin(), nTiles.end(), 0u) << endmsg;
  for (auto tell1 = 0u; tell1 < m_daqHelper.TotTellNumber(); ++tell1) {
    auto const& tiles = m_daqHelper.getADDInTell1(tell1);
    output.write(tiles.size());
    for (auto const& tile : tiles) {
      output.write(static_cast<unsigned int>(tile));
    }
  }

  return {{std::tuple {output.buffer(), "muon_geometry", Allen::NonEventData::MuonGeometry::id}}};
}
