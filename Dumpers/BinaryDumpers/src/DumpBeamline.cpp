/*****************************************************************************\
* (c) Copyright 2000-2019 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <tuple>
#include <vector>

#include <DetDesc/GenericConditionAccessorHolder.h>
#include <VPDet/DeVP.h>

#include <Dumpers/Identifiers.h>
#include <Dumpers/Utils.h>
#include <DD4hep/GrammarUnparsed.h>

#include "Dumper.h"

namespace {
  struct Beamline {

    Beamline() {}

    Beamline(std::vector<char>& data, DeVP const& velo, std::vector<float> const& offset)
    {
      DumpUtils::Writer output;

      auto const beamSpot = velo.beamSpot();
      float x = static_cast<float>(beamSpot.x()) + offset[0];
      float y = static_cast<float>(beamSpot.y()) + offset[1];
      std::cout << "Beamline position: x = " << x << ", y = " << y << std::endl;
      output.write(x, y);
      data = output.buffer();
    }
  };
} // namespace

/** @class DumpBeamline
 *  Dump beamline position.
 *
 *  @author Roel Aaij
 *  @date   2019-04-27
 */
class DumpBeamline final
  : public Allen::Dumpers::Dumper<void(Beamline const&), LHCb::DetDesc::usesConditions<Beamline>> {
public:
  DumpBeamline(const std::string& name, ISvcLocator* svcLoc);

  void operator()(const Beamline& beamline) const override;

  StatusCode initialize() override;

private:
  std::vector<char> m_data;

  Gaudi::Property<std::vector<float>> m_offset {this, "BeamSpotOffset", {0.f, 0.f}};
};

DECLARE_COMPONENT(DumpBeamline)

DumpBeamline::DumpBeamline(const std::string& name, ISvcLocator* svcLoc) :
  Dumper(name, svcLoc, {KeyValue {"BeamSpotLocation", location(name, "beamspot")}})
{}

StatusCode DumpBeamline::initialize()
{
  return Dumper::initialize().andThen([&] {
    register_producer(Allen::NonEventData::Beamline::id, "beamline", m_data);
    addConditionDerivation({DeVPLocation::Default}, inputLocation<Beamline>(), [&](DeVP const& velo) {
      auto beamline = Beamline {m_data, velo, m_offset};
      dump();
      return beamline;
    });
  });
}

void DumpBeamline::operator()(const Beamline&) const {}
