/***************************************************************************** \
 * (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/

/**
 * Convert PV::Vertex into LHCb::Event::PV::PrimaryVertexContainer
 *
 * author Dorothea vom Bruch and Wouter Hulsbergen
 *
 */

// Gaudi
#include "GaudiAlg/Transformer.h"

// LHCb
#include "Event/PrimaryVertices.h"

// Allen
#include "HostBuffers.cuh"
#include "Logger.h"
#include "PV_Definitions.cuh"
#include "patPV_Definitions.cuh"

using Vertices = LHCb::Event::PV::PrimaryVertexContainer ;

class AllenPVsToLHCbPVs final
  : public Gaudi::Functional::Transformer<Vertices(const HostBuffers&)> {
public:
  /// Standard constructor
  AllenPVsToLHCbPVs(const std::string& name, ISvcLocator* pSvcLocator);

  /// initialization
  StatusCode initialize() override;

  /// Algorithm execution
  Vertices operator()(const HostBuffers&) const override;

private:
  mutable Gaudi::Accumulators::SummingCounter<unsigned int> m_nbPVsCounter {this, "Nb PVs"};
};


DECLARE_COMPONENT(AllenPVsToLHCbPVs)

AllenPVsToLHCbPVs::AllenPVsToLHCbPVs(const std::string& name, ISvcLocator* pSvcLocator) :
  Transformer(
    name,
    pSvcLocator,
    // Inputs
    {KeyValue {"AllenOutput", "Allen/Out/HostBuffers"}},
    // Outputs
    {KeyValue {"OutputPVs", "Allen/PVs/PrimaryVertices"}})
{}

StatusCode AllenPVsToLHCbPVs::initialize()
{
  if (msgLevel(MSG::DEBUG)) debug() << "==> Initialize" << endmsg;
  return StatusCode::SUCCESS;
}

Vertices AllenPVsToLHCbPVs::operator()(const HostBuffers& host_buffers) const
{

  const unsigned i_event = 0;
  const unsigned n_pvs = host_buffers.host_number_of_multivertex[i_event];

  if (msgLevel(MSG::DEBUG)) debug() << "Number of PVs to convert = " << n_pvs << endmsg;

  Vertices pvcontainer ;
  auto& vertices = pvcontainer.vertices ;
  vertices.reserve(n_pvs);

  for (unsigned int i = 0; i < n_pvs; i++) {
    const PV::Vertex& vertex = host_buffers.host_reconstructed_multi_pvs[i_event * PatPV::max_number_vertices + i];

    Gaudi::SymMatrix3x3 poscov;
    poscov(0, 0) = vertex.cov00;
    poscov(1, 0) = vertex.cov10;
    poscov(1, 1) = vertex.cov11;
    poscov(2, 0) = vertex.cov20;
    poscov(2, 1) = vertex.cov21;
    poscov(2, 2) = vertex.cov22;
    auto& recvertex = vertices.emplace_back( Gaudi::XYZPoint{vertex.position.x, vertex.position.y, vertex.position.z} ) ;
    recvertex.setCovMatrix( poscov ) ;
    recvertex.setChi2( vertex.chi2 ) ;
    recvertex.setNDoF( vertex.ndof ) ;
  }

  m_nbPVsCounter += vertices.size();
  return pvcontainer ;
}
