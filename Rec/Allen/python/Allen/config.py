###############################################################################
# (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from Configurables import (ApplicationMgr, DumpUTGeometry, DumpFTGeometry,
                           DumpMuonTable, DumpMuonGeometry, DumpCaloGeometry,
                           DumpVPGeometry, DumpMagneticField, DumpBeamline,
                           DumpUTLookupTables, AllenUpdater)


def setup_allen_non_event_data_service(dump_binaries=False):
    """Setup Allen non-event data

    An ExtSvc is added to the ApplicationMgr to provide the Allen non-event
    data (geometries etc.)
    """
    from functools import partial
    ecal_location = "/dd/Structure/LHCb/DownstreamRegion/Ecal"
    hcal_location = "/dd/Structure/LHCb/DownstreamRegion/Hcal"
    ecal_geom = partial(
        DumpCaloGeometry, name="DumpEcal", Location=ecal_location)
    hcal_geom = partial(
        DumpCaloGeometry, name="DumpHcal", Location=hcal_location)
    producers = [
        p(DumpToFile=dump_binaries)
        for p in (DumpVPGeometry, DumpUTGeometry, DumpFTGeometry,
                  DumpMuonGeometry, DumpMuonTable, DumpMagneticField,
                  DumpBeamline, DumpUTLookupTables, ecal_geom, hcal_geom)
    ]
    ApplicationMgr().ExtSvc += [
        AllenUpdater(OutputLevel=2),
    ] + producers
