###############################################################################
# (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      #
###############################################################################
from __future__ import print_function
from Configurables import ProcessPhase, GaudiSequencer
from Configurables import ApplicationMgr
from Gaudi.Configuration import importOptions
from GaudiKernel.Configurable import applyConfigurableUsers
from Gaudi.Configuration import allConfigurables

importOptions("dump_mc.py")
importOptions("../tests/options/upgrade-minbias-ldst.py")

applyConfigurableUsers()


def printAlgos(confs, prefix=''):
    for alg in confs:
        if type(alg) is str:
            alg = allConfigurables.get(alg.split('/')[-1], alg)
        if type(alg) is ProcessPhase:
            print(prefix + ' ' + alg.getFullName())
            printAlgos([alg.name() + d + 'Seq' for d in alg.DetectorList],
                       prefix + '   ')
        elif type(alg) is GaudiSequencer:
            print(prefix + ' ' + alg.getFullName())
            printAlgos(alg.Members, prefix + '   ')
        elif type(alg) is str:
            print(prefix + ' ' + alg)
        else:
            print(prefix + ' ' + alg.getFullName())


printAlgos(ApplicationMgr().TopAlg)

from GaudiPython.Bindings import gbl, AppMgr

gaudi = AppMgr()
gaudi.initialize()

TES = gaudi.evtSvc()

gaudi.run(1)

coords = TES['Raw/Muon/Coords'].containedObjects()
print(coords[0])
