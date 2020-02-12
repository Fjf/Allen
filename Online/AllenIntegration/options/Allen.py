#!/usr/bin/env python2
from Configurables import LHCbApp, CondDB, ApplicationMgr
from Configurables import DumpUTGeometry, DumpFTGeometry, DumpMuonTable
from Configurables import DumpMuonGeometry, DumpVPGeometry, AllenUpdater
from Configurables import DumpMagneticField, DumpBeamline, DumpUTLookupTables
from Configurables import AllenConfiguration
from Configurables import MonitorSvc
from Configurables import Online__Configuration as OnlineConfiguration

app = LHCbApp(
    DataType="Upgrade",
    EvtMax=1000,
    Simulation=True,
    DDDBtag="dddb-20171122",
    CondDBtag="sim-20180530-vc-md100")

# Upgrade DBs
CondDB().Upgrade = True

producers = [
    p(DumpToFile=False)
    for p in (DumpVPGeometry, DumpUTGeometry, DumpFTGeometry, DumpMuonGeometry,
              DumpMuonTable, DumpMagneticField, DumpBeamline,
              DumpUTLookupTables)
]

online_conf = OnlineConfiguration("Application")
online_conf.debug = False
online_conf.classType = 1
online_conf.automatic = False
online_conf.monitorType = 'MonitorSvc'
online_conf.logDeviceType = 'RTL::Logger::LogDevice'
online_conf.logDeviceFormat = '%TIME%LEVEL%-8NODE: %-32PROCESS %-20SOURCE'
online_conf.OutputLevel = 3
online_conf.IOOutputLevel = 3

allen_conf = AllenConfiguration()
allen_conf.EventsPerSlice = 1000
allen_conf.NonStop = True
allen_conf.MPI = False
allen_conf.Receivers = "mlx5_0:1"
allen_conf.NThreads = 8
allen_conf.NSlices = 16
# allen_conf.Output = "tcp://192.168.1.101:35000"
# allen_conf.Device = "01:00.0"
allen_conf.Input = [
    "/scratch/raaij/mep/upgrade_mc_minbias_scifi_v5_pf3000.mep"
]
allen_conf.Device = "0"
allen_conf.OutputLevel = 2

monSvc = MonitorSvc('MonitorSvc')
monSvc.PartitionName = 'Allen'
monSvc.ExpandNameInfix = '<proc>'
monSvc.ExpandCounterServices = True
monSvc.UniqueServiceNames = True

# Add the services that will produce the non-event-data
ApplicationMgr().ExtSvc += [
    monSvc,
    AllenUpdater(OutputLevel=2),
] + producers

# Some extra stuff for timing table
ApplicationMgr().EvtSel = "NONE"
ApplicationMgr().ExtSvc += [
    'ToolSvc', 'AuditorSvc', 'AllenConfiguration',
    'Online::Configuration/Application', 'ZeroMQSvc'
]

# gaudi = AppMgr()