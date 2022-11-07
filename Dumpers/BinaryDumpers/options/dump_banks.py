###############################################################################
# (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      #
###############################################################################
from Configurables import LHCbApp, CondDB
from Configurables import GaudiSequencer
from Configurables import FTRawBankDecoder
from Configurables import (PrStoreUTHit, PrStoreFTHit)
from Configurables import ApplicationMgr
from Configurables import HistogramPersistencySvc
from Configurables import (AuditorSvc, SequencerTimerTool)
from Configurables import DumpUTGeometry
from Configurables import RootHistCnv__PersSvc
from Configurables import (VPClus, createODIN, DumpRawBanks, DumpUTHits,
                           DumpFTHits, DumpMuonCoords, DumpMuonCommonHits,
                           MuonRec, PrepareMuonHits, TransposeRawBanks)

# new MC in tmp/
#DDDBtag = "dddb-20171010"
#CondDBtag = "sim-20170301-vc-md100"

# For privately produced minbias and BsPhiPhi
# in /eos/lhcb/wg/SciFi/Simulation/MinBiasRawBankv5/
# and /eos/lhcb/wg/SciFi/Simulation/RawBanksv5/
DDDBtag = "dddb-20180815"
CondDBtag = "sim-20180530-vc-md100"

Evts_to_Run = 10  # set to -1 to process all

app = LHCbApp(
    DataType="Upgrade",
    EvtMax=Evts_to_Run,
    Simulation=True,
    DDDBtag=DDDBtag,
    CondDBtag=CondDBtag)
CondDB().Upgrade = True

# Decode VP, UT, FT and muons
vp_decoder = VPClus("VPClustering")
store_ut = PrStoreUTHit()
ft_decoder = FTRawBankDecoder("createFTClusters", DecodingVersion=5)
store_ft = PrStoreFTHit()
decode_muon = MuonRec()
muon_hits = PrepareMuonHits()

dec_seq = GaudiSequencer("DecodingSeq")
dec_seq.Members = [
    createODIN(), vp_decoder, store_ut, ft_decoder, store_ft, decode_muon,
    muon_hits
]

# Add the service that will dump the UT geometry
ApplicationMgr().ExtSvc += [DumpUTGeometry()]

# Dump raw banks and UT, FT and muon hits
transpose_banks = TransposeRawBanks(
    BankTypes=[
        "VP", "VPRetinaCluster", "UT", "FTCluster", "Muon", "ODIN", "Plume"
    ],
    RawEventLocations=["DAQ/RawEvent"])
dump_banks = DumpRawBanks()
dump_seq = GaudiSequencer("DumpSeq")
dump_seq.Members += [transpose_banks, dump_banks]

ApplicationMgr().TopAlg = [dec_seq, dump_seq]

# Some extra stuff for timing table
ApplicationMgr().ExtSvc += ['ToolSvc', 'AuditorSvc']
ApplicationMgr().AuditAlgorithms = True
AuditorSvc().Auditors += ['TimingAuditor']
SequencerTimerTool().OutputLevel = 4

# Some extra stuff to save histograms
ApplicationMgr().HistogramPersistency = "ROOT"
RootHistCnv__PersSvc('RootHistCnv').ForceAlphaIds = True
HistogramPersistencySvc().OutputFile = "dump-histos.root"
