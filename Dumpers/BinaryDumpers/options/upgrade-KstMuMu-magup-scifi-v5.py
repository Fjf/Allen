###############################################################################
# (c) Copyright 2019 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from GaudiConf import IOHelper
IOHelper('ROOT').inputFiles(
    [
        # SciFi v4, KstMuMu
        'PFN:root://eoslhcb.cern.ch//eos/lhcb/user/g/gligorov/UpgradeStudies/FTv4Sim/KstMuMu_mu/Boole-Extended-0.digi',
        'PFN:root://eoslhcb.cern.ch//eos/lhcb/user/g/gligorov/UpgradeStudies/FTv4Sim/KstMuMu_mu/Boole-Extended-1.digi',
        'PFN:root://eoslhcb.cern.ch//eos/lhcb/user/g/gligorov/UpgradeStudies/FTv4Sim/KstMuMu_mu/Boole-Extended-2.digi',
        'PFN:root://eoslhcb.cern.ch//eos/lhcb/user/g/gligorov/UpgradeStudies/FTv4Sim/KstMuMu_mu/Boole-Extended-3.digi',
        'PFN:root://eoslhcb.cern.ch//eos/lhcb/user/g/gligorov/UpgradeStudies/FTv4Sim/KstMuMu_mu/Boole-Extended-4.digi',
        'PFN:root://eoslhcb.cern.ch//eos/lhcb/user/g/gligorov/UpgradeStudies/FTv4Sim/KstMuMu_mu/Boole-Extended-5.digi',
        'PFN:root://eoslhcb.cern.ch//eos/lhcb/user/g/gligorov/UpgradeStudies/FTv4Sim/KstMuMu_mu/Boole-Extended-6.digi',
        'PFN:root://eoslhcb.cern.ch//eos/lhcb/user/g/gligorov/UpgradeStudies/FTv4Sim/KstMuMu_mu/Boole-Extended-7.digi',
        'PFN:root://eoslhcb.cern.ch//eos/lhcb/user/g/gligorov/UpgradeStudies/FTv4Sim/KstMuMu_mu/Boole-Extended-8.digi',
        'PFN:root://eoslhcb.cern.ch//eos/lhcb/user/g/gligorov/UpgradeStudies/FTv4Sim/KstMuMu_mu/Boole-Extended-9.digi',
    ],
    clear=True)
