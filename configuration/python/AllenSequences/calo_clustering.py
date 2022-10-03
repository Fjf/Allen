###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.generator import generate
from AllenConf.persistency import make_global_decision
from AllenConf.utils import line_maker
from AllenConf.calo_reconstruction import decode_calo
from AllenConf.validators import rate_validation
from AllenConf.hlt1_photon_lines import make_single_calo_cluster_line, make_standalone_pi0_line
from AllenConf.hlt1_reconstruction import hlt1_reconstruction
from AllenConf.hlt1_monitoring_lines import make_calo_digits_minADC_line

reconstructed_objects = hlt1_reconstruction()
ecal_clusters = reconstructed_objects["ecal_clusters"]

calo_cluster_line = line_maker(
    make_single_calo_cluster_line(
        ecal_clusters, name="Hlt1SingleCaloCluster", minEt=400))

calo_pi0_line = line_maker(
    make_standalone_pi0_line(
        ecal_clusters, 
        name="Hlt1Pi0CaloOnly", 
        minMass=100,  #MeV
        maxMass=500,  #MeV
        minEt=1400,
        minEt_clusters=1400,
        minSumEt_clusters=1400,
        minE19_clusters=0.6,
        enable_monitoring=True))

line_algorithms = [calo_cluster_line[0], calo_pi0_line[0]]

global_decision = make_global_decision(lines=line_algorithms)

lines = CompositeNode(
    "AllLines", [calo_cluster_line[1], calo_pi0_line[1]],
    NodeLogic.NONLAZY_OR,
    force_order=False)

calo_sequence = CompositeNode(
    "CaloClustering",
    [lines, global_decision,
     rate_validation(lines=line_algorithms)],
    NodeLogic.NONLAZY_AND,
    force_order=True)

generate(calo_sequence)
