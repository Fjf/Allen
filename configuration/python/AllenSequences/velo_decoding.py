###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.velo_reconstruction import decode_velo
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.generator import generate

decode_velo = CompositeNode("DecodeVelo",
                             [decode_velo(retina_decoding=False)["dev_sorted_velo_cluster_container"].producer])

generate(decode_velo)
