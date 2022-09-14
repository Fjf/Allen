###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.velo_reconstruction import velo_tracking, decode_velo
from AllenConf.utils import make_gec
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.generator import generate

with decode_velo.bind(retina_decoding=False):
    velo_tracking_sequence = CompositeNode(
        "VeloTrackingWithGEC",
        [make_gec("gec"), velo_tracking()],
        NodeLogic.LAZY_AND,
        force_order=True)

generate(velo_tracking_sequence)
