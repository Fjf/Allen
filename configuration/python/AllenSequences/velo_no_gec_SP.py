###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.velo_reconstruction import decode_velo, velo_tracking
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.generator import generate
from AllenConf.hlt1_calibration_lines import make_passthrough_line
from AllenConf.persistency import make_global_decision, make_routingbits_writer
from AllenConf.utils import line_maker
from AllenConf.validators import rate_validation

with decode_velo.bind(retina_decoding=False):
    velo_tracking_sequence = CompositeNode("VeloTrackingSP", [velo_tracking()])

passthrough_line = line_maker(make_passthrough_line(pre_scaler=0.04))
line_algorithms = [passthrough_line[0]]

global_decision = make_global_decision(lines=line_algorithms)

lines = CompositeNode(
    "AllLines", [passthrough_line[1]], NodeLogic.NONLAZY_OR, force_order=False)

with decode_velo.bind(retina_decoding=False):
    velo_tracking_sequence = CompositeNode(
        "VeloTrackingSP", [
            velo_tracking(), lines,
            make_routingbits_writer(lines=line_algorithms), global_decision,
            rate_validation(lines=line_algorithms)
        ],
        NodeLogic.NONLAZY_AND,
        force_order=True)

generate(velo_tracking_sequence)