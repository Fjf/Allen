###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.velo_reconstruction import velo_tracking
from AllenConf.utils import make_gec
from AllenConf.hlt1_reconstruction import hlt1_reconstruction
from AllenConf.hlt1_monitoring_lines import make_velo_micro_bias_line
from AllenConf.persistency import make_global_decision, make_routingbits_writer
from AllenCore.generator import generate
from AllenConf.utils import line_maker
from AllenConf.validators import rate_validation
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.generator import generate

reconstructed_objects = hlt1_reconstruction()
velo_tracks = reconstructed_objects["velo_tracks"]
velo_tracking = velo_tracking() 

velo_micro_bias_line = line_maker(
    make_velo_micro_bias_line(
        velo_tracks,
        pre_scaler=1., 
        min_velo_tracks=2))

line_algorithms = [velo_micro_bias_line[0]]
global_decision = make_global_decision(lines=line_algorithms)

lines = CompositeNode(
    "AllLines", [velo_micro_bias_line[1]], NodeLogic.NONLAZY_OR, force_order=False)

velo_tracking_sequence = CompositeNode(
    "VeloTracking", [
        velo_tracking,
        lines,
        make_routingbits_writer(lines=line_algorithms), global_decision,
        rate_validation(lines=line_algorithms)
    ], 
    NodeLogic.NONLAZY_AND, 
    force_order=True)

generate(velo_tracking_sequence)
