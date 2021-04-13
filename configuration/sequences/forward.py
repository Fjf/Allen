###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.utils import gec
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
from AllenConf.ut_reconstruction import decode_ut, make_ut_tracks
from AllenConf.scifi_reconstruction import decode_scifi, make_forward_tracks
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.event_list_utils import generate


def forward_tracking(name):
    decoded_velo = decode_velo()
    velo_tracks = make_velo_tracks(decoded_velo)
    decoded_ut = decode_ut()
    ut_tracks = make_ut_tracks(decoded_ut, velo_tracks)
    decoded_scifi = decode_scifi()
    forward_tracks = make_forward_tracks(decoded_scifi, ut_tracks)
    alg = forward_tracks["dev_scifi_track_hits"].producer
    return alg


forward_tracking_sequence = CompositeNode(
    "ForwardTrackingWithGEC",
    NodeLogic.AND, [gec("gec"), forward_tracking()],
    force_order=True,
    lazy=True)

generate(forward_tracking_sequence)
