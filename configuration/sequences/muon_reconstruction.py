###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from definitions.utils import gec
from definitions.velo_reconstruction import decode_velo, make_velo_tracks
from definitions.ut_reconstruction import decode_ut, make_ut_tracks
from definitions.scifi_reconstruction import decode_scifi, make_forward_tracks
from definitions.muon_reconstruction import decode_muon, is_muon
from definitions.event_list_utils import generate, make_leaf
from PyConf.control_flow import NodeLogic, CompositeNode


def gec_leaf(name, min_scifi_ut_clusters="0", max_scifi_ut_clusters="9750"):
    alg = gec(
        min_scifi_ut_clusters=min_scifi_ut_clusters,
        max_scifi_ut_clusters=max_scifi_ut_clusters)
    return make_leaf(name, alg=alg)


def muon_id_leaf(name):
    decoded_velo = decode_velo()
    velo_tracks = make_velo_tracks(decoded_velo)
    decoded_ut = decode_ut()
    ut_tracks = make_ut_tracks(decoded_ut, velo_tracks)
    decoded_scifi = decode_scifi()
    forward_tracks = make_forward_tracks(decoded_scifi, ut_tracks)
    decoded_muon = decode_muon()
    muonID = is_muon(decoded_muon, forward_tracks)
    alg = muonID["dev_is_muon"].producer
    return make_leaf(name, alg=alg)


muon_id_sequence = CompositeNode(
    "MuonIDWithGEC",
    [gec_leaf("gec"), muon_id_leaf("muon_id")],
    NodeLogic.LAZY_AND,
    forceOrder=True)

generate(muon_id_sequence)
