###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.HLT1 import setup_hlt1_node
from AllenConf.ut_reconstruction import make_ut_tracks
from AllenCore.generator import generate

with make_ut_tracks.bind(restricted=False):
    hlt1_node = setup_hlt1_node(enableRateValidator=True)

generate(hlt1_node)
