###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.HLT1 import setup_hlt1_node
from AllenCore.generator import generate

hlt1_node = setup_hlt1_node(
    enablePhysics=False,
    EnableGEC=False,
    withSMOG2=True,
    enableBGI=True,
    with_ut=False,
)
generate(hlt1_node)
