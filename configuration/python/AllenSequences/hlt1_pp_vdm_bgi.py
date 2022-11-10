###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.HLT1 import setup_hlt1_node
from AllenCore.generator import generate

hlt1_node = setup_hlt1_node(
    with_ut=False, enablePhysics=False, EnableGEC=False, withSMOG2=True, enableBGI=True)
generate(hlt1_node)
