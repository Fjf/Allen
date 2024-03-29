###############################################################################
# (c) Copyright 2023 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "COPYING".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
from AllenConf.HLT1 import setup_hlt1_node
from AllenCore.generator import generate

hlt1_node = setup_hlt1_node(
    with_ut=False,
    EnableGEC=False,
    enableRateValidator=True,
    with_calo=False,
    with_muon=False,
    with_v0s=False)
generate(hlt1_node)
