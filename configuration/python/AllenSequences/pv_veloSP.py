###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.primary_vertex_reconstruction import pv_finder
from AllenConf.utils import make_gec
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.generator import generate
from AllenConf.velo_reconstruction import decode_velo

with decode_velo.bind(retina_decoding=False) and make_gec.bind(count_ut=False):
  pv_finder_sequence = CompositeNode(
      "PVWithGEC", [pv_finder()],
      NodeLogic.LAZY_AND,
      force_order=True)

generate(pv_finder_sequence)
