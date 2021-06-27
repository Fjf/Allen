###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.ut_reconstruction import ut_tracking
from AllenConf.utils import gec
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenCore.event_list_utils import generate

ut_tracking_sequence = CompositeNode(
    "UTTrackingWithGEC", [gec("gec"), ut_tracking()],
    NodeLogic.LAZY_AND,
    force_order=True)

generate(ut_tracking_sequence)