###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from collections import OrderedDict
from PyConf.dataflow import GaudiDataHandle


class AllenAlgorithm(object):
    _all_algs = OrderedDict()

    def __new__(cls, name, **kwargs):
        i = super(AllenAlgorithm, cls).__new__(cls)
        for k, v in iter(cls.__slots__.items()):
            setattr(i, k, v)
        cls._all_algs[name] = i
        return cls._all_algs[name]

    @classmethod
    def getGaudiType(cls):
        return 'Algorithm'

    @classmethod
    def getDefaultProperties(cls):
        return cls.__slots__


class AllenDataHandle(GaudiDataHandle):
    def __init__(self, scope, dependencies, *args, **kwargs):
        super(AllenDataHandle, self).__init__(*args, **kwargs)
        self.Scope = scope
        self.Dependencies = dependencies
