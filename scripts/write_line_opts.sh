#!/bin/bash
###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
###############################################################################
source /cvmfs/sft.cern.ch/lcg/releases/clang/8.0.0.1/x86_64-centos7/setup.sh
python parse_bank_def.py -o $2 $1
