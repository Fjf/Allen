#!/bin/bash
###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the Apache License          #
# version 2 (Apache-2.0), copied verbatim in the file "COPYING".              #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################

# Prepare tests in a separate dir
TEST_DIR=$(mktemp -d)
cp -r ${1}/tests ${TEST_DIR}/
cp -r ${1}/python/AllenConf ${TEST_DIR}/tests/
cp -r ${1}/python/AllenCore ${TEST_DIR}/tests/
cp ${1}/tests/test_algorithms.py ${TEST_DIR}/tests/AllenCore/algorithms.py
cp -r ${1}/python/AllenCore/allen_standalone_generator.py ${TEST_DIR}/tests/AllenCore/generator.py
rm ${TEST_DIR}/tests/test_algorithms.py ${TEST_DIR}/tests/test_configuration.sh

# Run tests
echo $TEST_DIR
pytest -s ${TEST_DIR}/tests

# Cleanup and exit
exit_code=$?
#rm -rf ${TEST_DIR}
exit $exit_code
