#!/usr/bin/bash
###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
###############################################################################

set -euo pipefail

setupViews

set +x

for SEQUENCE_DATASET in $(ls -1 | grep "devices_throughputs" | grep -Ei "devices_throughputs_([a-z0-9_]+?)" | sed 's/^devices_throughputs_//' | sed 's/.csv$//') ; do

    if [ -f "test_throughput_details/${SEQUENCE_DATASET}_${BREAKDOWN_DEVICE_ID}_no_throughput_report.txt" ]; then 
        continue;
    fi

    INPUT_FILES=$(cat test_throughput_details/${SEQUENCE_DATASET}_${BREAKDOWN_DEVICE_ID}_input_files.txt)
    SEQUENCE=$(cat test_throughput_details/${SEQUENCE_DATASET}_${BREAKDOWN_DEVICE_ID}_sequence.txt)
    BUILDOPTIONS=$(cat test_throughput_details/${SEQUENCE_DATASET}_${BREAKDOWN_DEVICE_ID}_buildopts.txt)

    echo ""
    echo "********************************************************************************************************************************************"
    echo "********************************************************************************************************************************************"
    echo "Throughput of [branch ${CI_COMMIT_REF_NAME} (${CI_COMMIT_SHORT_SHA}), sequence ${SEQUENCE} over dataset ${INPUT_FILES}"
    echo ""
    echo ""

    if [ "${BUILDOPTIONS}" = "" ]; then
        BUILDOPTIONS_DISPLAY="default"
    else
        BUILDOPTIONS_DISPLAY=${BUILDOPTIONS}
    fi

    RC=0
    python3 checker/plotting/post_combined_message.py \
        -j "${REFERENCE_JOB}" \
        -l "Throughput of [branch **\`${CI_COMMIT_REF_NAME} (${CI_COMMIT_SHORT_SHA})\`**, sequence **\`${SEQUENCE}\`** over dataset **\`${INPUT_FILES}\`** build options \`${BUILDOPTIONS_DISPLAY}\`](https://gitlab.cern.ch/lhcb/Allen/pipelines/${CI_PIPELINE_ID})" \
        -t devices_throughputs_${SEQUENCE_DATASET}.csv \
        -b test_throughput_details/${SEQUENCE_DATASET}_${BREAKDOWN_DEVICE_ID}_algo_breakdown.csv \
        || RC=$?

    python3 checker/plotting/post_telegraf.py \
        -f devices_throughputs_${SEQUENCE_DATASET}.csv . \
        -s "${SEQUENCE}" -b "${CI_COMMIT_REF_NAME}" -d "${INPUT_FILES}" -o "${BUILDOPTIONS}" \
        || echo "WARNING: failed to post to telegraf"

    echo ""
    echo ""
done
