#!/usr/bin/bash
###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
###############################################################################

set -euxo pipefail

TOPLEVEL=$(realpath ${PWD})
ls "${TOPLEVEL}/test/reference"
mkdir -p ${TOPLEVEL}/generated_reference_files/
set +x;
RC=0
DIFFERENCES=""
for OUTPUT_FOLDER in run_physics_efficiency_output_*/ ; do
  echo "==============================="
  echo "Entering ${OUTPUT_FOLDER}"
  cd ${OUTPUT_FOLDER}
  ls -1

  DIFF_FOUND=0
  DIFFS_THISFOLDER=""
  for i in $( ls ); do
    echo "==============="
    echo " "
    echo " "
    echo "Checking ${i}"

    echo "Folder    : ${OUTPUT_FOLDER}"
    echo "File      : efficiency_${i}"

    FIRST=`grep -nr "Processing complete" ${i} | sed -e 's/\([0-9]*\).*/\1/'`
    NLINES=`wc -l ${i} | awk '{ print $1; }'`
    tail -n$((${NLINES}-${FIRST}-1)) ${i} | head -n$((${NLINES}-${FIRST}-3)) > efficiency_${i}
    cp efficiency_${i} ${TOPLEVEL}/generated_reference_files/${i}
    EXTRA_REASON=""

    if [ ! -f "${TOPLEVEL}/test/reference/${i}" ]; then
      echo "Reference : NOT FOUND - please, add a reference file!"
      touch "${TOPLEVEL}/test/reference/${i}"
      EXTRA_REASON="A reference file was not found. Please, create one!"
      echo ""
    else
      echo "Reference : test/reference/${i}"
      echo ""
    fi


    if ! diff -u -B -Z ${TOPLEVEL}/test/reference/${i} efficiency_${i}; then
      echo "***"
      echo "*** A difference was found. ${EXTRA_REASON}"
      echo "***"
      cp efficiency_${i} ${TOPLEVEL}/test/reference/${i}
      DIFF_FOUND=1
      DIFFS_THISFOLDER="${DIFFS_THISFOLDER}
      - ${OUTPUT_FOLDER}: ${i} ${EXTRA_REASON}"
    else
      echo "*** No differences found"
    fi
  done
  echo "---"
  if [ ${DIFF_FOUND} -ne 0 ]; then
    echo "*** Differences were found against reference files for ${OUTPUT_FOLDER}.";
    DIFFERENCES="${DIFFERENCES}
    ${DIFFS_THISFOLDER}"
    RC=3
  else
    echo "*** No differences found against reference files for ${OUTPUT_FOLDER}.";
  fi
  cd ${TOPLEVEL}
done

if [ ${RC} -ne 0 ]; then
  echo " "
  echo "*** Differences were found in efficiencies for:"
  echo "${DIFFERENCES}"
  echo " "
  echo "*** See above for diffs."

  git --version
  git -c user.name="Gitlab CI" -c user.email="noreply@cern.ch" commit test/reference -m "Update CI references" -m "patch generated by ${CI_JOB_URL}"
  git format-patch -1 --output=update-references.patch
  echo "======================================="
  echo " If the diffs make sense, you can update the references with:"
  echo "   curl ${CI_JOB_URL}/artifacts/raw/update-references.patch | git am"
  echo "======================================="
fi

exit ${RC}
