#!/bin/bash
###############################################################################
# (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           #
###############################################################################

FOLDER=$1
if [ -z "$FOLDER" ]; then

  echo "Runs Allen throughput scaling on GPU, with a varying number of streams."
  echo "Usage: $0 <folder_containing_Allen_executable> [<output_folder>=output_'hostname'] [<data_directory>=/scratch/dcampora/allen_data/201907/minbias_mag_down]"

else

  OUTPUT_FOLDER=$2
  if [ -z "$OUTPUT_FOLDER" ]; then
    OUTPUT_FOLDER="output_`hostname`"
  fi

  DATA_DIRECTORY=$3
  if [ -z "$DATA_DIRECTORY" ]; then
    DATA_DIRECTORY="/scratch/dcampora/allen_data/201907/minbias_mag_down"
  fi

  mkdir ${OUTPUT_FOLDER}

  NUM_PROCESSORS=`lscpu | grep "CPU(s):" | head -n1 | awk '{ print $2 }'`
  NUM_NUMA_NODES=`lscpu | grep "NUMA node(s):" | awk '{ print $3 }'`
  ITERATIONS=20

  for i in `seq 1 ${ITERATIONS}`; do
    echo "Running iteration ${i}/${ITERATIONS}..."

    numactl --cpunodebind=0 --membind=0 ${FOLDER}/Allen -f ${DATA_DIRECTORY} -n 500 -r 1000 -c 0 -m 500 -t $i > ${OUTPUT_FOLDER}/run_${i}.out
  done

  OUTPUT_FILE=${OUTPUT_FOLDER}/runs.csv
  touch $OUTPUT_FILE

  for i in `seq 1 ${ITERATIONS}`; do
    value=`cat ${OUTPUT_FOLDER}/run_${i}.out | grep events/s | awk '{ print $1 }'`
    cores=$i

    echo $cores, $value >> $OUTPUT_FILE
  done

  echo "Generated $OUTPUT_FILE"

fi
