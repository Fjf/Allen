Run Allen
============

.. _run_allen_standalone:

Standalone Allen
^^^^^^^^^^^^^^^^^^^^

Some input files are included with the project for testing:

* `input/minbias/mdf/MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster_v1.mdf`: Minbias sample produced from MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster TestFile DB entry. Includes raw banks with MC information.
* The directory `input/detector_configuration` contains the dumped geometry files for MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster
* Other dumped Allen geometries are located in `/scratch/allen_geometries` in the LHCb Online domain, and are used for other data sets in the CI tests
* Dumped Allen geometries can also be found in eos under `/eos/lhcb/wg/rta/WP6/Allen/geometries`

A run of the Allen program with the help option `-h` will let you know the basic options::

    Usage: ./Allen
     -g {folder containing detector configuration}=../input/detector_configuration/
     --mdf {comma-separated list of MDF files to use as input OR single text file containing one MDF file per line}
     --mep {comma-separated list of MEP files to use as input}
     --transpose-mep {Transpose MEPs instead of decoding from MEP layout directly}=0 (don't transpose)
     --print-status {show status of buffer and socket}=0
     --print-config {show current algorithm configuration}=0
     --write-configuration {write current algorithm configuration to file}=0
     -n, --number-of-events {number of events to process}=0 (all)
     -s, --number-of-slices {number of input slices to allocate}=0 (one more than the number of threads)
     --events-per-slice {number of events per slice}=1000
     -t, --threads {number of threads / streams}=1
     -r, --repetitions {number of repetitions per thread / stream}=1
     -m, --memory {memory to reserve on the device per thread / stream (megabytes)}=1000
     --host-memory {memory to reserve on the host per thread / stream (megabytes)}=200
     -v, --verbosity {verbosity [0-5]}=3 (info)
     -p, --print-memory {print memory usage}=0
     --sequence {sequence to run}
     --output-file {Write selected event to output file}
     --device {select device to use}=0
     --non-stop {Runs the program indefinitely}=0
     --with-mpi {Read events with MPI}
     --mpi-window-size {Size of MPI sliding window}=4
     --mpi-number-of-slices {Number of MPI network slices}=6
     --inject-mem-fail {Whether to insert random memory failures (0: off 1-15: rate of 1 in 2^N)}=0
     --monitoring-filename {ROOT file to write monitoring histograms to}=monitoring.root
     --monitoring-save-period {Number of seconds between writes of the monitoring histograms (0: off)}=0
     --disable-run-changes {Ignore signals to update non-event data with each run change}=1
     -h {show this help}

Here are some examples for run options. Note that if Allen was :ref:`built with cvmfs<build with cvmfs>`, one can prepend `./toolchain/wrapper` to all the following commands to execute in the correct environment.  ::

    # Run on an MDF input file shipped with Allen once
    ./Allen --sequence hlt1_pp_default --mdf ../input/minbias/mdf/MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster_v1.mdf

    # Run a total of 1000 events once with validation
    ./Allen --sequence hlt1_pp_validation -n 1000 --mdf /path/to/mdf/input/file

    # Run four streams, each with 4000 events and 20 repetitions
    ./Allen --sequence hlt1_pp_default -t 4 -n 4000 -r 20 --mdf /path/to/mdf/input/file

    # Run one stream with 5000 events and print all memory allocations
    ./Allen --sequence hlt1_pp_default -n 5000 -p 1 --mdf /path/to/mdf/input/file

    # Run on all events in all files listed in file.lst; four streams
    # with batches of 1000 events
    find /some/directory/with/files -type f | sort > files.lst
    ./Allen --sequence hlt1_pp_default -t 4 --events-per-slice 1000 --mdf /path/to/files.lst

.. _run_allen_in_gaudi_moore_eventloop:

As Gaudi project, event loop steered by Moore (offline)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use Gaudi to update non-event data such as alignment and configuration constants and use Moore to steer the event loop and call Allen one event at a time (this method will be used for the offline workflow).
To run Allen as the HLT1 trigger application, call the following options script from within the stack directory::

  ./Moore/run gaudirun.py Moore/Hlt/Moore/tests/options/default_input_and_conds_hlt1_retinacluster.py Moore/Hlt/Hlt1Conf/options/allen_hlt1_pp_default.py

To run a different sequence, the function call that sets up the
control flow can be wrapped using a `with` statement::

  from RecoConf.hlt1_allen import allen_gaudi_node_barriers
  with allen_gaudi_node_barriers.bind(sequence="hlt1_pp_no_gec"):
    run_allen(options)

How to study the HLT1 physics performance within Moore is described in :ref:`moore_performance_scripts`.

.. _run_allen_in_gaudi_allen_eventloop:

As Gaudi project, event loop steered by Allen (data-taking)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use Gaudi to update non-event data such as alignment and configuration constants and use Allen to steer the event loop, where batches of events (O(1000)) are processed together (this method will be used for data-taking).

When using MDF files as input, call from the Allen environment::

  ./Allen/build.${ARCHITECTURE}/run python Dumpers/BinaryDumpers/options/allen.py --mdf Allen/input/minbias/mdf/MiniBrunel_2018_MinBias_FTv4_DIGI_retinacluster_v1.mdf

When using MEP files as input, call from the MooreOnline environment, as MEP handling is implemented there::

  ./MooreOnline/build.${ARCHITECTURE}/run python Allen/Dumpers/BinaryDumpers/options/allen.py --sequence=Allen/InstallArea/${ARCHITECTURE}/constants/hlt1_pp_default.json --tags="dddb_tag,simcond_tag" --mep mep_file.mep
