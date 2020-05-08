Allen
=====

Welcome to Allen, a project providing a full HLT1 realization on GPU.

Requisites
----------
The project requires CMake 3.12, Python3 and a [compiler supporting C++17](https://en.cppreference.com/w/cpp/compiler_support).
Further requirements depend on the device chosen as target. For each target,
we show a proposed development setup with CVMFS and CentOS 7:

* CPU target: Any modern compiler can be used, such as gcc greater than 7.0:
    
    ```shell
    source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_97python3 x86_64-centos7-gcc8-opt
    ```
    
* CUDA target: The latest supported compilers are gcc-8 and clang-6. CUDA is
  available in cvmfs as well:

    ```shell
    source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_97python3 x86_64-centos7-gcc8-opt
    source /cvmfs/sft.cern.ch/lcg/contrib/cuda/10.2/x86_64-centos7/setup.sh
    ```
    
* HIP target: A local installation of ROCm at least version 3.3.0 is required.

* CUDACLANG target: A version of the clang compiler with ptx support is required,
  alongside a local installation of CUDA 10.1 (currently latest supported release):

    ```shell
    source /cvmfs/sft.cern.ch/lcg/releases/clang/10.0.0/x86_64-centos7/setup.sh
    ```

Optionally you can compile the project with ROOT. Then, trees will be filled with variables to check when running the UT tracking or SciFi tracking algorithms on x86 architecture.
In addition, histograms of reconstructible and reconstructed tracks are then filled in the track checker. For more details on how to use them to produce plots of efficiencies, momentum resolution etc. see [this readme](checker/tracking/readme.md).

[Building and running inside Docker](readme_docker.md)

Where to find input
-------------
Input from 5k events for each of the following decay modes can be found here:

* minimum bias, mag down: `/eos/lhcb/wg/rta/WP6/Allen/binary_input_2019-07/minbias/minbias_mag_down.tar.gz`
* Bs->PhiPhi, mag down: `/eos/lhcb/wg/rta/WP6/Allen/binary_input_2019-07/Bs2PhiPhi/mag_down.tar.gz`
* Bs->PhiPhi, mag up: `/eos/lhcb/wg/rta/WP6/Allen/binary_input_2019-07/Bs2PhiPhi/mag_up.tar.gz`
* J/Psi->MuMu, mag down: `/eos/lhcb/wg/rta/WP6/Allen/binary_input_2019-07/JpsiMuMu/mag_down.tar.gz`
* Ds->KKPi, mag down: `/eos/lhcb/wg/rta/WP6/Allen/binary_input_2019-07/Ds2KKPi/Ds2KKPi_mag_down.tar.gz`
* B->KstEE, mag down: `/eos/lhcb/wg/rta/WP6/Allen/binary_input_2019-07/KstEE/KstEE_mag_down.tar.gz`
* B->KstMuMu, mag down: `/eos/lhcb/wg/rta/WP6/Allen/binary_input_2019-07/KstMuMu/KstMuMu_mag_down.tar.gz`
* Z->MuMu, mag down: `/eos/lhcb/wg/rta/WP6/Allen/binary_input_2019-07/Z2MuMu/Z2MuMu_mag_down.tar.gz`
* Ks0->MuMu, mag down: `/eos/lhcb/wg/rta/WP6/Allen/binary_input_2019-07/Ks0mumu/Ks02MuMu_mag_down.tar.gz`

If other inputs are required, follow these instructions for producing them:
[https://gitlab.cern.ch/lhcb/Allen/blob/allen_tdr/Dumpers/readme.md](https://gitlab.cern.ch/lhcb/Allen/blob/allen_tdr/Dumpers/readme.md)

Allen selections require ODIN banks, which were not included with these samples. Random ODIN banks can be generated using `makeODIN.py`. From the Allen root directory:

```shell
python3 scripts/makeODIN.py /path/to/data/banks/
```

This will create a random ODIN bank for each bank in `/path/to/data/banks/VP`.

How to build it
---------------

The build process doesn't differ from standard cmake projects:

    mkdir build
    cd build
    cmake ..
    make

There are some cmake options to configure the build process:

* The sequence can be configured by specifying `-DSEQUENCE=<name_of_sequence>`. For a complete list of sequences available, check `configuration/sequences/`. Sequence names should be specified without the `.h`, ie. `-DSEQUENCE=VeloPVUTSciFiDecoding`.
* The build type can be specified to `RelWithDebInfo`, `Release` or `Debug`, e.g. `cmake -DCMAKE_BUILD_TYPE=Debug ..`
* ROOT can be enabled to generate monitoring plots using `-DUSE_ROOT=ON`
* If more verbose build output from the CUDA toolchain is desired, specify `-DCUDA_VERBOSE_BUILD=ON`
* If multiple versions of CUDA are installed and CUDA 10.0 is not the default, it can be specified using: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.0/bin/nvcc`

How to run it
-------------

Some binary input files are included with the project for testing.
A run of the program with the help option `-h` will let you know the basic options:

    Usage: ./Allen -h
    -f, --folder {folder containing data directories}=../input/minbias/
     -g, --geometry {folder containing detector configuration}=../input/detector_configuration/down/
    --mdf {comma-separated list of MDF files to use as input}
    --mep {comma-separated list of MEP files to use as input}
    --transpose-mep {Transpose MEPs instead of decoding from MEP layout directly}=0 (don't transpose)
    --configuration {path to json file containing values of configurable algorithm constants}=../configuration/constants/default.json
    --print-status {show status of buffer and socket}=0
    --print-config {show current algorithm configuration}=0
    --write-configuration {write current algorithm configuration to file}=0
    -n, --number-of-events {number of events to process}=0 (all)
    -s, --number-of-slices {number of input slices to allocate}=0 (one more than the number of threads)
    --events-per-slice {number of events per slice}=1000
    -t, --threads {number of threads / streams}=1
    -r, --repetitions {number of repetitions per thread / stream}=1
    -c, --validate {run validation / checkers}=1
    -m, --memory {memory to reserve per thread / stream (megabytes)}=1024
    -v, --verbosity {verbosity [0-5]}=3 (info)
    -p, --print-memory {print memory usage}=0
    -i, --import-tracks {import forward tracks dumped from Brunel}
    --cpu-offload {offload part of the computation to CPU}=1
    --output-file {Write selected event to output file}
    --device {select device to use}=0
    --non-stop {Runs the program indefinitely}=0
    --with-mpi {Read events with MPI}
    --mpi-window-size {Size of MPI sliding window}=4
    --mpi-number-of-slices {Number of MPI network slices}=6
    -h {show this help}


Here are some example run options:

    # Run all input files once with the tracking validation
    ./Allen

    # Specify input files, run once over all of them with tracking validation
    ./Allen -f ../input/minbias/

    # Run a total of 1000 events, round robin over the existing ones, without tracking validation
    ./Allen -c 0 -n 1000

    # Run four streams, each with 4000 events, 20 repetitions
    ./Allen -t 4 -n 4000 -r 20 -c 0

    # Run one stream and print all memory allocations
    ./Allen -n 5000 -p

How to enable Nvidia persistenced mode
-----------------------------------------
Enabling Nvidia [persistenced mode](https://docs.nvidia.com/deploy/driver-persistence/index.html) will increase the throughput of Allen, as the GPU will remain initialized even when no process is running. To enable:
`sudo systemctl enable nvidia-persistenced`, reboot the machine.

How to profile it
------------------
For profiling, Nvidia's nvprof can be used.
Since CUDA version 10.1, profiling was limited to the root user by default for security reasons. However, the system administrator of a GPU server can add a kernel module option such that regular users can use the profiler by following these instructions:

Add a file containing "option nvidia NVreg_RestrictProfilingToAdminUsers=0" to the `/etc/modprobe.d/` directory and reboot the machine. This will load the nvidia kernel module with "NVreg_RestrictProfilingToAdminUsers=0".

As a quick workaround one can also use the older version of nvprof:

    /usr/local/cuda-10.0/bin/nvprof ./Allen -c 0 -n 1000

Building as a Gaudi/LHCb project
--------------------------------

Allen can also be built as a Gaudi/LHCb cmake project; it then depends
on Rec and Online. To build Allen like this, is the same as building
any other Gaudi/LHCb project:

    LbLogin -c x86_64-centos7-gcc9-opt
    cd Allen
    lb-project-init
    make configure
    make install

### Build options
By default the `DefaultSequence` is selected, Allen is built with
CUDA, and the CUDA stack is searched for in `/usr/local/cuda`. These
defaults (and other cmake variables) can be changed by adding the same
flags that you would pass to a standalone build to the `CMAKEFLAGS`
environment variable before calling `make configure`.

For example, to specify another CUDA stack to be used set:
```console
$> export CMAKEFLAGS="-DCMAKE_CUDA_COMPILER=/path/to/alternative/nvcc"
```

### Runtime environment:
To setup the runtime environment for Allen, the same tools as for
other Gaudi/LHCb projects can be used:
```console
$> cd Allen
$> ./build.${BINARY_TAG}/run Allen ...
```

### Run Allen using the Python entry point:
```console
$> cd Allen
$> ./build.${CMTCONFIG}/run bindings/Allen.py
```

### Links to more readmes
The following readmes explain various aspects of Allen:

* [This readme](contributing.md) explains how to add a new algorithm to Allen.
* [This readme](selections.md ) explains how to add a new HLT1 line to Allen.
* [This readme](configuration/readme.md) explains how to configure the algorithms in an HLT1 sequence.
* [This readme](Rec/Allen/readme.md) explains how to call Allen from Moore and Brunel.
