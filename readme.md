Allen
=====

Welcome to Allen, a project with the aim to provide a full HLT1 realization on GPU.

Requisites
----------
The project requires a graphics card with CUDA support, CUDA 10.0 and a compiler supporting C++14.

If you are working from a node with CVMFS and CentOS 7, we suggest the following setup:

```shell
source /cvmfs/lhcb.cern.ch/lib/lcg/releases/gcc/7.3.0/x86_64-centos7/setup.sh
```

Regardless of the OS you are running on, you can check your compiler versions as follows:

```shell
$ g++ --version
g++ (GCC) 7.3.0

$ nvcc --version
Cuda compilation tools, release 9.2, V9.2.88
```

You can check your compiler standard compatibility by scrolling to the `C++14 features` chart [here](https://en.cppreference.com/w/cpp/compiler_support).

Optionally you can compile the project with ROOT. Then, trees will be filled with variables to check when running the UT tracking or SciFi tracking algorithms on x86 architecture.
In addition, histograms of reconstructible and reconstructed tracks are then filled in the track checker, they are saved in the file `output/PrCheckerPlots.root`. 
Plots of efficiencies versus various kinematic variables can be created by running `efficiency_plots.py` in the directory `checker/tracking/python_scripts`.

You can setup ROOT in CVMFS as follows:

```shell
source /cvmfs/lhcb.cern.ch/lib/lcg/releases/ROOT/6.08.06-d7e12/x86_64-centos7-gcc62-opt/bin/thisroot.sh
```

[Building and running inside Docker](readme_docker.md)

Where to find input
-------------
Input from 1k events can be found here: 

* minimum bias (for performance checks): `/afs/cern.ch/work/d/dovombru/public/gpu_input/1kevents_minbias_dump_PV_truth.tar.gz`
* Bs->PhiPhi (for efficiency checks): `/afs/cern.ch/work/d/dovombru/public/gpu_input/1kevents_BsPhiPhi_dump_PV_truth.tar.gz`

How to run it
-------------

The build process doesn't differ from standard cmake projects:

    mkdir build
    cd build
    cmake ..
    make

There are some cmake options to configure the build process:

* The sequence can be configured by specifying `-DSEQUENCE=<name_of_sequence>`. For a complete list of sequences available, check `configuration/sequences/`. Sequence names should be specified without the `.h`, ie. `-DSEQUENCE=VeloUT`.
* The build type can be specified to `RelWithDebInfo`, `Release` or `Debug`, e.g. `cmake -DCMAKE_BUILD_TYPE=Debug ..`
* If ROOT is available, it can be enabled to generate graphs by `-DUSE_ROOT=ON`
* Verbose compilation can be turned on with `-DVERBOSE_COMPILATION=TRUE`, by default it is set to `FALSE`.

The MC validation is standalone, it was written by
Manuel Schiller, Rainer Schwemmer, Daniel Cámpora and Dorothea vom Bruch.

Some binary input files are included with the project for testing.
A run of the program with no arguments will let you know the basic options:

    Usage: ./Allen
    -f {folder containing directories with raw bank binaries for every sub-detector}
    -g {folder containing detector configuration}
    -d {folder containing bin files with MC truth information}
    -n {number of events to process}=0 (all)
    -o {offset of events from which to start}=0 (beginning)
    -t {number of threads / streams}=1
    -r {number of repetitions per thread / stream}=1
    -c {run checkers}=0
    -k {simplified kalman filter}=0
    -m {reserve Megabytes}=1024
    -v {verbosity}=3 (info)
    -p (print memory usage)
    -x {run algorithms on x86 architecture as well (if possible)}=0


Here are some example run options:

    # Run all input files once with the tracking validation
    ./Allen

    # Specify input files, run once over all of them with tracking validation
    ./Allen -f ../input/minbias/banks/ -d ../input/minbias/MC_info/

    # Run a total of 1000 events, round robin over the existing ones, without tracking validation
    ./Allen -c 0 -n 1000

    # Run four streams, each with 4000 events, 20 repetitions
    ./Allen -t 4 -n 4000 -r 20 -c 0

    # Run one stream and print all memory allocations
    ./Allen -n 5000 -p


[This readme](contributing.md) explains how to add a new algorithm to the sequence and how to use the memory scheduler to define global memory variables for this sequence and pass on the dependencies. It also explains which checks to do before placing a merge request with your changes.
