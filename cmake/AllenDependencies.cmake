###############################################################################
# (c) Copyright 2000-2021 CERN for the benefit of the LHCb Collaboration      #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
if (NOT STANDALONE)
  message(STATUS "LHCb stack build")

  set(BUILD_TESTING ON)

  # Find modules we need
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules)

  if(NOT COMMAND lhcb_find_package)
    # Look for LHCb find_package wrapper
    find_file(LHCbFindPackage_FILE LHCbFindPackage.cmake)
    if(LHCbFindPackage_FILE)
        include(${LHCbFindPackage_FILE})
    else()
        # if not found, use the standard find_package
        macro(lhcb_find_package)
            find_package(${ARGV})
        endmacro()
    endif()
  endif()

  # -- Public dependencies
  lhcb_find_package(Rec 34.0 REQUIRED)

  find_package(AIDA REQUIRED)
  find_package(fmt REQUIRED)
  find_package(TBB REQUIRED)

  # Detect device target from binary tag
  set(target_hip FALSE)
  set(target_cuda FALSE)

  string(REPLACE "+" ";" compiler_split "${LCG_COMPILER}")
  foreach(device_comp IN LISTS compiler_split)
    # "${device_comp}" MATCHES "cuda([0-9]+)_([0-9]+)((gcc|clang)([0-9]+))?" OR
    if (DEFINED CMAKE_CUDA_COMPILER)
      set(target_cuda TRUE)
    elseif ("${device_comp}" STREQUAL "hip")
      set(target_hip TRUE)
    endif()
  endforeach()

  if(${target_hip} AND NOT ${target_cuda})
    set(device "HIP")
  elseif(${target_cuda} AND NOT ${target_hip})
    set(device "CUDA")
  elseif(${target_cuda} AND ${target_hip})
  	message(FATAL_ERROR "Cannot simultaneously build for HIP and CUDA targets")
  else()
	set(device "CPU")
  endif()

  set(TARGET_DEVICE ${device} CACHE STRING "Target architecture of the device")
endif()

message(STATUS "Allen device target: " ${TARGET_DEVICE})

# Device runtime libraries
if(TARGET_DEVICE STREQUAL "CUDA")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
elseif(TARGET_DEVICE STREQUAL "HIP")
  # Setup HIPCC compiler
  if(NOT DEFINED ROCM_PATH)
    if(NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "/opt/rocm" CACHE PATH "Path where ROCM has been installed")
    else()
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path where ROCM has been installed")
    endif()
  endif()

  set(HIP_PATH "${ROCM_PATH}/hip")
  set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin")
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP QUIET REQUIRED)

  # Do this by hand because ROCm cmake is not relocatable
  find_library(HIP_RUNTIME_LIB amdhip64
    HINTS
      ${HIP_PATH}/lib
    NO_DEFAULT_PATH)

  message(STATUS "Found HIP: " ${HIP_VERSION})
  message(STATUS "HIP runtime: ${HIP_RUNTIME_LIB}")
endif()

find_package(cppgsl REQUIRED)

# std::filesytem detection
find_package(Filesystem REQUIRED)

find_package(umesimd REQUIRED)

find_package(PkgConfig)
pkg_check_modules(zmq libzmq REQUIRED IMPORTED_TARGET)
pkg_check_modules(sodium libsodium REQUIRED IMPORTED_TARGET)
if(NOT STANDALONE)
  pkg_check_modules(git2 libgit2 REQUIRED IMPORTED_TARGET)  # for GitEntityResolver
endif()

if(WITH_Allen_PRIVATE_DEPENDENCIES)
  # We need a Python 3 interpreter
  find_package(Python 3 REQUIRED Interpreter Development.Module)

  # Catch2 for tests
  find_package(Catch2 REQUIRED)

  # Find libClang, required for parsing the Allen codebase
  find_package(Clang QUIET)
  if (TARGET libclang)
    get_target_property(LIBCLANG_LIBDIR libclang IMPORTED_LOCATION_RELEASE)
    get_filename_component(LIBCLANG_LIBDIR "${LIBCLANG_LIBDIR}" PATH)
  elseif(EXISTS /cvmfs/sft.cern.ch)
    # As a last resort, try a hard-coded directory in cvmfs
    set(LIBCLANG_LIBDIR /cvmfs/lhcb.cern.ch/lib/lcg/releases/clang/12.0.0/x86_64-${LCG_OS}/lib)
    set(LIBCLANG_ALTERNATIVE_FOUND ON)
    message(STATUS "Using predefined CVMFS libclang directory")
  else()
    message(FATAL_ERROR "No suitable libClang installation found. "
                        "You may provide a custom path by setting LIBCLANG_LIBDIR manually")
  endif()
  message(STATUS "Found libclang at ${LIBCLANG_LIBDIR}")

  # https://github.com/nlohmann/json
  find_package(nlohmann_json REQUIRED)
  find_package(Threads REQUIRED)

  # Boost
  find_package(Boost 1.75 REQUIRED COMPONENTS filesystem iostreams thread regex
    serialization program_options unit_test_framework headers)

  if(NOT STANDALONE)
    find_package(Rangev3 REQUIRED)
    find_package(yaml-cpp REQUIRED)

    # pybind11 is available in LCG, but it's installed with setup.py,
    # so the CMake files are in a non-standard location and we have to
    # make sure we can find them
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir(), end=\"\");" OUTPUT_VARIABLE PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
    find_package(pybind11 CONFIG REQUIRED)
  endif()
endif()

if (STANDALONE)
  # Support for STANDALONE without CMAKE_PREFIX_PATH
  if(EXISTS $ENV{ROOTSYS}/cmake/ROOTConfig.cmake) # ROOT was compiled with cmake
    set(ALLEN_ROOT_CMAKE $ENV{ROOTSYS})
  elseif(EXISTS $ENV{ROOTSYS}/ROOTConfig.cmake)
    set(ALLEN_ROOT_CMAKE $ENV{ROOTSYS})
  elseif($ENV{ROOTSYS}) # ROOT was compiled with configure/make
    set(ALLEN_ROOT_CMAKE $ENV{ROOTSYS}/etc)
  endif()
else()
  set(Allen_PERSISTENT_OPTIONS TARGET_DEVICE)
endif()

find_package(ROOT REQUIRED HINTS ${ALLEN_ROOT_CMAKE} COMPONENTS RIO Core Cling Hist Tree)
if (NOT ROOT_FOUND)
  message(FATAL_ERROR "ROOT could not be found, please either set ROOTSYS or alternatively add the ROOT path to the CMAKE_PREFIX_PATH")
endif()
message(STATUS "Found ROOT: " ${ROOT_INCLUDE_DIRS})

find_package(TBB REQUIRED)
message(STATUS "Found TBB version " ${TBB_VERSION})
