###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
###############################################################################

# Deal with configuration generation machinery
# * If clang is available, we can and will generate the configuration files
# * Otherwise, fail and message that it is not possible to generate configurations
set(CODE_GENERATION_DIR ${PROJECT_BINARY_DIR}/code_generation)
set(PROJECT_SEQUENCE_DIR ${CODE_GENERATION_DIR}/sequences)
set(SEQUENCE_DEFINITION_DIR ${PROJECT_SEQUENCE_DIR}/AllenConf)
set(ALLEN_ALGORITHMS_DIR ${PROJECT_SEQUENCE_DIR}/AllenAlgorithms)
set(ALLEN_GENERATED_INCLUDE_FILES_DIR ${PROJECT_SEQUENCE_DIR}/include)
set(ALLEN_CORE_DIR ${PROJECT_SEQUENCE_DIR}/AllenCore)
set(ALLEN_PARSER_DIR ${PROJECT_SEQUENCE_DIR}/parser)
set(ALGORITHMS_OUTPUTFILE ${ALLEN_ALGORITHMS_DIR}/algorithms.py)
set(PARSED_ALGORITHMS_OUTPUTFILE ${CODE_GENERATION_DIR}/parsed_algorithms.pickle)
set(ALGORITHMS_GENERATION_SCRIPT ${PROJECT_SOURCE_DIR}/configuration/parser/ParseAlgorithms.py)

include_guard(GLOBAL)

file(MAKE_DIRECTORY ${CODE_GENERATION_DIR})
file(MAKE_DIRECTORY ${ALLEN_PARSER_DIR})
file(MAKE_DIRECTORY ${ALLEN_GENERATED_INCLUDE_FILES_DIR})
file(MAKE_DIRECTORY ${ALLEN_ALGORITHMS_DIR})

set(MINIMUM_REQUIRED_LIBCLANG_VERSION 9)
if(LIBCLANG_FOUND AND "${LIBCLANG_MAJOR_VERSION}" LESS ${MINIMUM_REQUIRED_LIBCLANG_VERSION})
  message(STATUS "libClang version found (${LIBCLANG_VERSION}) does not meet minimum version requirement (${MINIMUM_REQUIRED_LIBCLANG_VERSION})")
endif()

if(NOT LIBCLANG_FOUND OR "${LIBCLANG_MAJOR_VERSION}" LESS ${MINIMUM_REQUIRED_LIBCLANG_VERSION})
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # In macOS, libClang typically exists even if llvm-config does not exist.
    # Attempt default directory
    set(LIBCLANG_LIBDIR /Library/Developer/CommandLineTools/usr/lib)
    set(LIBCLANG_ALTERNATIVE_FOUND ON)
    message(STATUS "Using predefined macos libclang directory")
  elseif(EXISTS /cvmfs/sft.cern.ch)
    # As a last resource, try a hard-coded directory in cvmfs
    set(LIBCLANG_LIBDIR /cvmfs/sft.cern.ch/lcg/releases/clang/11.1.0-b24ba/x86_64-centos7/lib)
    set(LIBCLANG_ALTERNATIVE_FOUND ON)
    message(STATUS "Using predefined CVMFS libclang directory")
  else()
    message(FATAL_ERROR "No suitable libClang installation found. "
                        "You may provide a custom path to llvm-config by setting LLVM_CONFIG manually")
  endif()
endif()

message(STATUS "Found libclang at ${LIBCLANG_LIBDIR}")

# Macos requires DYLD_LIBRARY_PATH, otherwise LD_LIBRARY_PATH
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(LIBRARY_PATH_VARNAME "DYLD_LIBRARY_PATH")
else()
  set(LIBRARY_PATH_VARNAME "LD_LIBRARY_PATH")
endif()

# We will invoke the parser a few times, set its required environment in a variable
set(PARSER_ENV PYTHONPATH=$ENV{PYTHONPATH}:${PROJECT_SOURCE_DIR}/scripts ${LIBRARY_PATH_VARNAME}=${LIBCLANG_LIBDIR}:$ENV{LD_LIBRARY_PATH})

# Parse Allen algorithms
# TODO: Parsing should depend on ALL algorithm headers and ALL algorithm sources
add_custom_command(
  OUTPUT "${PARSED_ALGORITHMS_OUTPUTFILE}"
  COMMENT "Parsing Allen algorithms"
  COMMAND
    ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate parsed_algorithms --filename "${PARSED_ALGORITHMS_OUTPUTFILE}" --prefix_project_folder "${PROJECT_SOURCE_DIR}"
  DEPENDS "${PROJECT_SOURCE_DIR}/configuration/parser/ParseAlgorithms.py")

# Symlink Allen build directories
add_custom_command(
  OUTPUT "${SEQUENCE_DEFINITION_DIR}" "${ALLEN_CORE_DIR}"
  COMMENT "Making symlink of sequence definitions and configuration utilities"
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink "${PROJECT_SOURCE_DIR}/configuration/python/AllenConf" "${SEQUENCE_DEFINITION_DIR}" &&
    ${CMAKE_COMMAND} -E create_symlink "${PROJECT_SOURCE_DIR}/configuration/AllenCore" "${ALLEN_CORE_DIR}"
  DEPENDS "${PROJECT_SOURCE_DIR}/configuration/python/AllenConf" "${PROJECT_SOURCE_DIR}/configuration/AllenCore")
add_custom_target(generate_conf_core DEPENDS "${SEQUENCE_DEFINITION_DIR}" "${ALLEN_CORE_DIR}")

# Generate algorithms.py
add_custom_command(
  OUTPUT "${ALGORITHMS_OUTPUTFILE}"
  COMMAND
    ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate views --filename "${ALGORITHMS_OUTPUTFILE}" --parsed_algorithms "${PARSED_ALGORITHMS_OUTPUTFILE}" &&
    ${CMAKE_COMMAND} -E touch ${ALLEN_ALGORITHMS_DIR}/__init__.py
  WORKING_DIRECTORY ${ALLEN_PARSER_DIR}
  DEPENDS "${PARSED_ALGORITHMS_OUTPUTFILE}" generate_conf_core)
add_custom_target(generate_algorithms_view DEPENDS "${ALGORITHMS_OUTPUTFILE}")

# Generate Allen AlgorithmDB
add_custom_command(
  OUTPUT "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/AlgorithmDB.h"
  COMMENT "Generating AlgorithmDB"
  COMMAND ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate db --filename "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/AlgorithmDB.h" --parsed_algorithms "${PARSED_ALGORITHMS_OUTPUTFILE}"
  WORKING_DIRECTORY ${ALLEN_PARSER_DIR}
  DEPENDS "${PARSED_ALGORITHMS_OUTPUTFILE}")
add_custom_target(algorithm_db_generation DEPENDS "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/AlgorithmDB.h")
add_library(algorithm_db INTERFACE)
add_dependencies(algorithm_db algorithm_db_generation)
target_include_directories(algorithm_db INTERFACE $<BUILD_INTERFACE:${ALLEN_GENERATED_INCLUDE_FILES_DIR}>)
install(TARGETS algorithm_db
      EXPORT Allen
      LIBRARY DESTINATION lib)

# Generate StructToTuple.cuh
add_custom_command(
  OUTPUT "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/StructToTuple.cuh"
  COMMAND
    ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate struct_to_tuple --filename "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/StructToTuple.cuh" --parsed_algorithms "${PARSED_ALGORITHMS_OUTPUTFILE}" --struct_to_tuple_folder "${PROJECT_SOURCE_DIR}/configuration/parser/struct_to_tuple"
  WORKING_DIRECTORY ${ALLEN_PARSER_DIR}
  DEPENDS "${PARSED_ALGORITHMS_OUTPUTFILE}")
add_custom_target(struct_to_tuple_generation DEPENDS "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/StructToTuple.cuh")
add_library(struct_to_tuple INTERFACE)
add_dependencies(struct_to_tuple struct_to_tuple_generation)
target_include_directories(struct_to_tuple INTERFACE $<BUILD_INTERFACE:${ALLEN_GENERATED_INCLUDE_FILES_DIR}>)
install(TARGETS struct_to_tuple
      EXPORT Allen
      LIBRARY DESTINATION lib)

if(SEPARABLE_COMPILATION)
  add_custom_command(
    OUTPUT "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/ExternLines.cuh"
    COMMAND
      ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate extern_lines --filename "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/ExternLines.cuh" --parsed_algorithms "${PARSED_ALGORITHMS_OUTPUTFILE}"
    WORKING_DIRECTORY ${ALLEN_PARSER_DIR}
    DEPENDS "${PARSED_ALGORITHMS_OUTPUTFILE}")
else()
  add_custom_command(
    OUTPUT "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/ExternLines.cuh"
    COMMAND
      ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate extern_lines_nosepcomp --filename "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/ExternLines.cuh" --parsed_algorithms "${PARSED_ALGORITHMS_OUTPUTFILE}"
    WORKING_DIRECTORY ${ALLEN_PARSER_DIR}
    DEPENDS "${PARSED_ALGORITHMS_OUTPUTFILE}")
endif()
add_custom_target(extern_lines_generation DEPENDS "${ALLEN_GENERATED_INCLUDE_FILES_DIR}/ExternLines.cuh")
add_library(extern_lines INTERFACE)
add_dependencies(extern_lines extern_lines_generation)
target_include_directories(extern_lines INTERFACE $<BUILD_INTERFACE:${ALLEN_GENERATED_INCLUDE_FILES_DIR}>)
install(TARGETS extern_lines
      EXPORT Allen
      LIBRARY DESTINATION lib)

if(NOT STANDALONE AND TARGET_DEVICE STREQUAL "CPU")
  # We need to get the list of algorithms at configuration time in order to
  # know the list of files that will be required of this build
  set(ALGORITHM_WRAPPERS_FOLDER ${CODE_GENERATION_DIR}/algorithm_wrappers)
  set(ALGORITHM_WRAPPERS_LISTFILE ${ALGORITHM_WRAPPERS_FOLDER}/algorithm_list.txt)
  file(MAKE_DIRECTORY ${ALGORITHM_WRAPPERS_FOLDER})
  execute_process(COMMAND ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate wrapperlist --filename "${ALGORITHM_WRAPPERS_LISTFILE}" --algorithm_wrappers_folder "${ALGORITHM_WRAPPERS_FOLDER}" --prefix_project_folder "${PROJECT_SOURCE_DIR}")
  file(READ "${ALGORITHM_WRAPPERS_LISTFILE}" WRAPPED_ALGORITHM_SOURCES) # WRAPPED_ALGORITHM_SOURCES="a.cpp b.cpp c.cpp"

  # Build step that will produce all .cpp conversion files
  add_custom_command(
    OUTPUT ${WRAPPED_ALGORITHM_SOURCES}
    COMMENT "Generating wrapped algorithm sources"
    COMMAND
      ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate wrappers --parsed_algorithms "${PARSED_ALGORITHMS_OUTPUTFILE}" --algorithm_wrappers_folder "${ALGORITHM_WRAPPERS_FOLDER}"
    WORKING_DIRECTORY ${PROJECT_SEQUENCE_DIR}
    DEPENDS "${PARSED_ALGORITHMS_OUTPUTFILE}")
elseif(STANDALONE)
  find_package(Git REQUIRED)
  add_custom_command(
    OUTPUT "${PROJECT_SEQUENCE_DIR}/LHCb" "${PROJECT_SEQUENCE_DIR}/PyConf" "${PROJECT_SEQUENCE_DIR}/Gaudi" "${PROJECT_SEQUENCE_DIR}/GaudiKernel"
    COMMENT "Checking out configuration utilities from the LHCb stack"
    COMMAND
      ${CMAKE_COMMAND} -E env ${GIT_EXECUTABLE} clone https://gitlab.cern.ch/lhcb/LHCb.git --no-checkout &&
      ${CMAKE_COMMAND} -E env ${GIT_EXECUTABLE} --work-tree=LHCb --git-dir=LHCb/.git checkout HEAD -- PyConf &&
      ${CMAKE_COMMAND} -E env ${GIT_EXECUTABLE} clone https://gitlab.cern.ch/gaudi/Gaudi.git --no-checkout &&
      ${CMAKE_COMMAND} -E env ${GIT_EXECUTABLE} --work-tree=Gaudi --git-dir=Gaudi/.git checkout HEAD -- GaudiKernel &&
      ${CMAKE_COMMAND} -E create_symlink LHCb/PyConf/python/PyConf PyConf &&
      ${CMAKE_COMMAND} -E create_symlink Gaudi/GaudiKernel/python/GaudiKernel GaudiKernel
    WORKING_DIRECTORY ${PROJECT_SEQUENCE_DIR})
  add_custom_target(checkout_gaudi_dirs DEPENDS "${PROJECT_SEQUENCE_DIR}/LHCb" "${PROJECT_SEQUENCE_DIR}/PyConf" "${PROJECT_SEQUENCE_DIR}/Gaudi" "${PROJECT_SEQUENCE_DIR}/GaudiKernel")
endif()

function(generate_sequence sequence)
  set(sequence_dir ${PROJECT_SEQUENCE_DIR}/${sequence})
  file(MAKE_DIRECTORY ${sequence_dir})
  if(NOT STANDALONE)
    configure_file(${PROJECT_SOURCE_DIR}/scripts/generate_script.sh.in ${sequence_dir}/generate_${sequence}.sh @ONLY)
    add_custom_command(
      OUTPUT "${PROJECT_BINARY_DIR}/${sequence}.json"
      COMMAND
        ${CMAKE_BINARY_DIR}/run bash ${sequence_dir}/generate_${sequence}.sh &&
        ${CMAKE_COMMAND} -E rename "${sequence_dir}/Sequence.json" "${PROJECT_BINARY_DIR}/${sequence}.json"
      DEPENDS "${PROJECT_SOURCE_DIR}/configuration/sequences/${sequence}.py" generate_algorithms_view
      WORKING_DIRECTORY ${sequence_dir})
  else()
    add_custom_command(
      OUTPUT "${PROJECT_BINARY_DIR}/${sequence}.json"
      COMMAND
        ${CMAKE_COMMAND} -E env "${LIBRARY_PATH_VARNAME}=$ENV{LD_LIBRARY_PATH}" "PYTHONPATH=${PROJECT_SEQUENCE_DIR}:$ENV{PYTHONPATH}" "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/configuration/sequences/${sequence}.py" &&
        ${CMAKE_COMMAND} -E rename "${sequence_dir}/Sequence.json" "${PROJECT_BINARY_DIR}/${sequence}.json"
      DEPENDS "${PROJECT_SOURCE_DIR}/configuration/sequences/${sequence}.py" generate_algorithms_view checkout_gaudi_dirs
      WORKING_DIRECTORY ${sequence_dir})
  endif()
  add_custom_target(sequence_${sequence} DEPENDS "${PROJECT_BINARY_DIR}/${sequence}.json")
  add_dependencies(Stream sequence_${sequence})
  install(FILES "${PROJECT_BINARY_DIR}/${sequence}.json" DESTINATION constants)
endfunction()
