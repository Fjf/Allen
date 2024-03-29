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
set(ALGORITHMS_OUTPUTFILE ${ALLEN_ALGORITHMS_DIR}/allen_standalone_algorithms.py)
set(PARSED_ALGORITHMS_OUTPUTFILE ${CODE_GENERATION_DIR}/parsed_algorithms.pickle)
set(ALGORITHMS_GENERATION_SCRIPT ${PROJECT_SOURCE_DIR}/configuration/parser/ParseAlgorithms.py)
set(DEFAULT_PROPERTIES_SRC ${PROJECT_SOURCE_DIR}/configuration/src/default_properties.cpp)

include_guard(GLOBAL)

file(MAKE_DIRECTORY ${CODE_GENERATION_DIR})
file(MAKE_DIRECTORY ${ALLEN_PARSER_DIR})
file(MAKE_DIRECTORY ${ALLEN_GENERATED_INCLUDE_FILES_DIR})
file(MAKE_DIRECTORY ${ALLEN_ALGORITHMS_DIR})

# We will invoke the parser a few times, set its required environment in a variable
# Add the scripts folder only if we are invoking with a CMAKE_TOOLCHAIN_FILE
if(CMAKE_TOOLCHAIN_FILE)
  set(PARSER_ENV PYTHONPATH=$ENV{PYTHONPATH}:${PROJECT_SOURCE_DIR}/scripts LD_LIBRARY_PATH=${LIBCLANG_LIBDIR}:$ENV{LD_LIBRARY_PATH})
else()
  set(PARSER_ENV PYTHONPATH=$ENV{PYTHONPATH}:${LIBCLANG_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages LD_LIBRARY_PATH=${LIBCLANG_LIBDIR}:$ENV{LD_LIBRARY_PATH})
endif()

# Parse Allen algorithms
# Parsing should depend on ALL algorithm headers (which include the Parameters section)
# We need to get the list of algorithms at configuration time in order to
# know the list of files that will be required of this build
set(ALGORITHM_HEADERS_LIST ${CODE_GENERATION_DIR}/algorithm_headers_list.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate algorithm_headers_list --filename "${ALGORITHM_HEADERS_LIST}" --prefix_project_folder "${PROJECT_SOURCE_DIR}")
file(READ "${ALGORITHM_HEADERS_LIST}" ALGORITHM_HEADERS_FILES) # ALGORITHM_HEADERS_FILES="a.cuh b.cuh c.cuh"

add_custom_command(
  OUTPUT "${PARSED_ALGORITHMS_OUTPUTFILE}"
  COMMENT "Parsing Allen algorithms"
  COMMAND
    ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate parsed_algorithms --filename "${PARSED_ALGORITHMS_OUTPUTFILE}" --prefix_project_folder "${PROJECT_SOURCE_DIR}"
  DEPENDS "${PROJECT_SOURCE_DIR}/configuration/parser/ParseAlgorithms.py" ${ALGORITHM_HEADERS_FILES})
add_custom_target(parsed_algorithms DEPENDS "${PARSED_ALGORITHMS_OUTPUTFILE}")

# Symlink Allen build directories
file(RELATIVE_PATH PROJECT_SOURCE_DIR_RELPATH ${PROJECT_SEQUENCE_DIR} ${PROJECT_SOURCE_DIR})
message(STATUS "Set project source dir to: ${PROJECT_SOURCE_DIR_RELPATH}")
add_custom_command(
  OUTPUT "${SEQUENCE_DEFINITION_DIR}" "${ALLEN_CORE_DIR}"
  COMMENT "Making symlink of sequence definitions and configuration utilities"
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink "${PROJECT_SOURCE_DIR_RELPATH}/configuration/python/AllenConf" "${SEQUENCE_DEFINITION_DIR}" &&
    ${CMAKE_COMMAND} -E create_symlink "${PROJECT_SOURCE_DIR_RELPATH}/configuration/python/AllenCore" "${ALLEN_CORE_DIR}"
  DEPENDS "${PROJECT_SOURCE_DIR}/configuration/python/AllenConf" "${PROJECT_SOURCE_DIR}/configuration/python/AllenCore")
add_custom_target(generate_conf_core DEPENDS "${SEQUENCE_DEFINITION_DIR}" "${ALLEN_CORE_DIR}")

# Generate Allen AlgorithmDB
add_custom_command(
  OUTPUT "${CODE_GENERATION_DIR}/AlgorithmDB.cpp"
  COMMENT "Generating AlgorithmDB"
  COMMAND ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate db --filename "${CODE_GENERATION_DIR}/AlgorithmDB.cpp" --parsed_algorithms "${PARSED_ALGORITHMS_OUTPUTFILE}"
  WORKING_DIRECTORY ${ALLEN_PARSER_DIR}
  DEPENDS "${PARSED_ALGORITHMS_OUTPUTFILE}")
add_custom_target(algorithm_db_generation DEPENDS "${CODE_GENERATION_DIR}/AlgorithmDB.cpp")
add_library(algorithm_db OBJECT "${CODE_GENERATION_DIR}/AlgorithmDB.cpp")
add_dependencies(algorithm_db algorithm_db_generation)
target_link_libraries(algorithm_db
  PUBLIC
    EventModel
    HostEventModel
    Backend
    AllenCommon
    Gear)

add_executable(default_properties ${DEFAULT_PROPERTIES_SRC})
target_link_libraries(default_properties PRIVATE AllenLib HostEventModel EventModel)

# Generate allen standalone algorithms file
add_custom_command(
  OUTPUT "${ALGORITHMS_OUTPUTFILE}"
  COMMAND
    ${CMAKE_COMMAND} -E env ${PARSER_ENV} ${Python_EXECUTABLE} ${ALGORITHMS_GENERATION_SCRIPT} --generate views --filename "${ALGORITHMS_OUTPUTFILE}" --parsed_algorithms "${PARSED_ALGORITHMS_OUTPUTFILE}" --default_properties $<TARGET_FILE:default_properties> &&
    ${CMAKE_COMMAND} -E touch ${ALLEN_ALGORITHMS_DIR}/__init__.py
  WORKING_DIRECTORY ${ALLEN_PARSER_DIR}
  DEPENDS "${PARSED_ALGORITHMS_OUTPUTFILE}" "${SEQUENCE_DEFINITION_DIR}" "${ALLEN_CORE_DIR}" default_properties)
add_custom_target(generate_algorithms_view DEPENDS "${ALGORITHMS_OUTPUTFILE}")
install(FILES "${ALGORITHMS_OUTPUTFILE}" DESTINATION python/AllenAlgorithms)

# Target that the generation of the sequences can depend on
add_custom_target(Sequences DEPENDS generate_algorithms_view)

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
  set(LHCBROOT $ENV{LHCBROOT} CACHE STRING "LHCB root directory")
  if (LHCBROOT)
    add_custom_command(
      OUTPUT "${PROJECT_SEQUENCE_DIR}/PyConf"
      COMMENT "Selecting user-specified LHCBROOT"
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${LHCBROOT}/PyConf/python/PyConf ${PROJECT_SEQUENCE_DIR}/PyConf)
    add_custom_target(checkout_lhcb DEPENDS "${PROJECT_SEQUENCE_DIR}/PyConf")
    message(STATUS "LHCBROOT set to ${LHCBROOT}")
  else()
    find_package(Git REQUIRED)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/external/LHCb")
    set(LHCBROOT "${PROJECT_BINARY_DIR}/external/LHCb")
    file(RELATIVE_PATH LHCBROOT_RELPATH ${PROJECT_SEQUENCE_DIR} ${LHCBROOT}) 

    add_custom_command(
      OUTPUT "${PROJECT_SEQUENCE_DIR}/PyConf"
      COMMENT "Checking out LHCb project from the LHCb stack"
      COMMAND
        ${CMAKE_COMMAND} -E env ${GIT_EXECUTABLE} clone https://gitlab.cern.ch/lhcb/LHCb.git ${PROJECT_BINARY_DIR}/external/LHCb &&
        ${CMAKE_COMMAND} -E create_symlink ${LHCBROOT_RELPATH}/PyConf/python/PyConf ${PROJECT_SEQUENCE_DIR}/PyConf)
    add_custom_target(checkout_lhcb DEPENDS "${PROJECT_SEQUENCE_DIR}/PyConf")
    message(STATUS "LHCBROOT set to ${LHCBROOT}")
  endif()

  set(GAUDIROOT $ENV{GAUDIROOT} CACHE STRING "GAUDI root directory")
  if (GAUDIROOT)
    add_custom_command(
      OUTPUT "${PROJECT_SEQUENCE_DIR}/GaudiKernel"
      COMMENT "Selecting user-specified GAUDIROOT"
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${GAUDIROOT}/GaudiKernel/python/GaudiKernel ${PROJECT_SEQUENCE_DIR}/GaudiKernel)
    add_custom_target(checkout_gaudi DEPENDS "${PROJECT_SEQUENCE_DIR}/GaudiKernel")
    message(STATUS "GAUDIROOT set to ${GAUDIROOT}")
  else()
    find_package(Git REQUIRED)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/external/Gaudi")
    set(GAUDIROOT "${PROJECT_BINARY_DIR}/external/Gaudi")
    file(RELATIVE_PATH GAUDIROOT_RELPATH ${PROJECT_SEQUENCE_DIR} ${GAUDIROOT})
    add_custom_command(
      OUTPUT "${PROJECT_SEQUENCE_DIR}/GaudiKernel"
      COMMENT "Checking out Gaudi project from the LHCb stack"
      COMMAND
        ${CMAKE_COMMAND} -E env ${GIT_EXECUTABLE} clone https://gitlab.cern.ch/gaudi/Gaudi.git ${PROJECT_BINARY_DIR}/external/Gaudi &&
        ${CMAKE_COMMAND} -E create_symlink ${GAUDIROOT_RELPATH}/GaudiKernel/python/GaudiKernel ${PROJECT_SEQUENCE_DIR}/GaudiKernel)
    add_custom_target(checkout_gaudi DEPENDS "${PROJECT_SEQUENCE_DIR}/GaudiKernel")
    message(STATUS "GAUDIROOT set to ${GAUDIROOT_RELPATH}")
  endif()
endif()

file(GLOB python_allen_conf "${PROJECT_SOURCE_DIR}/configuration/python/AllenConf/*py")
file(GLOB python_allen_core "${PROJECT_SOURCE_DIR}/configuration/python/AllenCore/*py")
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
      DEPENDS "${PROJECT_SOURCE_DIR}/configuration/python/AllenSequences/${sequence}.py" "${ALGORITHMS_OUTPUTFILE}" ${python_allen_conf} ${python_allen_core}
      WORKING_DIRECTORY ${sequence_dir})
  else()
    add_custom_command(
      OUTPUT "${PROJECT_BINARY_DIR}/${sequence}.json"
      COMMAND
        ${CMAKE_COMMAND} -E env "${LIBRARY_PATH_VARNAME}=$ENV{LD_LIBRARY_PATH}" "PYTHONPATH=${PROJECT_SEQUENCE_DIR}:$ENV{PYTHONPATH}" "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/configuration/python/AllenCore/gen_allen_json.py" "--seqpath" "${PROJECT_SOURCE_DIR}/configuration/python/AllenSequences/${sequence}.py" "--no-register-keys" &&
        ${CMAKE_COMMAND} -E rename "${sequence_dir}/Sequence.json" "${PROJECT_BINARY_DIR}/${sequence}.json"
      DEPENDS "${PROJECT_SOURCE_DIR}/configuration/python/AllenSequences/${sequence}.py" "${ALGORITHMS_OUTPUTFILE}" "${PROJECT_SEQUENCE_DIR}/GaudiKernel" "${PROJECT_SEQUENCE_DIR}/PyConf" ${python_allen_conf} ${python_allen_core}
      WORKING_DIRECTORY ${sequence_dir})
  endif()
  add_custom_target(sequence_${sequence} DEPENDS "${PROJECT_BINARY_DIR}/${sequence}.json")
  add_dependencies(Sequences sequence_${sequence})
  install(FILES "${PROJECT_BINARY_DIR}/${sequence}.json" DESTINATION constants)
endfunction()
