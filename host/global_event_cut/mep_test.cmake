###############################################################################
# (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/backend/include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/setup/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/sequence/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/device/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/device/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/device/global_event_cut/include)
include_directories(${CMAKE_SOURCE_DIR}/device/utils/prefix_sum/include)
include_directories(${CMAKE_SOURCE_DIR}/device/utils/float_operations/include)
include_directories(${CMAKE_SOURCE_DIR}/device/utils/sorting/include)
include_directories(${CMAKE_SOURCE_DIR}/device/utils/binary_search/include)
include_directories(${CMAKE_SOURCE_DIR}/device/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/device/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/device/event_model/UT/include)
include_directories(${CMAKE_SOURCE_DIR}/device/event_model/SciFi/include)
include_directories(${CMAKE_SOURCE_DIR}/device/event_model/associate/include)
include_directories(${CMAKE_SOURCE_DIR}/device/global_event_cut/include)
include_directories(${CMAKE_SOURCE_DIR}/device/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/device/UT/UTDecoding/include)
include_directories(${CMAKE_SOURCE_DIR}/device/UT/sorting/include)
include_directories(${CMAKE_SOURCE_DIR}/device/UT/consolidate/include)
include_directories(${CMAKE_SOURCE_DIR}/device/UT/compassUT/include)
include_directories(${CMAKE_SOURCE_DIR}/device/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/device/velo/calculate_phi_and_sort/include)
include_directories(${CMAKE_SOURCE_DIR}/device/velo/consolidate_tracks/include)
include_directories(${CMAKE_SOURCE_DIR}/device/velo/mask_clustering/include)
include_directories(${CMAKE_SOURCE_DIR}/device/velo/prefix_sum/include)
include_directories(${CMAKE_SOURCE_DIR}/device/velo/search_by_triplet/include)
include_directories(${CMAKE_SOURCE_DIR}/device/velo/simplified_kalman_filter/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/preprocessing/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/common/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/search_initial_windows/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/find_compatible_windows/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/triplet_seeding/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/triplet_keep_best/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/extend_tracks_x/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/composite_algorithms/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/extend_tracks_first_layers_x/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/extend_tracks_uv/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/quality_filters/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/fit/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/looking_forward/calculate_parametrization/include)
include_directories(${CMAKE_SOURCE_DIR}/device/SciFi/consolidate/include)
include_directories(${CMAKE_SOURCE_DIR}/device/associate/include)
include_directories(${CMAKE_SOURCE_DIR}/device/muon/common/include)
include_directories(${CMAKE_SOURCE_DIR}/device/muon/preprocessing/include)
include_directories(${CMAKE_SOURCE_DIR}/device/muon/is_muon/include)
include_directories(${CMAKE_SOURCE_DIR}/device/muon/classification/include)
include_directories(${CMAKE_SOURCE_DIR}/device/muon/decoding/include)
include_directories(${CMAKE_SOURCE_DIR}/device/muon/decoding_steps/include)
include_directories(${CMAKE_SOURCE_DIR}/device/utils/include)
include_directories(${CMAKE_SOURCE_DIR}/device/PV/patPV/include)
include_directories(${CMAKE_SOURCE_DIR}/device/PV/common/include)
include_directories(${CMAKE_SOURCE_DIR}/device/raw_banks/include)
include_directories(${CMAKE_SOURCE_DIR}/host/velo/clustering/include)
include_directories(${CMAKE_SOURCE_DIR}/host/utils/prefix_sum/include)
include_directories(${CMAKE_SOURCE_DIR}/host/global_event_cut/include)
include_directories(${CMAKE_SOURCE_DIR}/host/prefix_sum/include)
include_directories(${CMAKE_SOURCE_DIR}/checker/tracking/include)
include_directories(${CMAKE_SOURCE_DIR}/checker/pv/include)
include_directories(${CMAKE_SOURCE_DIR}/checker/selections/include)
include_directories(${CMAKE_SOURCE_DIR}/device/PV/beamlinePV/include)
include_directories(${CMAKE_SOURCE_DIR}/device/kalman/ParKalman/include)
include_directories(${CMAKE_SOURCE_DIR}/device/vertex_fit/common/include)
include_directories(${CMAKE_SOURCE_DIR}/device/vertex_fit/vertex_fitter/include)
include_directories(${CMAKE_SOURCE_DIR}/device/selections/Hlt1/include)
include_directories(${CMAKE_SOURCE_DIR}/external)
include_directories(${PROJECT_BINARY_DIR}/configuration/sequences)
include_directories(${CPPGSL_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

add_executable(mep_gec test/mep_gec.cpp ${CMAKE_SOURCE_DIR}/backend/src/CPUBackend.cpp
  ${CMAKE_SOURCE_DIR}/host/global_event_cut/src/HostGlobalEventCut.cpp)

target_include_directories(mep_gec PUBLIC
  ${CPPGSL_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/stream/gear/include
  ${CMAKE_SOURCE_DIR}/host/global_event_cut/include
  ${CMAKE_SOURCE_DIR}/main/include
  ${Boost_INCLUDE_DIRS})

if (STANDALONE)
  target_link_libraries(mep_gec PUBLIC Common mdf ${MPI_CXX_LIBRARIES})
else()
  find_package(fmt REQUIRED)
  target_link_libraries(mep_gec PUBLIC Common mdf ${MPI_CXX_LIBRARIES} fmt::fmt)
endif()

target_compile_definitions(mep_gec PUBLIC TARGET_DEVICE_CPU)

install(TARGETS mep_gec RUNTIME DESTINATION bin OPTIONAL)
