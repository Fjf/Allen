###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
###############################################################################
file(GLOB matching_match "match/src/*cu")
file(GLOB matching_consolidate "consolidate/src/*cu")
file(GLOB matching_consolidate_cpp "consolidate/src/*cpp")

allen_add_device_library(track_matching STATIC
  ${matching_match}
  ${matching_consolidate}
)

allen_add_host_library(track_matching_host STATIC
  ${matching_consolidate_cpp}
)

target_link_libraries(track_matching_host PRIVATE Backend HostEventModel EventModel Utils)
target_link_libraries(track_matching PRIVATE Backend HostEventModel EventModel Utils track_matching_host)

target_include_directories(track_matching_host PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/match/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/consolidate/include>)

target_include_directories(track_matching PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/match/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/consolidate/include>)

target_include_directories(WrapperInterface INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/match/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>)
