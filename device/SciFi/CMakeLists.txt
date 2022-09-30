###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
###############################################################################
file(GLOB scifi_preprocessing "preprocessing/src/*cu")
file(GLOB scifi_looking_forward "looking_forward/src/*cu")
file(GLOB scifi_consolidate "consolidate/src/*cu")
file(GLOB scifi_hybridseeding "hybridseeding/src/*cu")
file(GLOB scifi_hybridseeding_consolidate "hybridseeding/consolidate/src/*cu")
file(GLOB scifi_hybridseeding_consolidate_cpp "hybridseeding/consolidate/src/*cpp")
file(GLOB scifi_consolidate_cpp "consolidate/src/*cpp")

allen_add_device_library(SciFi STATIC
  ${scifi_preprocessing}
  ${scifi_looking_forward}
  ${scifi_consolidate}
  ${scifi_hybridseeding}
  ${scifi_hybridseeding_consolidate}
)
allen_add_host_library(SciFiHost STATIC
  ${scifi_consolidate_cpp}
  ${scifi_hybridseeding_consolidate_cpp}
)

target_link_libraries(SciFiHost PRIVATE Backend HostEventModel EventModel Utils)
target_link_libraries(SciFi PRIVATE Backend HostEventModel EventModel Utils SciFiHost)

target_include_directories(SciFiHost PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hybridseeding/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hybridseeding/consolidate/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/looking_forward/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/consolidate/include>)

target_include_directories(SciFi PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hybridseeding/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hybridseeding/consolidate/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/looking_forward/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/consolidate/include>)
