###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
###############################################################################
file(GLOB UT_decoding "UTDecoding/src/*cu")
file(GLOB UT_consolidate "consolidate/src/*cu")
file(GLOB CompassUT_tracking "compassUT/src/*cu")

add_library(UTCommon INTERFACE)
target_include_directories(UTCommon INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>)
install(TARGETS UTCommon EXPORT Allen)

allen_add_device_library(UT STATIC
  ${UT_decoding}
  ${UT_consolidate}
  ${CompassUT_tracking}
)

target_link_libraries(UT
  PUBLIC
    UTCommon
  PRIVATE
    Backend
    HostEventModel
    EventModel
    Utils)

target_include_directories(UT PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/UTDecoding/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/consolidate/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/compassUT/include>)

target_link_libraries(WrapperInterface INTERFACE UTCommon)
