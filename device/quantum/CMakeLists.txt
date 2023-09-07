###############################################################################
# (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      #
###############################################################################
file(GLOB quantum_sources "src/*cu")

include_directories(include)
include_directories(${PROJECT_SOURCE_DIR}/device/velo/common/include)
include_directories(${PROJECT_SOURCE_DIR}/device/event_model/common/include)
include_directories(${PROJECT_SOURCE_DIR}/device/event_model/velo/include)
include_directories(${PROJECT_SOURCE_DIR}/main/include)
include_directories(${PROJECT_SOURCE_DIR}/stream/gear/include)
include_directories(${PROJECT_SOURCE_DIR}/stream/sequence/include)

allen_add_device_library(Quantum SHARED
        ${quantum_sources}
        )

find_package(Python COMPONENTS Interpreter Development)

if ("${TARGET_DEVICE}" STREQUAL "CUDA")
    target_link_libraries(Quantum PRIVATE -lcustatevec)
endif()
target_link_libraries(Quantum PRIVATE ${Python_LIBRARIES})
target_include_directories(Quantum PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${Python_INCLUDE_DIRS})
target_link_libraries(Quantum PRIVATE Backend HostEventModel EventModel Utils)
