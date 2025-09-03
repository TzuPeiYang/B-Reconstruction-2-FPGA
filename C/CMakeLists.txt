cmake_minimum_required(VERSION 3.22)
project(Treeeeee C CXX)
# Set C standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wall -O2")

# Add directories
include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/lib)
file(GLOB SRC_CXX src/*.cpp lib/*.cpp)

# Define output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Combine sources
set(SOURCES ${SRC_CXX})

# Add executable
add_executable(network_output ${SOURCES})

# Enable OpenMP
option(OMP "Enable OpenMP support" OFF)
if(OMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(network_output PUBLIC OpenMP::OpenMP_CXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOMP")
endif()

target_link_libraries(network_output PUBLIC m)

# Output directories
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/*
    COMMENT "Cleaning all build files"
)