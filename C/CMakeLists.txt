cmake_minimum_required(VERSION 3.22)
project(Treeeeee C CXX)
# Set CXX standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -fsanitize=address -g")

# Add directories
include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRC_CXX src/*.cpp)

# Define output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Combine sources
set(SOURCES ${SRC_CXX})

# Add executable
add_executable(model_inference ${SOURCES})

# Enable OpenMP
option(OMP "Enable OpenMP support" OFF)
if(OMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(model_inference PUBLIC OpenMP::OpenMP_CXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOMP")
endif()

include_directories(/usr/local/include)

target_link_libraries(model_inference PUBLIC m)
target_link_libraries(model_inference PUBLIC yaml-cpp)

# Output directories
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/*
    COMMENT "Cleaning all build files"
)