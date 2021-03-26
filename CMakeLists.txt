cmake_minimum_required(VERSION 3.17)
project(cbert_slu)

list(APPEND CMAKE_PREFIX_PATH "./libtorch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILE main.cpp)

add_executable(cbert_slu ${SOURCE_FILE})
#add_library(cbert_slu SHARED ${SOURCE_FILE})

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
endif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#target_include_directories(cbert_nlu PRIVATE includes/tensorflow-lite)
target_link_directories(cbert_slu PUBLIC libs/osx_x86_64)
target_link_libraries(cbert_slu ${TORCH_LIBRARIES})