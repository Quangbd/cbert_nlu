cmake_minimum_required(VERSION 3.17)
project(cbert_nlu)

list(APPEND CMAKE_PREFIX_PATH "./libtorch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILE main.cpp tokenizer.h tokenizer.cpp utf8proc.h)

add_executable(cbert_nlu ${SOURCE_FILE})
#add_library(cbert_nlu SHARED ${SOURCE_FILE})

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
endif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#target_include_directories(cbert_nlu PRIVATE includes/tensorflow-lite)
target_link_directories(cbert_nlu PUBLIC libs/osx_x86_64)
target_link_libraries(cbert_nlu utf8proc pthread m dl ${TORCH_LIBRARIES})