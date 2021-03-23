cmake_minimum_required(VERSION 3.17)
project(cbert_nlu)

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILE main.cpp tokenizer.h tokenizer.cpp utf8proc.h)

add_executable(cbert_nlu ${SOURCE_FILE})
#add_library(cbert_nlu SHARED ${SOURCE_FILE})

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
endif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

#target_include_directories(cbert_nlu PRIVATE includes/tensorflow-lite)
target_link_directories(cbert_nlu PUBLIC libs/osx_x86_64)
target_link_libraries(cbert_nlu utf8proc pthread m dl)