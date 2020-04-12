cmake_minimum_required (VERSION 2.6)
project (autodiff)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type: Release")
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(GTest REQUIRED)
 
include_directories(include
    ${GTEST_INCLUDE_DIRS}
)
 
add_executable(test_forward_scalar src/test_forward_scalar.cpp)
target_link_libraries(test_forward_scalar
    gtest
    gtest_main
    ${GTEST_LIBRARIES}
    pthread
)
