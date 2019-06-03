cmake_minimum_required(VERSION 3.5)
project(denoising_projection)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "-Ofast -std=gnu++11 -fopenmp -fexceptions")

include_directories(
  include
)

add_executable(denoising_projection 
                src/main.cpp
                src/ParameterManager.cpp
              )
              
find_package(PkgConfig REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
target_link_libraries(denoising_projection ${PCL_LIBRARIES})
include_directories(${VTK_INCLUDE_DIRS})
target_link_libraries(denoising_projection ${VTK_LIBRARIES})

