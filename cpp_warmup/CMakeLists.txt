cmake_minimum_required(VERSION 3.16)
project(transforms_cpp)

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)

add_executable(transforms src/transforms.cpp)
target_link_libraries(transforms Eigen3::Eigen)
