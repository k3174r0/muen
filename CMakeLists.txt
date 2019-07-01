cmake_minimum_required(VERSION 3.10)

project(muen)
add_executable(${PROJECT_NAME} src/main.cpp)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

#Darknet Settings
#Reference: https://bitbucket.org/ChernyshovYuriy/darknetapp/src/master/
set(DARKNET_PATH /path/to/darknet)  #set yr darknet path.
set(DARKNET_LIB_NAME libdarknet)
include_directories(${DARKNET_PATH}/include)
add_library(${DARKNET_LIB_NAME} SHARED IMPORTED)
set_property(TARGET ${DARKNET_LIB_NAME} PROPERTY IMPORTED_LOCATION "${DARKNET_PATH}/libdarknet.so")


find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
#    -gencode arch=compute_61,code=sm_61
#    -gencode arch=compute_62,code=sm_62
)
add_definitions(-DGPU)
add_definitions(-DCUDNN)

find_package(Boost REQUIRED COMPONENTS thread)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(-DOPENCV)
add_definitions(-O4 -g)

link_directories(
    /usr/local/cuda/lib64
)

target_link_libraries(${PROJECT_NAME}
    m
    pthread
    stdc++
    cuda
    cudart
    cublas
    curand
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${OpenCV_LIBS}
    ${DARKNET_LIB_NAME}
)