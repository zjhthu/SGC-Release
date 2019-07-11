cmake_minimum_required(VERSION 3.2)
project(SUNCGData)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib)

find_package(Boost 1.58.0 EXACT REQUIRED COMPONENTS system python)
find_package(PythonLibs  2.7 EXACT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Numpy REQUIRED)
find_package(Glog REQUIRED)

message('${Boost_INCLUDE_DIRS}')

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
# include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

# create the lib
CUDA_ADD_LIBRARY(SUNCGData SHARED src/SUNCGData.cu)
set_target_properties(SUNCGData PROPERTIES PREFIX "")
message('${PYTHON_LIBRARIES}')
message('${Boost_LIBRARIES}')
# link
target_link_libraries(SUNCGData ${PYTHON_LIBRARIES} ${GLOG_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

