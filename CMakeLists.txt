cmake_minimum_required(VERSION 3.10.0)

project(MnnBatchDemo)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR /home/ubuntu/testDemo/libs/linux/opencv4.5)
set(OpenCV_INCLUDE_DIRS /home/ubuntu/testDemo/libs/linux/opencv4.5/include/opencv4)
set(OpenCV_LIBRARIES /home/ubuntu/testDemo/libs/linux/opencv4.5/lib)
list(APPEND CMAKE_PREFIX_PATH /home/ubuntu/testDemo/libs/linux/opencv4.5)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_LIBRARIES})
find_package(OpenCV REQUIRED)

message("opencv libs : ", ${OpenCV_LIBS})

set(mnn_dir /home/ubuntu/testDemo/libs/linux/MNN)
include_directories( ${mnn_dir}/include)
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${mnn_dir}/lib/libMNN.so)

add_executable(MNN_Batch_demo MNN_Batch_demo.cpp )

target_link_libraries(MNN_Batch_demo MNN ${OpenCV_LIBS} )
