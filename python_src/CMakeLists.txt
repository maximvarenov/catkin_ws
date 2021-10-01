
cmake_minimum_required(VERSION 2.8.3)
project(python_src)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
   sensor_msgs
  std_msgs  
cv_bridge
image_transport 
  rtt_roscomm 
  rtt_std_msgs
 rtt_geometry_msgs

  message_filters
)


find_package(OpenCV 3 REQUIRED)

find_package(PCL  1.2  REQUIRED  )

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
)

catkin_package(
  )

link_directories(${CATKIN_LIBRARY_DIRS}  ${OpenCV_LIBRARY_DIRS} ${PCL_LIBRARY_DIRS}
)
add_definitions(${OpenCV_DEFINITIONS} ${PCL_DEFINITIONS})  
  

