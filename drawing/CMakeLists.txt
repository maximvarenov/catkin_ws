cmake_minimum_required(VERSION 2.8.3)
project(drawing)



## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
pcl_ros
)


find_package(PCL 1.8 REQUIRED )


catkin_package()

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
include
${catkin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(drawing src/drawing.cpp)
target_link_libraries(drawing  ${catkin_LIBRARIES} ${PCL_LIBRARIES}   )


add_executable(test_pcl src/test_pcl.cpp)
target_link_libraries (test_pcl   ${PCL_LIBRARIES})
 

