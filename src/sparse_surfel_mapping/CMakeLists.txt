cmake_minimum_required(VERSION 3.8)
project(sparse_surfel_mapping)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters)
find_package(Eigen3 REQUIRED)

find_package(sip_interface REQUIRED)


# include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# link directories
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# set dependencies
set(
  DEPS
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  pcl_conversions
  
  sip_interface
)

# main node executable
add_executable(
  surfel_mapping_node
  src/surfel_map_exec.cpp

  src/mapper/preprocess.cpp
  src/mapper/surfel.cpp
  src/mapper/voxel.cpp
  src/mapper/spatial_hash.cpp
  src/mapper/surfel_map.cpp

  src/planner/viewpoint.cpp
  src/planner/viewpoint_generator.cpp
  src/planner/coverage_tracker.cpp
  src/planner/inspection_planner.cpp
  src/planner/rrt.cpp

  src/ros_nodes/surfel_map_node.cpp
  src/ros_nodes/planner_node.cpp

)

ament_target_dependencies(
  surfel_mapping_node
  ${DEPS}
)

target_link_libraries(
  surfel_mapping_node
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)

add_executable(
  trajectory_tracker
  src/ros_nodes/trajectory_tracker.cpp
)
ament_target_dependencies(
  trajectory_tracker
  rclcpp
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
)

# install targets
install(
  TARGETS surfel_mapping_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS trajectory_tracker
  DESTINATION lib/${PROJECT_NAME}
)

# install headers (lib?)
install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# install launch file
install(
  DIRECTORY
  # launch
  # config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
