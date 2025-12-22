cmake_minimum_required(VERSION 3.8)
project(lidar_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

cmake_policy(SET CMP0074 NEW)

set(EXEC_TRAJ trajectory_node)
set(EXEC_PLANNER planner_node)
set(EXEC_PERCEPTION lidar_perception_node)
set(EXEC_TEST test_flight)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(PCL 1.12 REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(sip_interface REQUIRED)

add_library(project_includes INTERFACE)

target_include_directories(
  project_includes INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(
  ${EXEC_PERCEPTION}
  src/lidar_perception_node.cpp
  src/surfel_extractor.cpp
  src/surfel_visualizer.cpp
)
add_executable(
  ${EXEC_PLANNER}
  src/planner_node.cpp
)
add_executable(
  ${EXEC_TRAJ}
  src/trajectory_node.cpp
)

add_executable(
  ${EXEC_TEST}
  src/test_flight_node.cpp
)

set(
  DEPS_PERCEPTION
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2_ros
  tf2_sensor_msgs
  tf2
  pcl_conversions
)

set(
  DEPS_PLANNER
  rclcpp
)

set(
  DEPS_TRAJ
  rclcpp
  rclcpp_action
  nav_msgs
  sip_interface
)

set(
  DEPS_TEST
  rclcpp
  rclcpp_action
  nav_msgs
  sip_interface
)

target_include_directories(
  ${EXEC_PERCEPTION} PRIVATE
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(
  ${EXEC_PERCEPTION}
  project_includes
  ${PCL_LIBRARIES}
)

target_link_libraries(
  ${EXEC_PLANNER}
  project_includes
)

target_link_libraries(
  ${EXEC_TRAJ}
  project_includes
  Eigen3::Eigen
)

target_link_libraries(
  ${EXEC_TEST}
  project_includes
  Eigen3::Eigen
)

ament_target_dependencies(
  ${EXEC_PERCEPTION}
  ${DEPS_PERCEPTION}
)

ament_target_dependencies(
  ${EXEC_PLANNER}
  ${DEPS_PLANNER}
)

ament_target_dependencies(
  ${EXEC_TRAJ}
  ${DEPS_TRAJ}
)

ament_target_dependencies(
  ${EXEC_TEST}
  ${DEPS_TEST}
)

install(
  TARGETS ${EXEC_PERCEPTION}
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS ${EXEC_PLANNER}
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS ${EXEC_TRAJ}
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS ${EXEC_TEST}
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include launch 
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
