cmake_minimum_required(VERSION 3.8)
project(close_approach)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =========================
# Find packages
# =========================
find_package(ament_cmake REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS
  common
  filters
  io
  segmentation
  search
  kdtree
  surface
  features
  registration
)

find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(inha_interfaces REQUIRED)

# =========================
# Sources
# =========================
set(APPROACH_CPP_SOURCES
  src/main.cpp
  src/roi_filter.cpp
  src/convexhull.cpp
  src/plane_filter.cpp
  src/edge_extractor.cpp
  src/error_estimator.cpp
  src/pid_controller.cpp
)

set(APPROACH_HEAD_CONTROL_CPP_SOURCES
  src/approach_head_control.cpp
)

set(ALIGN_PERSON_CPP_SOURCES
  src/align_to_person.cpp
)

set(DEAD_RECKONING_APPROACH_CPP_SOURCES
  src/dead_reckoning_approach.cpp
)

# =========================
# Executables
# =========================
add_executable(approach_node
  ${APPROACH_CPP_SOURCES}
)

add_executable(approach_head_control_node
  ${APPROACH_HEAD_CONTROL_CPP_SOURCES}
)

# add_executable(accumulation_and_icp_node
#   src/accumulation_and_icp.cpp
# )

add_executable(align_to_person_node
  ${ALIGN_PERSON_CPP_SOURCES}
)

add_executable(dead_reckoning_approach_node
  ${DEAD_RECKONING_APPROACH_CPP_SOURCES}
)

add_executable(retreat_node
  src/retreat.cpp
)

# =========================
# Include directories
# =========================
target_include_directories(approach_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(approach_head_control_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# target_include_directories(accumulation_and_icp_node
#   PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:include>
#     ${PCL_INCLUDE_DIRS}
# )

target_include_directories(align_to_person_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)

target_include_directories(dead_reckoning_approach_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(retreat_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# =========================
# C++ standard
# =========================
target_compile_features(approach_node PUBLIC cxx_std_17)
target_compile_features(approach_head_control_node PUBLIC cxx_std_17)
# target_compile_features(accumulation_and_icp_node PUBLIC cxx_std_17)
target_compile_features(align_to_person_node PUBLIC cxx_std_17)
target_compile_features(dead_reckoning_approach_node PUBLIC cxx_std_17)
target_compile_features(retreat_node PUBLIC cxx_std_17)

# =========================
# ament dependencies
# =========================
ament_target_dependencies(approach_node
  rclcpp
  rclcpp_action
  geometry_msgs
  sensor_msgs
  nav_msgs
  vision_msgs
  visualization_msgs
  message_filters
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
  inha_interfaces
)

ament_target_dependencies(approach_head_control_node
  rclcpp
  rclcpp_action
  geometry_msgs
  sensor_msgs
  vision_msgs
  visualization_msgs
  std_srvs
  message_filters
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
  inha_interfaces
)

# ament_target_dependencies(accumulation_and_icp_node
#   rclcpp
#   rclcpp_action
#   geometry_msgs
#   sensor_msgs
#   nav_msgs
#   vision_msgs
#   visualization_msgs
#   message_filters
#   pcl_conversions
#   tf2
#   tf2_ros
#   tf2_geometry_msgs
#   inha_interfaces
# )

ament_target_dependencies(align_to_person_node
  rclcpp
  rclcpp_action
  geometry_msgs
  sensor_msgs
  nav_msgs
  vision_msgs
  visualization_msgs
  message_filters
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
  inha_interfaces
)

ament_target_dependencies(dead_reckoning_approach_node
  rclcpp
  rclcpp_action
  geometry_msgs
  tf2
  tf2_ros
  inha_interfaces
)

ament_target_dependencies(retreat_node
  rclcpp
  rclcpp_action
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  inha_interfaces
)

# =========================
# External libraries
# =========================
target_link_libraries(approach_node
  ${PCL_LIBRARIES}
  ${PCL_IO_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(approach_head_control_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

# target_link_libraries(accumulation_and_icp_node
#   ${PCL_LIBRARIES}
#   ${PCL_IO_LIBRARIES}
# )

target_link_libraries(align_to_person_node
  ${PCL_LIBRARIES}
  ${PCL_IO_LIBRARIES}
)

# tf2_geometry_msgs 직접 링크 보강
# ROS2 배포판/패키지 구성에 따라 imported target이 있거나 없을 수 있어서 둘 다 대응
if(TARGET tf2_geometry_msgs::tf2_geometry_msgs)
  target_link_libraries(approach_node
    tf2_geometry_msgs::tf2_geometry_msgs
  )

  target_link_libraries(retreat_node
    tf2_geometry_msgs::tf2_geometry_msgs
  )

  target_link_libraries(approach_head_control_node
    tf2_geometry_msgs::tf2_geometry_msgs
  )

  # target_link_libraries(accumulation_and_icp_node
  #   tf2_geometry_msgs::tf2_geometry_msgs
  # )

  target_link_libraries(align_to_person_node
    tf2_geometry_msgs::tf2_geometry_msgs
  )

  target_link_libraries(dead_reckoning_approach_node
    tf2_geometry_msgs::tf2_geometry_msgs
  )
else()
  target_link_libraries(approach_node
    ${tf2_geometry_msgs_LIBRARIES}
  )

  target_link_libraries(retreat_node
    ${tf2_geometry_msgs_LIBRARIES}
  )

  target_link_libraries(approach_head_control_node
    ${tf2_geometry_msgs_LIBRARIES}
  )

  # target_link_libraries(accumulation_and_icp_node
  #   ${tf2_geometry_msgs_LIBRARIES}
  # )

  target_link_libraries(align_to_person_node
    ${tf2_geometry_msgs_LIBRARIES}
  )

  target_link_libraries(dead_reckoning_approach_node
    ${tf2_geometry_msgs_LIBRARIES}
  )
endif()

# =========================
# Compile definitions
# =========================
target_compile_definitions(approach_node
  PRIVATE
    ${PCL_DEFINITIONS}
)

target_compile_definitions(approach_head_control_node
  PRIVATE
    ${PCL_DEFINITIONS}
)

# target_compile_definitions(accumulation_and_icp_node
#   PRIVATE
#     ${PCL_DEFINITIONS}
# )

target_compile_definitions(align_to_person_node
  PRIVATE
    ${PCL_DEFINITIONS}
)

# =========================
# Install
# =========================
install(
  TARGETS
    approach_node
    approach_head_control_node
    # accumulation_and_icp_node
    align_to_person_node
    dead_reckoning_approach_node
    retreat_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  PROGRAMS
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/detector.py
  DESTINATION lib/${PROJECT_NAME}
)

# =========================
# Export
# =========================
ament_export_include_directories(include)

ament_export_dependencies(
  geometry_msgs
  sensor_msgs
  nav_msgs
  vision_msgs
  visualization_msgs
  std_srvs
  message_filters
  pcl_conversions
  rclcpp
  rclcpp_action
  rosidl_default_runtime
  tf2
  tf2_ros
  tf2_geometry_msgs
  inha_interfaces
)

# =========================
# Testing
# =========================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
