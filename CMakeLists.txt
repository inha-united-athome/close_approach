cmake_minimum_required(VERSION 3.8)
project(close_approach)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

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
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)

find_package(inha_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CloseApproachError.msg"
  DEPENDENCIES
    std_msgs
    geometry_msgs
)

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

add_executable(approach_node
  ${APPROACH_CPP_SOURCES}
)

add_executable(approach_head_control_node
  ${APPROACH_HEAD_CONTROL_CPP_SOURCES}
)

add_executable(align_to_person_node
  ${ALIGN_PERSON_CPP_SOURCES}
)

add_executable(dead_reckoning_approach_node
  ${DEAD_RECKONING_APPROACH_CPP_SOURCES}
)

add_executable(retreat_node
  src/retreat.cpp
)

add_executable(submap_close_approach_perception_node
  src/submap_close_approach_perception_node.cpp
)

add_executable(submap_close_approach_controller_node
  src/submap_close_approach_controller_node.cpp
)

foreach(target
  approach_node
  approach_head_control_node
  align_to_person_node
  dead_reckoning_approach_node
  retreat_node
  submap_close_approach_perception_node
  submap_close_approach_controller_node
)
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${PCL_INCLUDE_DIRS}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_compile_features(${target} PUBLIC cxx_std_17)
endforeach()

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
  tf2_geometry_msgs
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

ament_target_dependencies(submap_close_approach_perception_node
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  pcl_conversions
  tf2
  tf2_ros
  tf2_sensor_msgs
)

ament_target_dependencies(submap_close_approach_controller_node
  rclcpp
  geometry_msgs
)

target_link_libraries(approach_node
  ${PCL_LIBRARIES}
  ${PCL_IO_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(approach_head_control_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(align_to_person_node
  ${PCL_LIBRARIES}
  ${PCL_IO_LIBRARIES}
)

target_link_libraries(submap_close_approach_perception_node
  ${PCL_LIBRARIES}
  ${PCL_IO_LIBRARIES}
)

if(TARGET tf2_geometry_msgs::tf2_geometry_msgs)
  target_link_libraries(approach_node tf2_geometry_msgs::tf2_geometry_msgs)
  target_link_libraries(retreat_node tf2_geometry_msgs::tf2_geometry_msgs)
  target_link_libraries(approach_head_control_node tf2_geometry_msgs::tf2_geometry_msgs)
  target_link_libraries(align_to_person_node tf2_geometry_msgs::tf2_geometry_msgs)
  target_link_libraries(dead_reckoning_approach_node tf2_geometry_msgs::tf2_geometry_msgs)
else()
  target_link_libraries(approach_node ${tf2_geometry_msgs_LIBRARIES})
  target_link_libraries(retreat_node ${tf2_geometry_msgs_LIBRARIES})
  target_link_libraries(approach_head_control_node ${tf2_geometry_msgs_LIBRARIES})
  target_link_libraries(align_to_person_node ${tf2_geometry_msgs_LIBRARIES})
  target_link_libraries(dead_reckoning_approach_node ${tf2_geometry_msgs_LIBRARIES})
endif()

rosidl_target_interfaces(submap_close_approach_perception_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
rosidl_target_interfaces(submap_close_approach_controller_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_compile_definitions(approach_node PRIVATE ${PCL_DEFINITIONS})
target_compile_definitions(approach_head_control_node PRIVATE ${PCL_DEFINITIONS})
target_compile_definitions(align_to_person_node PRIVATE ${PCL_DEFINITIONS})
target_compile_definitions(submap_close_approach_perception_node PRIVATE ${PCL_DEFINITIONS})

install(
  TARGETS
    approach_node
    approach_head_control_node
    align_to_person_node
    dead_reckoning_approach_node
    retreat_node
    submap_close_approach_perception_node
    submap_close_approach_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)

ament_export_dependencies(
  geometry_msgs
  sensor_msgs
  std_msgs
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
  tf2_sensor_msgs
  inha_interfaces
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
