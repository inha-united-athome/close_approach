cmake_minimum_required(VERSION 3.8)
project(close_approach)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io segmentation search kdtree surface features registration)
find_package(OpenCV REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(inha_interfaces REQUIRED)


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


add_executable(approach_node ${APPROACH_CPP_SOURCES})
add_executable(approach_head_control_node ${APPROACH_HEAD_CONTROL_CPP_SOURCES})
add_executable(accumulation_and_icp_node src/accumulation_and_icp.cpp)
add_executable(align_to_person_node ${ALIGN_PERSON_CPP_SOURCES})

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

target_include_directories(accumulation_and_icp_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)

target_include_directories(align_to_person_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)

target_compile_features(approach_node PUBLIC cxx_std_17)
target_compile_features(approach_head_control_node PUBLIC cxx_std_17)
target_compile_features(accumulation_and_icp_node PUBLIC cxx_std_17)
target_compile_features(align_to_person_node PUBLIC cxx_std_17)

target_link_libraries(approach_node
  ${PCL_LIBRARIES}
  ${PCL_IO_LIBRARIES}
  ${OpenCV_LIBS}
)


target_compile_definitions(approach_node
  PRIVATE
    ${PCL_DEFINITIONS}
)

target_link_libraries(approach_head_control_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

target_compile_definitions(approach_head_control_node
  PRIVATE
    ${PCL_DEFINITIONS}
)

target_link_libraries(accumulation_and_icp_node
  ${PCL_LIBRARIES}
  ${PCL_IO_LIBRARIES}
)

target_compile_definitions(accumulation_and_icp_node
  PRIVATE
    ${PCL_DEFINITIONS}
)

target_link_libraries(align_to_person_node
  ${PCL_LIBRARIES}
  ${PCL_IO_LIBRARIES}
)

target_compile_definitions(align_to_person_node
  PRIVATE
    ${PCL_DEFINITIONS}
)




ament_target_dependencies(approach_node
  geometry_msgs
  message_filters
  pcl_conversions
  rclcpp
  rclcpp_action
  sensor_msgs
  tf2
  tf2_ros
  vision_msgs
  visualization_msgs
  inha_interfaces
)

ament_target_dependencies(approach_head_control_node
  geometry_msgs
  message_filters
  pcl_conversions
  rclcpp
  rclcpp_action
  sensor_msgs
  tf2
  tf2_ros
  vision_msgs
  visualization_msgs
  inha_interfaces
  std_srvs
)


ament_target_dependencies(accumulation_and_icp_node
  geometry_msgs
  message_filters
  pcl_conversions
  rclcpp
  rclcpp_action
  sensor_msgs
  tf2
  tf2_ros
  nav_msgs
  vision_msgs
  visualization_msgs
  inha_interfaces
)

ament_target_dependencies(align_to_person_node
  geometry_msgs
  message_filters
  pcl_conversions
  rclcpp
  rclcpp_action
  sensor_msgs
  tf2
  tf2_ros
  nav_msgs
  vision_msgs
  visualization_msgs
  inha_interfaces
)

target_compile_features(approach_node PUBLIC cxx_std_17)
target_compile_features(approach_head_control_node PUBLIC cxx_std_17)
target_compile_features(accumulation_and_icp_node PUBLIC cxx_std_17)
target_compile_features(align_to_person_node PUBLIC cxx_std_17)

# target_compile_definitions(align_to_person_node
#   PRIVATE
#     ${PCL_DEFINITIONS}
# )

# 일단 주석 처리 권장
# target_link_libraries(approach_node "${cpp_typesupport_target}")
# target_link_libraries(approach_head_control_node "${cpp_typesupport_target}")
# target_link_libraries(accumulation_and_icp_node "${cpp_typesupport_target}")
# target_link_libraries(align_to_person_node "${cpp_typesupport_target}")

install(
  TARGETS approach_node
  approach_head_control_node
  accumulation_and_icp_node
  align_to_person_node
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



ament_export_include_directories(include)
ament_export_dependencies(
  geometry_msgs
  message_filters
  pcl_conversions
  rclcpp
  rosidl_default_runtime
  sensor_msgs
  tf2
  tf2_ros
  vision_msgs
  visualization_msgs
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
