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

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(inha_interfaces REQUIRED)

# =========================
# Custom messages (same package)
# =========================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ApproachError.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

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
  src/target_selector.cpp
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

add_executable(bev_viewer_node
  test/bev_viewer.cpp
)

add_executable(edge_yaw_viewer_node
  test/edge_yaw_viewer.cpp
  test/trt_semantic_segmenter.cpp
  test/trt_depth_estimator.cpp
)

add_executable(depth_anything_image_test
  test/depth_anything_image_test.cpp
  test/trt_depth_estimator.cpp
)

# ── 새 4개 노드 ─────────────────────────────────────────────────────────────
add_executable(pc_detector_node
  src/pc_detector.cpp
  src/roi_filter.cpp
  src/convexhull.cpp
  src/plane_filter.cpp
  src/edge_extractor.cpp
  src/error_estimator.cpp
)

add_executable(edge_detector_node
  src/edge_detector.cpp
)

add_executable(depth_edge_detector_node
  src/depth_edge_detector.cpp
  test/trt_depth_estimator.cpp
)

add_executable(approach_manager_node
  src/approach_manager.cpp
)

add_executable(approach_controller_node
  src/approach_controller.cpp
  src/pid_controller.cpp
)

add_executable(approach_debug_logger_node
  src/approach_debug_logger.cpp
)

add_executable(visual_servo_node
  src/visual_servo.cpp
)

add_executable(align_decider_node
  src/align_decider.cpp
)

add_executable(y_decider_node
  src/y_decider.cpp
)

add_executable(y_nav_goal_decider_node
  src/y_nav_goal_decider.cpp
)

add_executable(align_measure_node
  src/align_measure.cpp
)
# ── 라이브 파라미터 튜닝 테스트 노드 ────────────────────────────────────────
add_executable(self_filter_tuner_node
  test/self_filter_tuner.cpp
)

add_executable(depth_edge_tuner_node
  test/depth_edge_tuner.cpp
  test/trt_depth_estimator.cpp
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

target_include_directories(bev_viewer_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(edge_yaw_viewer_node
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)

target_include_directories(depth_anything_image_test
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)

target_include_directories(pc_detector_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(edge_detector_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(depth_edge_detector_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(approach_manager_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(approach_controller_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(approach_debug_logger_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)

target_include_directories(visual_servo_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(align_decider_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(y_decider_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(y_nav_goal_decider_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(align_measure_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(self_filter_tuner_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(depth_edge_tuner_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    ${OpenCV_INCLUDE_DIRS}
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
target_compile_features(bev_viewer_node PUBLIC cxx_std_17)
target_compile_features(edge_yaw_viewer_node PUBLIC cxx_std_17)
target_compile_features(depth_anything_image_test PUBLIC cxx_std_17)
target_compile_features(pc_detector_node PUBLIC cxx_std_17)
target_compile_features(edge_detector_node PUBLIC cxx_std_17)
target_compile_features(depth_edge_detector_node PUBLIC cxx_std_17)
target_compile_features(approach_manager_node PUBLIC cxx_std_17)
target_compile_features(approach_controller_node PUBLIC cxx_std_17)
target_compile_features(approach_debug_logger_node PUBLIC cxx_std_17)
target_compile_features(visual_servo_node PUBLIC cxx_std_17)
target_compile_features(align_decider_node PUBLIC cxx_std_17)
target_compile_features(y_decider_node PUBLIC cxx_std_17)
target_compile_features(y_nav_goal_decider_node PUBLIC cxx_std_17)
target_compile_features(align_measure_node PUBLIC cxx_std_17)
target_compile_features(self_filter_tuner_node PUBLIC cxx_std_17)
target_compile_features(depth_edge_tuner_node PUBLIC cxx_std_17)

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

ament_target_dependencies(bev_viewer_node
  rclcpp
  sensor_msgs
  cv_bridge
)

ament_target_dependencies(edge_yaw_viewer_node
  rclcpp
  sensor_msgs
)

ament_target_dependencies(pc_detector_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  visualization_msgs
  message_filters
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
)

ament_target_dependencies(edge_detector_node
  rclcpp
  std_msgs
  sensor_msgs
  inha_interfaces
)

ament_target_dependencies(depth_edge_detector_node
  rclcpp
  std_msgs
  sensor_msgs
  inha_interfaces
)

ament_target_dependencies(approach_manager_node
  rclcpp
  rclcpp_action
  std_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  inha_interfaces
)

ament_target_dependencies(approach_controller_node
  rclcpp
  std_msgs
  geometry_msgs
)

ament_target_dependencies(approach_debug_logger_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  pcl_conversions
)

ament_target_dependencies(visual_servo_node
  rclcpp
  std_msgs
  geometry_msgs
  vision_msgs
)

ament_target_dependencies(align_decider_node
  rclcpp
  rclcpp_action
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  inha_interfaces
<<<<<<< HEAD
)

ament_target_dependencies(y_decider_node
  rclcpp
  rclcpp_action
  geometry_msgs
  sensor_msgs
  std_srvs
  visualization_msgs
  pcl_conversions
  tf2
  tf2_ros
  inha_interfaces
)

ament_target_dependencies(y_nav_goal_decider_node
  rclcpp
  rclcpp_action
  geometry_msgs
  nav_msgs
  std_srvs
  tf2
  tf2_ros
  inha_interfaces
=======
>>>>>>> c105935 (임시 얼라인)
)

ament_target_dependencies(align_measure_node
  rclcpp
  rclcpp_action
  geometry_msgs
  std_msgs
  sensor_msgs
  pcl_conversions
  tf2
  tf2_ros
  inha_interfaces
)

ament_target_dependencies(self_filter_tuner_node
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_conversions
  tf2
  tf2_ros
)

ament_target_dependencies(depth_edge_tuner_node
  rclcpp
  sensor_msgs
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

target_link_libraries(bev_viewer_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(edge_yaw_viewer_node
  ${OpenCV_LIBS}
)

target_link_libraries(depth_anything_image_test
  ${OpenCV_LIBS}
)

# TensorRT is optional so non-GPU development machines can still build the
# existing edge viewer. When found, native semantic/depth inference is enabled.
set(TENSORRT_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH
  "TensorRT prefix (for example the cafm_yolo26 Conda environment)")
file(GLOB _TENSORRT_PIP_LIB_DIRS
  "${TENSORRT_ROOT}/lib/python*/site-packages/tensorrt_libs"
)
file(GLOB _CUDA_TARGET_INCLUDE_DIRS
  "/usr/local/cuda/targets/*/include"
)
file(GLOB _CUDA_TARGET_LIB_DIRS
  "/usr/local/cuda/targets/*/lib"
  "/usr/local/cuda/targets/*/lib64"
)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS
    ${TENSORRT_ROOT}/include
    /usr/include
    /usr/include/x86_64-linux-gnu
    /usr/include/aarch64-linux-gnu
    /usr/local/include
    /usr/local/cuda/include
    /usr/src/tensorrt/include
)
find_library(TENSORRT_NVINFER_LIBRARY nvinfer
  HINTS
    ${_TENSORRT_PIP_LIB_DIRS}
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
    /usr/local/cuda/lib64
)
if(NOT TENSORRT_NVINFER_LIBRARY)
  find_file(TENSORRT_NVINFER_LIBRARY
    NAMES libnvinfer.so.10 libnvinfer.so
    HINTS ${_TENSORRT_PIP_LIB_DIRS}
  )
endif()
find_library(CUDA_RUNTIME_LIBRARY cudart
  HINTS
    ${_CUDA_TARGET_LIB_DIRS}
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/x86_64-linux/lib
    /usr/local/cuda/targets/aarch64-linux/lib
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
)
find_path(CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h
  HINTS
    ${_CUDA_TARGET_INCLUDE_DIRS}
    /usr/local/cuda/include
    /usr/local/cuda/targets/x86_64-linux/include
    /usr/local/cuda/targets/aarch64-linux/include
    /usr/include
    /usr/include/x86_64-linux-gnu
    /usr/include/aarch64-linux-gnu
)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIBRARY AND
   CUDA_RUNTIME_LIBRARY AND CUDA_RUNTIME_INCLUDE_DIR)
  message(STATUS "TensorRT found: enabling semantic and depth inference")
  get_filename_component(TENSORRT_NVINFER_DIR
    "${TENSORRT_NVINFER_LIBRARY}" DIRECTORY)
  foreach(_trt_target edge_yaw_viewer_node depth_anything_image_test
                      depth_edge_detector_node depth_edge_tuner_node)
    target_compile_definitions(${_trt_target} PRIVATE CAFM_HAS_TENSORRT=1)
    target_include_directories(${_trt_target} SYSTEM PRIVATE
      ${TENSORRT_INCLUDE_DIR}
      ${CUDA_RUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(${_trt_target}
      ${TENSORRT_NVINFER_LIBRARY}
      ${CUDA_RUNTIME_LIBRARY}
    )
    set_target_properties(${_trt_target} PROPERTIES
      BUILD_RPATH "${TENSORRT_NVINFER_DIR}"
      INSTALL_RPATH "${TENSORRT_NVINFER_DIR}"
    )
  endforeach()
else()
  message(WARNING
    "TensorRT/CUDA development files not found: viewers build without "
    "semantic/depth inference (TensorRT include=${TENSORRT_INCLUDE_DIR}, "
    "CUDA include=${CUDA_RUNTIME_INCLUDE_DIR})")
endif()

target_link_libraries(pc_detector_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(edge_detector_node
  ${OpenCV_LIBS}
)

target_link_libraries(depth_edge_detector_node
  ${OpenCV_LIBS}
)

target_link_libraries(approach_debug_logger_node
  ${PCL_LIBRARIES}
  ${PCL_IO_LIBRARIES}
)

target_link_libraries(self_filter_tuner_node
  ${PCL_LIBRARIES}
)

target_link_libraries(depth_edge_tuner_node
  ${OpenCV_LIBS}
)

target_link_libraries(y_decider_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(align_measure_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
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

  target_link_libraries(align_decider_node
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

  target_link_libraries(align_decider_node
    ${tf2_geometry_msgs_LIBRARIES}
  )
endif()

# =========================
# rosidl typesupport 링크 (자체 패키지 메시지 사용 노드)
# =========================
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

foreach(_target pc_detector_node edge_detector_node depth_edge_detector_node
                approach_manager_node approach_controller_node
                approach_debug_logger_node align_measure_node)
  target_link_libraries(${_target} "${cpp_typesupport_target}")
endforeach()

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

target_compile_definitions(approach_debug_logger_node
  PRIVATE
    ${PCL_DEFINITIONS}
)

target_compile_definitions(y_decider_node
  PRIVATE
    ${PCL_DEFINITIONS}
)

target_compile_definitions(align_measure_node
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
    bev_viewer_node
    edge_yaw_viewer_node
    depth_anything_image_test
    pc_detector_node
    edge_detector_node
    depth_edge_detector_node
    approach_manager_node
    approach_controller_node
    approach_debug_logger_node
    visual_servo_node
    align_decider_node
    y_decider_node
    y_nav_goal_decider_node
    align_measure_node
    self_filter_tuner_node
    depth_edge_tuner_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
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
