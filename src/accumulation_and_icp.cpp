#include "close_approach/accumulation_and_icp.hpp"

#include <filesystem>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <sstream>
#include <vector>

namespace {

bool hasDebugArg(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "debug" || arg == "--debug") {
      return true;
    }
  }
  return false;
}

std::vector<char *> filterDebugArgs(int argc, char **argv) {
  std::vector<char *> filtered_args;
  filtered_args.reserve(static_cast<std::size_t>(argc));
  if (argc > 0) {
    filtered_args.push_back(argv[0]);
  }

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "debug" || arg == "--debug") {
      continue;
    }
    filtered_args.push_back(argv[i]);
  }

  return filtered_args;
}

} // namespace

AccumulationAndIcp::AccumulationAndIcp(bool debug_enabled)
    : Node("accumulation_and_icp"),
      qos_best_effort_(rclcpp::QoS(rclcpp::KeepLast(10)).best_effort()),
      qos_reliable_(rclcpp::QoS(rclcpp::KeepLast(10)).reliable()),
      tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_),
      debug_enabled_(debug_enabled) {

  if (debug_enabled_) {
    debug_output_dir_ =
        std::filesystem::current_path() / "debug" / "accumulation";
    std::filesystem::create_directories(debug_output_dir_);
    RCLCPP_INFO(this->get_logger(), "Debug point clouds will be saved to %s",
                debug_output_dir_.c_str());
  }

  // --- 파라미터 선언 ---
  this->declare_parameter<std::string>("lidar_topic_name", "/livox/lidar");
  this->declare_parameter<std::string>("odom_topic_name", "/odom");
  this->declare_parameter<std::string>("lidar_frame", "livox_lidar");
  this->declare_parameter<std::string>("base_frame", "base");
  this->declare_parameter<std::string>("map_frame", "map");
  this->declare_parameter<float>("voxel_leaf_size", 0.05F);
  this->declare_parameter<int>("icp_max_iterations", 50);
  this->declare_parameter<double>("icp_transform_epsilon", 1e-8);
  this->declare_parameter<double>("icp_max_correspondence_distance", 0.5);
  this->declare_parameter<float>("roi_x_min", 0.1F);
  this->declare_parameter<float>("roi_x_max", 5.0F);
  this->declare_parameter<float>("roi_y_min", -2.5F);
  this->declare_parameter<float>("roi_y_max", 2.5F);
  this->declare_parameter<float>("roi_z_min", 0.0F);
  this->declare_parameter<float>("roi_z_max", 1.5F);

  this->get_parameter("lidar_topic_name", lidar_topic_name_);
  this->get_parameter("odom_topic_name", odom_topic_name_);
  this->get_parameter("lidar_frame", lidar_frame_);
  this->get_parameter("base_frame", base_frame_);
  this->get_parameter("map_frame", map_frame_);
  this->get_parameter("voxel_leaf_size", voxel_leaf_size_);
  this->get_parameter("icp_max_iterations", icp_max_iterations_);
  this->get_parameter("icp_transform_epsilon", icp_transform_epsilon_);
  this->get_parameter("icp_max_correspondence_distance",
                      icp_max_correspondence_distance_);
  this->get_parameter("roi_x_min", roi_x_min_);
  this->get_parameter("roi_x_max", roi_x_max_);
  this->get_parameter("roi_y_min", roi_y_min_);
  this->get_parameter("roi_y_max", roi_y_max_);
  this->get_parameter("roi_z_min", roi_z_min_);
  this->get_parameter("roi_z_max", roi_z_max_);

  // --- Service Server ---
  service_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  accum_service_server_ = this->create_service<AccumService>(
      "accumulate",
      std::bind(&AccumulationAndIcp::handle_accumulate_request, this, std::placeholders::_1, std::placeholders::_2),
      rclcpp::ServicesQoS(), service_callback_group_);

  // --- Publisher ---
  accumulated_cloud_publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/accumulation/accumulated_cloud", qos_reliable_);

  // --- Odom Subscriber (항상 켜두고 최신 값만 캐싱) ---
  odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_name_, qos_best_effort_,
      std::bind(&AccumulationAndIcp::odomCallback, this,
                std::placeholders::_1));

  // 누적 클라우드 초기화
  accumulated_cloud_ =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  prev_odom_pose_ = Eigen::Matrix4f::Identity();
  curr_odom_pose_ = Eigen::Matrix4f::Identity();

  RCLCPP_INFO(this->get_logger(), "AccumulationAndIcp node started.");
}

// ============================================================
//  Service Server 콜백
// ============================================================

void AccumulationAndIcp::handle_accumulate_request(
    const std::shared_ptr<AccumService::Request> request,
    std::shared_ptr<AccumService::Response> response) {

  RCLCPP_INFO(this->get_logger(),
              "Received accumulation request: start=%d, hz=%.1f, max_scans=%d",
              request->start, request->hz, request->max_scans);

  if (request->start) {
    if (algorithm_running_) {
      stopAlgorithm();
    }

    resetState();
    max_scans_ = request->max_scans;
    double hz = request->hz > 0.0 ? request->hz : 1.0;
    startAlgorithm(hz);
    response->success = true;
    response->total_accumulated = accumulated_count_.load();
    RCLCPP_INFO(this->get_logger(),
                "Accumulation started at %.1f Hz, max_scans=%d", hz,
                max_scans_.load());
    return;
  } else {
    stopAlgorithm();
    response->success = true;
    response->total_accumulated = accumulated_count_.load();
    RCLCPP_INFO(this->get_logger(), "Accumulation stopped by user");
    return;
  }
}

// ============================================================
//  Start / Stop / Reset
// ============================================================

void AccumulationAndIcp::startAlgorithm(double hz) {
  // LiDAR 구독 시작
  pointcloud_subscriber_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(
          lidar_topic_name_, qos_best_effort_,
          std::bind(&AccumulationAndIcp::pointcloudCallback, this,
                    std::placeholders::_1));

  // Timer 시작 (hz에 맞춰 누적)
  auto period = std::chrono::duration<double>(1.0 / hz);
  accumulation_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&AccumulationAndIcp::accumulationTimerCallback, this));

  algorithm_running_ = true;
  RCLCPP_INFO(this->get_logger(), "Algorithm started (period=%.3fs)", 1.0 / hz);
}

void AccumulationAndIcp::stopAlgorithm() {
  algorithm_running_ = false;

  if (accumulation_timer_) {
    accumulation_timer_->cancel();
    accumulation_timer_.reset();
  }
  if (pointcloud_subscriber_) {
    pointcloud_subscriber_.reset();
  }

  RCLCPP_INFO(this->get_logger(), "Algorithm stopped");
}

void AccumulationAndIcp::resetState() {
  accumulated_cloud_->clear();
  accumulated_count_ = 0;
  has_new_scan_ = false;
  has_prev_odom_ = false;
  prev_odom_pose_ = Eigen::Matrix4f::Identity();
  curr_odom_pose_ = Eigen::Matrix4f::Identity();
  trajectory_history_.clear();
  trajectory_index_ = 0;
  current_trajectory_ = Trajectory();
}

// ============================================================
//  Subscriber 콜백
// ============================================================

void AccumulationAndIcp::pointcloudCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
  if (!algorithm_running_) {
    return;
  }

  auto cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*msg, *cloud);

  // NaN / Inf 제거
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

  if (cloud->empty()) {
    return;
  }

  // base 기준 ROI를 적용하기 위해 먼저 lidar -> base 변환을 수행한다.
  geometry_msgs::msg::TransformStamped tf_lidar_to_base;
  try {
    tf_lidar_to_base =
        tf_buffer_.lookupTransform(base_frame_, lidar_frame_, msg->header.stamp,
                                   rclcpp::Duration(0, 50000000));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "TF lookup failed (%s -> %s): %s",
                         lidar_frame_.c_str(), base_frame_.c_str(), ex.what());
    return;
  }

  Eigen::Affine3f lidar_to_base_tf = Eigen::Affine3f::Identity();
  const auto &base_t = tf_lidar_to_base.transform.translation;
  const auto &base_r = tf_lidar_to_base.transform.rotation;
  lidar_to_base_tf.translation() << base_t.x, base_t.y, base_t.z;
  lidar_to_base_tf.rotate(Eigen::Quaternionf(base_r.w, base_r.x, base_r.y,
                                             base_r.z));

  pcl::PointCloud<pcl::PointXYZ>::Ptr base_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*cloud, *base_cloud, lidar_to_base_tf);

  SetROI(base_cloud);
  if (base_cloud->empty()) {
    RCLCPP_DEBUG(this->get_logger(), "Scan empty after base ROI filtering");
    return;
  }

  // ROI 적용 후 누적을 위해 base -> map 변환을 수행한다.
  geometry_msgs::msg::TransformStamped tf_base_to_map;
  try {
    tf_base_to_map =
        tf_buffer_.lookupTransform(map_frame_, base_frame_, msg->header.stamp,
                                   rclcpp::Duration(0, 50000000));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "TF lookup failed (%s -> %s): %s",
                         base_frame_.c_str(), map_frame_.c_str(), ex.what());
    return;
  }

  Eigen::Affine3f base_to_map_tf = Eigen::Affine3f::Identity();
  const auto &map_t = tf_base_to_map.transform.translation;
  const auto &map_r = tf_base_to_map.transform.rotation;
  base_to_map_tf.translation() << map_t.x, map_t.y, map_t.z;
  base_to_map_tf.rotate(
      Eigen::Quaternionf(map_r.w, map_r.x, map_r.y, map_r.z));

  pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*base_cloud, *map_cloud, base_to_map_tf);
  saveDebugCloud(map_cloud, "incoming_scan");

  // 최신 스캔 버퍼에 저장
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    latest_scan_ = map_cloud;
    latest_scan_stamp_ = msg->header.stamp;
    has_new_scan_ = true;
  }
}

void AccumulationAndIcp::odomCallback(
    const nav_msgs::msg::Odometry::ConstSharedPtr &msg) {
  std::lock_guard<std::mutex> lock(odom_mutex_);
  curr_odom_pose_ = odomMsgToMatrix(*msg);
}

// ============================================================
//  Timer 콜백 (핵심: 누적 로직)
// ============================================================

void AccumulationAndIcp::accumulationTimerCallback() {
  if (!algorithm_running_) {
    return;
  }

  // 새 스캔이 있는지 확인
  pcl::PointCloud<pcl::PointXYZ>::Ptr current_scan;
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    if (!has_new_scan_ || !latest_scan_ || latest_scan_->empty()) {
      return;
    }
    current_scan = latest_scan_;
    has_new_scan_ = false;
  }

  // 다운샘플링
  downsample(current_scan);

  if (current_scan->empty()) {
    RCLCPP_WARN(this->get_logger(), "Scan empty after downsampling");
    return;
  }

  // --- 첫 번째 스캔: 그냥 누적 클라우드로 설정 ---
  if (accumulated_cloud_->empty()) {
    *accumulated_cloud_ = *current_scan;
    accumulated_count_++;

    // odom 초기값 저장
    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      prev_odom_pose_ = curr_odom_pose_;
      has_prev_odom_ = true;
    }

    saveDebugCloud(accumulated_cloud_, "accumulated");
    publishAccumulatedCloud();
    RCLCPP_INFO(this->get_logger(), "First scan accumulated (%zu points)",
                accumulated_cloud_->size());

    if (max_scans_.load() > 0 && accumulated_count_.load() >= max_scans_.load()) {
      stopAlgorithm();
      RCLCPP_INFO(this->get_logger(),
                  "Accumulation complete: %d scans accumulated",
                  accumulated_count_.load());
    }
    return;
  }

  // --- 2번째 스캔부터: Odom 초기값 → ICP ---

  // Odom delta 계산 (초기값)
  Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f traj = Eigen::Matrix4f::Identity();
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (has_prev_odom_) {
      // delta = prev_inv * curr (이전 대비 현재의 상대 변환)
      initial_guess = prev_odom_pose_.inverse() * curr_odom_pose_;
      traj = prev_odom_pose_;

      prev_odom_pose_ = curr_odom_pose_;
    }
  }

  // ICP 수행: accumulated(target) ← current_scan(source)
  Eigen::Matrix4f refined_transform =
      runICP(accumulated_cloud_, current_scan, initial_guess);

  // 정합된 스캔을 누적 클라우드에 합침
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_scan(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*current_scan, *aligned_scan, refined_transform);

  traj = refined_transform * traj;
  current_trajectory_.x = traj(0, 3);
  current_trajectory_.y = traj(1, 3);
  current_trajectory_.theta = std::atan2(traj(1, 0), traj(0, 0));
  current_trajectory_.index = trajectory_index_++;

  trajectory_history_.push_back(current_trajectory_);

  saveDebugCloud(aligned_scan, "aligned_scan");

  *accumulated_cloud_ += *aligned_scan;

  // 누적 클라우드도 주기적으로 다운샘플링 (메모리 폭발 방지)
  downsample(accumulated_cloud_);

  accumulated_count_++;
  saveDebugCloud(accumulated_cloud_, "accumulated");
  publishAccumulatedCloud();

  RCLCPP_INFO(this->get_logger(),
              "Scan #%d accumulated (total %zu points, ICP converged)",
              accumulated_count_.load(), accumulated_cloud_->size());

  if (max_scans_.load() > 0 && accumulated_count_.load() >= max_scans_.load()) {
    stopAlgorithm();
    RCLCPP_INFO(this->get_logger(), "Accumulation complete: %d scans accumulated",
                accumulated_count_.load());
  }
}

// ============================================================
//  ICP
// ============================================================

Eigen::Matrix4f
AccumulationAndIcp::runICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
                           const pcl::PointCloud<pcl::PointXYZ>::Ptr &source,
                           const Eigen::Matrix4f &initial_guess) {

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(source);
  icp.setInputTarget(target);
  icp.setMaximumIterations(icp_max_iterations_);
  icp.setTransformationEpsilon(icp_transform_epsilon_);
  icp.setMaxCorrespondenceDistance(icp_max_correspondence_distance_);

  pcl::PointCloud<pcl::PointXYZ> aligned;
  icp.align(aligned, initial_guess);

  if (icp.hasConverged()) {
    RCLCPP_DEBUG(this->get_logger(), "ICP converged, fitness: %.6f",
                 icp.getFitnessScore());
    return icp.getFinalTransformation();
  } else {
    RCLCPP_WARN(this->get_logger(),
                "ICP did NOT converge, using initial guess");
    return initial_guess;
  }
}

// ============================================================
//  유틸리티
// ============================================================

void AccumulationAndIcp::downsample(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setInputCloud(cloud);
  voxel.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);

  auto filtered =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  voxel.filter(*filtered);
  cloud = filtered;
}

void AccumulationAndIcp::SetROI(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(roi_x_min_, roi_x_max_);
  pass.filter(*cloud);
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(roi_y_min_, roi_y_max_);
  pass.filter(*cloud);
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(roi_z_min_, roi_z_max_);
  pass.filter(*cloud);
}

void AccumulationAndIcp::publishAccumulatedCloud() {
  // map 프레임에 쌓인 누적 클라우드를 base 프레임으로 변환해서 퍼블리시
  geometry_msgs::msg::TransformStamped tf_map_to_base;
  try {
    tf_map_to_base =
        tf_buffer_.lookupTransform(base_frame_, map_frame_, rclcpp::Time(0),
                                   rclcpp::Duration(0, 50000000));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Cannot publish: TF %s -> %s failed: %s",
                         map_frame_.c_str(), base_frame_.c_str(), ex.what());
    return;
  }

  Eigen::Affine3f eigen_tf = Eigen::Affine3f::Identity();
  const auto &t = tf_map_to_base.transform.translation;
  const auto &r = tf_map_to_base.transform.rotation;
  eigen_tf.translation() << t.x, t.y, t.z;
  eigen_tf.rotate(Eigen::Quaternionf(r.w, r.x, r.y, r.z));

  pcl::PointCloud<pcl::PointXYZ>::Ptr base_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*accumulated_cloud_, *base_cloud, eigen_tf);

  sensor_msgs::msg::PointCloud2 output;
  pcl::toROSMsg(*base_cloud, output);
  output.header.frame_id = base_frame_;
  output.header.stamp = this->now();
  accumulated_cloud_publisher_->publish(output);
}

void AccumulationAndIcp::saveDebugCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const std::string &stage) {
  if (!debug_enabled_ || !cloud || cloud->empty()) {
    return;
  }

  std::ostringstream filename;
  filename << std::setw(6) << std::setfill('0') << debug_cloud_index_ << "_"
           << stage << ".pcd";

  const auto file_path = debug_output_dir_ / filename.str();
  const int result = pcl::io::savePCDFileBinary(file_path.string(), *cloud);
  if (result == 0) {
    ++debug_cloud_index_;
    RCLCPP_DEBUG(this->get_logger(), "Saved debug cloud: %s",
                 file_path.c_str());
  } else {
    RCLCPP_WARN(this->get_logger(), "Failed to save debug cloud: %s",
                file_path.c_str());
  }
}

Eigen::Matrix4f
AccumulationAndIcp::odomMsgToMatrix(const nav_msgs::msg::Odometry &odom) {
  Eigen::Affine3f affine = Eigen::Affine3f::Identity();
  const auto &p = odom.pose.pose.position;
  const auto &q = odom.pose.pose.orientation;

  affine.translation() << p.x, p.y, p.z;
  affine.rotate(Eigen::Quaternionf(q.w, q.x, q.y, q.z));

  return affine.matrix();
}

// ============================================================
//  main
// ============================================================

int main(int argc, char **argv) {
  const bool debug_enabled = hasDebugArg(argc, argv);
  std::vector<char *> filtered_args = filterDebugArgs(argc, argv);
  rclcpp::init(static_cast<int>(filtered_args.size()), filtered_args.data());
  
  auto node = std::make_shared<AccumulationAndIcp>(debug_enabled);
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  
  rclcpp::shutdown();
  return 0;
}
