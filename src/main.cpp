#include "close_approach/main.hpp"

#include <filesystem>
#include <functional>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>

namespace {

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct DebugColor {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
};

double elapsed_ms(const Clock::time_point &from) {
  return std::chrono::duration_cast<Ms>(Clock::now() - from).count();
}

std::string currentTimeForFilename() {
  const auto now = std::chrono::system_clock::now();
  const auto now_time_t = std::chrono::system_clock::to_time_t(now);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

  std::tm local_time{};
  localtime_r(&now_time_t, &local_time);

  std::ostringstream stamp;
  stamp << std::put_time(&local_time, "%Y%m%d_%H%M%S") << "_"
        << std::setw(3) << std::setfill('0') << ms.count();
  return stamp.str();
}

void applySpatialRoiBounds(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                           float x_min, float x_max, float y_abs_max,
                           float z_max) {
  auto roi_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  roi_cloud->points.reserve(cloud->points.size());
  for (const auto &pt : cloud->points) {
    if (pt.x < x_min || pt.x > x_max) {
      continue;
    }
    if (std::abs(pt.y) > y_abs_max) {
      continue;
    }
    if (pt.z > z_max) {
      continue;
    }
    roi_cloud->points.push_back(pt);
  }
  roi_cloud->width = static_cast<std::uint32_t>(roi_cloud->points.size());
  roi_cloud->height = 1;
  roi_cloud->is_dense = false;
  cloud = roi_cloud;
}

Eigen::Affine2f transformToAffine2D(
    const geometry_msgs::msg::TransformStamped &tf) {
  const float yaw = static_cast<float>(tf2::getYaw(tf.transform.rotation));
  const float cy = std::cos(yaw);
  const float sy = std::sin(yaw);

  Eigen::Affine2f affine = Eigen::Affine2f::Identity();
  affine.linear()(0, 0) = cy;
  affine.linear()(0, 1) = -sy;
  affine.linear()(1, 0) = sy;
  affine.linear()(1, 1) = cy;
  affine.translation().x() = static_cast<float>(tf.transform.translation.x);
  affine.translation().y() = static_cast<float>(tf.transform.translation.y);
  return affine;
}

DebugColor debugPaletteColor(std::size_t idx) {
  static const std::array<DebugColor, 8> colors = {
      DebugColor{200, 200, 200}, DebugColor{120, 180, 255},
      DebugColor{255, 180, 100}, DebugColor{180, 255, 120},
      DebugColor{255, 120, 200}, DebugColor{120, 255, 220},
      DebugColor{220, 160, 255}, DebugColor{255, 240, 120}};
  return colors[idx % colors.size()];
}

void addDebugPoint(pcl::PointCloud<pcl::PointXYZRGB> &cloud,
                   const Eigen::Vector2f &p, float z,
                   const DebugColor &color) {
  pcl::PointXYZRGB pt;
  pt.x = p.x();
  pt.y = p.y();
  pt.z = z;
  pt.r = color.r;
  pt.g = color.g;
  pt.b = color.b;
  cloud.points.push_back(pt);
}

void addDebugSegment(pcl::PointCloud<pcl::PointXYZRGB> &cloud,
                     const Eigen::Vector2f &a, const Eigen::Vector2f &b,
                     float z, const DebugColor &color, int samples = 80) {
  for (int i = 0; i <= samples; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(samples);
    addDebugPoint(cloud, a + t * (b - a), z, color);
  }
}

void addDebugCross(pcl::PointCloud<pcl::PointXYZRGB> &cloud,
                   const Eigen::Vector2f &p, float z,
                   const DebugColor &color, float size = 0.04F) {
  addDebugSegment(cloud, p + Eigen::Vector2f(-size, 0.0F),
                  p + Eigen::Vector2f(size, 0.0F), z, color, 16);
  addDebugSegment(cloud, p + Eigen::Vector2f(0.0F, -size),
                  p + Eigen::Vector2f(0.0F, size), z, color, 16);
}

void addDebugObb(pcl::PointCloud<pcl::PointXYZRGB> &cloud, const OBB &obb,
                 float z, const DebugColor &color) {
  const Eigen::Vector2f p1 =
      obb.center + (obb.length1 * 0.5F) * obb.axis1 +
      (obb.length2 * 0.5F) * obb.axis2;
  const Eigen::Vector2f p2 =
      obb.center - (obb.length1 * 0.5F) * obb.axis1 +
      (obb.length2 * 0.5F) * obb.axis2;
  const Eigen::Vector2f p3 =
      obb.center - (obb.length1 * 0.5F) * obb.axis1 -
      (obb.length2 * 0.5F) * obb.axis2;
  const Eigen::Vector2f p4 =
      obb.center + (obb.length1 * 0.5F) * obb.axis1 -
      (obb.length2 * 0.5F) * obb.axis2;
  addDebugSegment(cloud, p1, p2, z, color, 60);
  addDebugSegment(cloud, p2, p3, z, color, 60);
  addDebugSegment(cloud, p3, p4, z, color, 60);
  addDebugSegment(cloud, p4, p1, z, color, 60);
}

void addDebugAxes(pcl::PointCloud<pcl::PointXYZRGB> &cloud, float z,
                  float length) {
  const Eigen::Vector2f origin(0.0F, 0.0F);
  addDebugSegment(cloud, origin, Eigen::Vector2f(length, 0.0F), z,
                  {255, 0, 0}, 120);
  addDebugSegment(cloud, origin, Eigen::Vector2f(0.0F, length), z + 0.02F,
                  {0, 255, 0}, 120);
  addDebugCross(cloud, origin, z + 0.04F, {255, 255, 255}, 0.05F);
}

} // namespace

ApproachNode::ApproachNode()
    : Node("approach_node"),
      qos_best_effort_(rclcpp::QoS(rclcpp::KeepLast(10)).best_effort()),
      qos_reliable_(rclcpp::QoS(rclcpp::KeepLast(10)).reliable()),
      tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {

  this->declare_parameter<bool>("debug", true);
  this->declare_parameter<double>("debug_save_period_sec", 1.0);
  this->declare_parameter<std::string>("debug_output_dir",
                                       "/home/thor/inha_logs/module/close_approach");
  this->get_parameter("debug", debug_enabled_);
  this->get_parameter("debug_save_period_sec", debug_save_period_sec_);
  std::string debug_output_dir;
  this->get_parameter("debug_output_dir", debug_output_dir);
  debug_output_dir_ = debug_output_dir;

  if (debug_enabled_) {
    std::filesystem::create_directories(debug_output_dir_);
    RCLCPP_INFO(this->get_logger(), "Debug files will be saved to %s",
                debug_output_dir_.c_str());
  }

  /*
  객체 생성
  */
  roi_filter_ = std::make_shared<Filter>();
  convex_hull_ = std::make_shared<ConvexHull>();
  plane_filter_ = std::make_shared<PlaneFilter>();
  pid_controller_ = std::make_shared<PIDController>();
  edge_extractor_ = std::make_shared<EdgeExtractor>();
  error_estimator_ = std::make_shared<ErrorEstimator>();
  target_selector_ = std::make_shared<TargetSelector>();

  /*
  파라미터 선언
  */
  this->declare_parameter<std::string>(
      "pointcloud_topic_name", "/camera/camera_head/depth/color/points");
  this->declare_parameter<std::string>("lidar_topic_name", "/livox/lidar");
  this->declare_parameter<std::string>("info_topic_name",
                                       "/camera/camera_head/color/camera_info");
  this->declare_parameter<std::string>("target_frame", "base_nav");
  this->declare_parameter<std::string>("odom_frame", "odom");
  this->declare_parameter<float>("roi_x_min", 0.1F);
  this->declare_parameter<float>("roi_x_max", 2.0F);
  this->declare_parameter<float>("roi_y_abs_max", 0.8F);
  this->declare_parameter<float>("roi_z_max", 1.5F);
  this->declare_parameter<float>("leaf_size", 0.03F);
  this->declare_parameter<int>("mean_k", 50);
  this->declare_parameter<float>("stddev_mul_thresh", 0.5F);
  this->declare_parameter<float>("ground_height", 0.1F);
  this->declare_parameter<float>("cluster_tolerance", 0.05F);
  this->declare_parameter<int>("min_cluster_size", 100);
  this->declare_parameter<int>("max_cluster_size", 10000);
  this->declare_parameter<float>("min_cluster_area", 0.09F);
  this->declare_parameter<float>("fx", 605.7842407226562F);
  this->declare_parameter<float>("fy", 604.492919921875F);
  this->declare_parameter<float>("cx", 323.7266845703125F);
  this->declare_parameter<float>("cy", 250.73828125F);
  this->declare_parameter<float>("kp_x", 0.25F);
  this->declare_parameter<float>("kp_y", 1.5F);
  this->declare_parameter<float>("kp_theta", 1.2F);
  this->declare_parameter<float>("ki_x", 0.0F);
  this->declare_parameter<float>("ki_y", 0.0F);
  this->declare_parameter<float>("ki_theta", 0.0F);
  this->declare_parameter<float>("kd_x", 0.0F);
  this->declare_parameter<float>("kd_y", 0.0F);
  this->declare_parameter<float>("kd_theta", 0.0F);
  this->declare_parameter<float>("tol_x", 0.03F);     // 3cm
  this->declare_parameter<float>("tol_y", 0.08F);     // 8cm
  this->declare_parameter<float>("tol_theta", 0.08F); // 4~5도
  this->declare_parameter<float>("target_standoff_distance", 0.3F); // 로봇과 목표 사이의 간격
  this->declare_parameter<float>("max_v", 0.10F);
  this->declare_parameter<float>("max_w", 0.2F);
  this->declare_parameter<float>("decel_dist_max", 0.20F);
  this->declare_parameter<float>("decel_dist_min", 0.05F);
  this->declare_parameter<float>("decel_ratio", 0.5F);
  this->declare_parameter<float>("align_timeout_sec", 5.0F);
  this->declare_parameter<float>("dwell_duration_sec", 2.0F);
  this->declare_parameter<float>("spike_dy_max", 0.25F);
  this->declare_parameter<float>("spike_dtheta_max", 0.25F);
  this->declare_parameter<int>("max_consecutive_outliers", 5);
  this->declare_parameter<float>("trail_min_dist", 0.03F);
  this->declare_parameter<float>("trail_min_yaw", 0.052F);
  this->declare_parameter<float>("lidar_max_age_sec", 0.3F);
  this->declare_parameter<float>("target_selector.edge_threshold", 0.03F);
  this->declare_parameter<float>("target_selector.l_shape_step_deg", 1.0F);
  this->declare_parameter<float>("target_selector.l_shape_sigma", 0.03F);
  this->declare_parameter<float>("target_selector.ray_segment_tolerance", 0.03F);
  this->declare_parameter<float>("target_selector.min_target_edge_support", 0.02F);
  this->declare_parameter<float>("target_selector.min_fill_ratio", 0.50F);
  this->declare_parameter<float>("target_selector.circle_rmse_threshold", 0.04F);
  this->declare_parameter<float>("target_selector.circle_min_radius", 0.15F);
  this->declare_parameter<float>("target_selector.circle_max_radius", 0.8F);
  this->declare_parameter<int>("target_selector.acquire_confirm_frames", 3);
  this->declare_parameter<int>("target_selector.relock_confirm_frames", 3);
  this->declare_parameter<int>("target_selector.max_lost_frames", 8);
  this->declare_parameter<float>("target_selector.acquire_hit_gate", 0.15F);
  this->declare_parameter<float>("target_selector.acquire_yaw_gate", 0.17F);
  this->declare_parameter<float>("target_selector.lock_hit_gate", 0.30F);
  this->declare_parameter<float>("target_selector.lock_yaw_gate", 0.35F);
  this->declare_parameter<float>("target_selector.lock_min_length_ratio", 0.50F);
  this->declare_parameter<float>("target_selector.lock_max_length_ratio", 2.00F);
  this->declare_parameter<float>("target_selector.lock_min_area_ratio", 0.40F);
  this->declare_parameter<float>("target_selector.lock_max_area_ratio", 2.50F);
  this->declare_parameter<float>("target_selector.lock_min_normal_dot", 0.70F);
  this->declare_parameter<float>("target_selector.score_hit_weight", 2.0F);
  this->declare_parameter<float>("target_selector.score_yaw_weight", 1.0F);
  this->declare_parameter<float>("target_selector.score_length_weight", 0.5F);
  this->declare_parameter<float>("target_selector.score_area_weight", 0.3F);
  this->declare_parameter<float>("target_selector.score_support_weight", 0.5F);

  this->get_parameter("pointcloud_topic_name", pointcloud_topic_name_);
  this->get_parameter("lidar_topic_name", lidar_topic_name_);
  this->get_parameter("info_topic_name", info_topic_name_);
  this->get_parameter("target_frame", target_frame_);
  this->get_parameter("odom_frame", odom_frame_);
  this->get_parameter("roi_x_min", roi_x_min_);
  this->get_parameter("roi_x_max", roi_x_max_);
  this->get_parameter("roi_y_abs_max", roi_y_abs_max_);
  this->get_parameter("roi_z_max", roi_z_max_);
  this->get_parameter("leaf_size", leaf_size_);
  this->get_parameter("mean_k", mean_k_);
  this->get_parameter("stddev_mul_thresh", stddev_mul_thresh_);
  this->get_parameter("ground_height", ground_height_);
  this->get_parameter("cluster_tolerance", cluster_tolerance_);
  this->get_parameter("min_cluster_size", min_cluster_size_);
  this->get_parameter("max_cluster_size", max_cluster_size_);
  this->get_parameter("min_cluster_area", min_cluster_area_);
  this->get_parameter("fx", fx_);
  this->get_parameter("fy", fy_);
  this->get_parameter("cx", cx_);
  this->get_parameter("cy", cy_);
  this->get_parameter("kp_x", kp_x_);
  this->get_parameter("kp_y", kp_y_);
  this->get_parameter("kp_theta", kp_theta_);
  this->get_parameter("ki_x", ki_x_);
  this->get_parameter("ki_y", ki_y_);
  this->get_parameter("ki_theta", ki_theta_);
  this->get_parameter("kd_x", kd_x_);
  this->get_parameter("kd_y", kd_y_);
  this->get_parameter("kd_theta", kd_theta_);
  this->get_parameter("tol_x", tol_x_);
  this->get_parameter("tol_y", tol_y_);
  this->get_parameter("tol_theta", tol_theta_);
  this->get_parameter("target_standoff_distance", target_standoff_distance_);
  this->get_parameter("max_v", max_v_);
  this->get_parameter("max_w", max_w_);
  this->get_parameter("decel_dist_max", decel_dist_max_);
  this->get_parameter("decel_dist_min", decel_dist_min_);
  this->get_parameter("decel_ratio", decel_ratio_);
  this->get_parameter("align_timeout_sec", align_timeout_sec_);
  this->get_parameter("dwell_duration_sec", dwell_duration_sec_);
  this->get_parameter("spike_dy_max", spike_dy_max_);
  this->get_parameter("spike_dtheta_max", spike_dtheta_max_);
  this->get_parameter("max_consecutive_outliers", max_consecutive_outliers_);
  this->get_parameter("trail_min_dist", trail_min_dist_);
  this->get_parameter("trail_min_yaw", trail_min_yaw_);
  this->get_parameter("lidar_max_age_sec", lidar_max_age_sec_);
  this->get_parameter("target_selector.edge_threshold",
                      target_selector_params_.edge_threshold);
  this->get_parameter("target_selector.l_shape_step_deg",
                      target_selector_params_.l_shape_angle_step_deg);
  this->get_parameter("target_selector.l_shape_sigma",
                      target_selector_params_.l_shape_sigma);
  this->get_parameter("target_selector.ray_segment_tolerance",
                      target_selector_params_.ray_segment_tolerance);
  this->get_parameter("target_selector.min_target_edge_support",
                      target_selector_params_.min_target_edge_support_ratio);
  this->get_parameter("target_selector.min_fill_ratio",
                      target_selector_params_.min_fill_ratio);
  this->get_parameter("target_selector.circle_rmse_threshold",
                      target_selector_params_.circle_rmse_threshold);
  this->get_parameter("target_selector.circle_min_radius",
                      target_selector_params_.circle_min_radius);
  this->get_parameter("target_selector.circle_max_radius",
                      target_selector_params_.circle_max_radius);
  this->get_parameter("target_selector.acquire_confirm_frames",
                      target_selector_params_.acquire_confirm_frames);
  this->get_parameter("target_selector.relock_confirm_frames",
                      target_selector_params_.relock_confirm_frames);
  this->get_parameter("target_selector.max_lost_frames",
                      target_selector_params_.max_lost_frames);
  this->get_parameter("target_selector.acquire_hit_gate",
                      target_selector_params_.acquire_hit_gate);
  this->get_parameter("target_selector.acquire_yaw_gate",
                      target_selector_params_.acquire_yaw_gate);
  this->get_parameter("target_selector.lock_hit_gate",
                      target_selector_params_.lock_hit_gate);
  this->get_parameter("target_selector.lock_yaw_gate",
                      target_selector_params_.lock_yaw_gate);
  this->get_parameter("target_selector.lock_min_length_ratio",
                      target_selector_params_.lock_min_length_ratio);
  this->get_parameter("target_selector.lock_max_length_ratio",
                      target_selector_params_.lock_max_length_ratio);
  this->get_parameter("target_selector.lock_min_area_ratio",
                      target_selector_params_.lock_min_area_ratio);
  this->get_parameter("target_selector.lock_max_area_ratio",
                      target_selector_params_.lock_max_area_ratio);
  this->get_parameter("target_selector.lock_min_normal_dot",
                      target_selector_params_.lock_min_normal_dot);
  this->get_parameter("target_selector.score_hit_weight",
                      target_selector_params_.score_hit_weight);
  this->get_parameter("target_selector.score_yaw_weight",
                      target_selector_params_.score_yaw_weight);
  this->get_parameter("target_selector.score_length_weight",
                      target_selector_params_.score_length_weight);
  this->get_parameter("target_selector.score_area_weight",
                      target_selector_params_.score_area_weight);
  this->get_parameter("target_selector.score_support_weight",
                      target_selector_params_.score_support_weight);

  target_selector_params_.cluster_tolerance = cluster_tolerance_;
  target_selector_params_.min_cluster_size = min_cluster_size_;
  target_selector_params_.max_cluster_size = max_cluster_size_;
  target_selector_params_.min_cluster_area = min_cluster_area_;

  /*
  ROS2 Publisher && Subscriber 설정
  */

  this->approach_action_server_ = rclcpp_action::create_server<ApproachAction>(
      this, "approach",
      std::bind(&ApproachNode::handle_goal, this, std::placeholders::_1,
                std::placeholders::_2),
      std::bind(&ApproachNode::handle_cancel, this,
                std::placeholders::_1),
      std::bind(&ApproachNode::handle_accepted, this,
                std::placeholders::_1));
  
  camera_info_subscriber_ = this->create_subscription<CameraInfoMsg>(
      info_topic_name_, qos_best_effort_,
      std::bind(&ApproachNode::cameraInfoCallback, this,
                std::placeholders::_1));

  cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel", qos_reliable_);
  filtered_pointcloud_publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/approach/filtered_pointcloud", qos_best_effort_);
  obb_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
      "/approach/obb_marker", qos_reliable_);
  target_edge_publisher_ =
      this->create_publisher<visualization_msgs::msg::Marker>(
          "/approach/target_edge_marker", qos_reliable_);
  debugging_pointcloud_publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/approach/debugging_pointcloud", qos_best_effort_);

  /*
  후진용 trail publisher (transient_local + reliable: 늦게 뜬 retreat_node도
  최신 trail 한 번 받음). odom 프레임 기준 base 위치 시퀀스.
  */
  rclcpp::QoS trail_qos(rclcpp::KeepLast(1));
  trail_qos.reliable();
  trail_qos.transient_local();
  trail_publisher_ =
      this->create_publisher<nav_msgs::msg::Path>("/approach/trail", trail_qos);

  trail_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&ApproachNode::recordTrailPose, this));

  /*
  파라미터설정
  */
  roi_filter_->setParameters(leaf_size_, mean_k_, stddev_mul_thresh_,
                             ground_height_, cluster_tolerance_,
                             min_cluster_size_, max_cluster_size_);
  pid_controller_->setParameters(kp_x_, kp_y_, kp_theta_, ki_x_, ki_y_,
                                 ki_theta_, kd_x_, kd_y_, kd_theta_);
  pid_controller_->setLimits(max_v_, max_w_);
  target_selector_->setParameters(target_selector_params_);
}
/*
Ros2 Action 관련 함수들
*/
rclcpp_action::GoalResponse
ApproachNode::handle_goal(const rclcpp_action::GoalUUID &uuid,
                          std::shared_ptr<const ApproachAction::Goal> goal) {
  (void)uuid;
  if (!goal || !std::isfinite(goal->goal_distance) ||
      goal->goal_distance <= 0.0F) {
    RCLCPP_WARN(this->get_logger(),
                "Rejecting approach goal: invalid goal_distance=%.3f",
                goal ? goal->goal_distance : -1.0F);
    return rclcpp_action::GoalResponse::REJECT;
  }

  RCLCPP_INFO(this->get_logger(),
              "Received approach goal: goal_distance=%.3f",
              goal->goal_distance);
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse ApproachNode::handle_cancel(
    const std::shared_ptr<GoalHandleApproach> goal_handle) {
  RCLCPP_INFO(this->get_logger(), "Received approach cancel");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void ApproachNode::handle_accepted(
    const std::shared_ptr<GoalHandleApproach> goal_handle) {
  RCLCPP_INFO(this->get_logger(), "Approach goal accepted");
  std::thread(std::bind(&ApproachNode::execute, this, goal_handle)).detach();
}

void ApproachNode::execute(
    const std::shared_ptr<GoalHandleApproach> goal_handle) {

  auto feedback = std::make_shared<ApproachAction::Feedback>();
  auto result = std::make_shared<ApproachAction::Result>();
  const auto goal = goal_handle->get_goal();
  target_standoff_distance_ = goal->goal_distance;
  RCLCPP_INFO(this->get_logger(),
              "Set target_standoff_distance from action goal: %.3f",
              target_standoff_distance_);

  if (!algorithm_start_flag) {
    algorithm_start_flag = true;
    startAlgorithm();
    RCLCPP_INFO(this->get_logger(), "Algorithm started");
  }

  rclcpp::Rate loop_rate(10); // 10Hz 제어 루프 속도 설정

  while (rclcpp::ok()) {
    if (goal_handle->is_canceling()) {
      RCLCPP_INFO(this->get_logger(), "Approach goal canceled");
      goal_handle->abort(result);
      stopAlgorithm();
      return;
    }

    if (control_success) {
      RCLCPP_INFO(this->get_logger(), "Approach goal succeeded");
      result->success = true;
      result->success_message = "Approach goal succeeded";
      goal_handle->succeed(result);
      stopAlgorithm();
      return;
    }

    if (control_failure) {
      RCLCPP_INFO(this->get_logger(), "Approach goal failed");
      result->success = false;
      result->success_message = failure_message;
      goal_handle->abort(result);
      stopAlgorithm();
      return;
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Executing approach");

    feedback->x_error = se2_error.x;
    feedback->y_error = se2_error.y;
    feedback->theta_error = se2_error.degree_theta;
    goal_handle->publish_feedback(feedback);

    loop_rate.sleep();
  }
}

/*
콜백함수(메인 함수)
*/
void ApproachNode::pointCloudCallback(
    const PointCloudMsg::ConstSharedPtr &pointcloud_msg) {

  if (!algorithm_start_flag) {
    return;
  }

  const auto t_cb = Clock::now();
  const double sensor_latency_ms =
      (this->now() - pointcloud_msg->header.stamp).seconds() * 1000.0;

  cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
  kdtree.reset(new pcl::search::KdTree<pcl::PointXYZ>);

  /*
  포인트 클라우드 전처리
  (전체 클라우드 -> 다운샘플링 -> 이상치 제거)
  */
  pcl::fromROSMsg(*pointcloud_msg, *cloud);
  if (cloud->empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Empty point cloud received.");
    return;
  }

  const auto t0 = Clock::now();
  roi_filter_->voxel_downsampling(cloud);
  const double t_voxel = elapsed_ms(t0);

  const auto t1 = Clock::now();
  roi_filter_->remove_outliers(cloud);
  const double t_outlier = elapsed_ms(t1);

  geometry_msgs::msg::TransformStamped tf;
  if (getTransform(target_frame_, pointcloud_msg->header.frame_id, tf,
                   pointcloud_msg->header.stamp)) {
    RCLCPP_DEBUG(this->get_logger(), "TF received");
  } else {
    RCLCPP_WARN(
        this->get_logger(),
        "Failed to get TF. Skipping point cloud transformation and return");
    return;
  }

  const auto t2 = Clock::now();
  roi_filter_->remove_ground(cloud, tf.transform);
  applySpatialRoiBounds(cloud, -std::numeric_limits<float>::infinity(),
                        roi_x_max_, roi_y_abs_max_, roi_z_max_);
  const double t_ground = elapsed_ms(t2);

  /*
  LiDAR 융합: lidarCallback이 base 프레임으로 변환·전처리해서 캐싱해둔 클라우드를
  여기서 합친다. 카메라 FOV 밖에서도 LiDAR가 객체 윤곽을 잡아주는 보강.
  PTP로 stamp 동기 되어있다는 가정 하에 stamp 기준 staleness 체크 (릴레이 끊김
  방지). 너무 오래된 캐시는 skip.
  */
  {
    std::lock_guard<std::mutex> lock(lidar_cloud_mutex_);
    if (latest_lidar_cloud_ && !latest_lidar_cloud_->empty()) {
      rclcpp::Time camera_stamp(pointcloud_msg->header.stamp);
      const double age =
          std::abs((camera_stamp - latest_lidar_stamp_).seconds());
      if (age < static_cast<double>(lidar_max_age_sec_)) {
        *cloud += *latest_lidar_cloud_;
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Stale lidar cloud (%.2fs > %.2fs), skip concat",
                             age, lidar_max_age_sec_);
      }
    }
  }

  /*
  카메라는 가까이 접근했을 때 roi_x_min 에 의해 객체가 통째로 잘려나가는
  문제를 피하기 위해 x_min 없이 좌우/거리/높이 ROI만 적용한다. LiDAR cloud 는
  lidarCallback 에서 이미 applySpatialRoi 통과한 상태로 합쳐져 있으므로 로봇
  본체 자기반사 / 360° 잡음은 그쪽에서 걸러진다.
  */
  if (cloud->empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "No points left after ground removal & concat.");
    return;
  }

  // 디버깅용 포인트클라우드
  sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
  pcl::toROSMsg(*cloud, filtered_cloud_msg);
  filtered_cloud_msg.header.stamp = this->now();
  filtered_cloud_msg.header.frame_id = target_frame_;
  debugging_pointcloud_publisher_->publish(filtered_cloud_msg);
  auto debug_roi_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*cloud);

  const auto t3 = Clock::now();
  geometry_msgs::msg::TransformStamped base_to_odom_tf;
  if (!getTransform(odom_frame_, target_frame_, base_to_odom_tf,
                    pointcloud_msg->header.stamp)) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "Failed to get odom<-base TF for target lock.");
    geometry_msgs::msg::Twist stop;
    cmd_vel_publisher_->publish(stop);
    return;
  }

  TargetSelectorResult selection;
  const bool target_valid =
      target_selector_->select(cloud, transformToAffine2D(base_to_odom_tf),
                               selection);
  const double t_cluster = elapsed_ms(t3);

  if (!target_valid) {
    saveDebugOverlay(debug_roi_cloud, &selection);
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "Target selector waiting: state=%s lost=%d reason=%s",
                         selection.state.c_str(), selection.lost_count,
                         selection.reason.c_str());
    geometry_msgs::msg::Twist stop;
    cmd_vel_publisher_->publish(stop);
    return;
  }

  if (selection.selected_cloud && !selection.selected_cloud->empty()) {
    cloud = selection.selected_cloud;
    publish3Dpointcloud(cloud);

    const auto t4 = Clock::now();
    auto final_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*cloud);
    roi_filter_->projection_filter(final_cloud);
    const double t_proj = elapsed_ms(t4);
    (void)t_proj;
  }

  const auto t5 = Clock::now();
  obb = selection.obb;
  const double t_obb = elapsed_ms(t5);
  publish2DOBB(obb.center, obb.axis1, obb.axis2, obb.length1, obb.length2);

  TargetEdge target_edge = selection.edge;
  if (selection.newly_locked || initial_dist_ <= 1e-4F) {
    initial_dist_ =
        std::abs(target_edge.target_center.dot(target_edge.normal_axis) -
                 target_standoff_distance_);
    RCLCPP_INFO(this->get_logger(),
                "Target locked (%s): cluster=%zu hit_base=(%.3f, %.3f) "
                "hit_odom=(%.3f, %.3f) yaw=%.3f len=%.3f area=%.3f "
                "fill=%.3f target_support=%.3f init_dist=%.3f",
                selection.is_round ? "ROUND" : "L-SHAPE",
                selection.cluster_id, selection.hit_base.x(),
                selection.hit_base.y(), selection.hit_odom.x(),
                selection.hit_odom.y(),
                std::atan2(target_edge.normal_axis.y(),
                           target_edge.normal_axis.x()),
                target_edge.target_length, selection.area,
                selection.metrics.fill_ratio,
                selection.metrics.target_edge_support_ratio, initial_dist_);
  }

  publishTargetEdge(target_edge);
  saveDebugOverlay(debug_roi_cloud, &selection);

  // error estimator => SE(2) error 측정
  SE2Error candidate =
      error_estimator_->estimate_error(target_edge, target_standoff_distance_);

  /*
  스파이크 필터: 클러스터 오탐/OBB flipping으로 인한 단일 프레임 이상치를 제거.
  한 사이클당 y 또는 theta 변화가 임계치를 넘으면 직전 valid 에러를 재사용.
  연속으로 N회 이상 튀면 씬이 실제로 바뀐 것으로 보고 새 값을 받아들인다.
  */
  if (!se2_error_initialized_) {
    se2_error = candidate;
    se2_error_initialized_ = true;
    consecutive_outliers_ = 0;
  } else {
    const float dy = std::abs(candidate.y - se2_error.y);
    const float dth = std::abs(candidate.degree_theta - se2_error.degree_theta);
    const bool is_spike = (dy > spike_dy_max_) || (dth > spike_dtheta_max_);

    if (is_spike) {
      consecutive_outliers_++;
      ++spike_count_;
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "Spike rejected (#%zu total): dy=%.3f dth=%.3f  — keeping prev error",
          spike_count_, dy, dth);
      if (consecutive_outliers_ >= max_consecutive_outliers_) {
        // 연속 이상치 → 실제로 씬이 변한 것으로 판단, 새 값 채택 (EMA 리셋)
        RCLCPP_WARN(this->get_logger(),
                    "Consecutive spikes exceeded; accepting new error.");
        se2_error = candidate;
        consecutive_outliers_ = 0;
      }
      // else: se2_error 유지 (직전 valid 값)
    } else {
      se2_error = candidate;
      consecutive_outliers_ = 0;
    }
  }

  if (debug_enabled_) {
    std::ostringstream line;
    line << "[MEAS-STAGE] #" << measure_cycle_
         << "  voxel=" << std::fixed << std::setprecision(1) << t_voxel
         << "ms  outlier=" << t_outlier << "ms  ground=" << t_ground
         << "ms  cluster=" << t_cluster << "ms  proj+slice=" << t_proj
         << "ms  obb=" << t_obb << "ms  |  sensor_lat="
         << sensor_latency_ms << "ms";
    writeMeasureLog(line.str());
  }

  /*
  pid controller => SE(2) -> 로봇의 모델 방정식 -> v_x, w_z 측정
  */
  rclcpp::Time current_time = this->now();
  if (previous_time_.nanoseconds() == 0) {
    previous_time_ = current_time;
  }
  float dt = (current_time - previous_time_).seconds();
  previous_time_ = current_time;

  /*
  상태머신: APPROACH → ALIGN_THETA → APPROACH(1회 재보정) → DWELL → DONE.
  - APPROACH: 종/횡/각도 PID 동시 제어 + 시작거리 기반 동적 감속 ramp.
  - ALIGN_THETA: v_x=0, w_z만 kp_theta * e_theta 로 제자리 회전 정렬.
  - theta 정렬 후 x 오차가 다시 커졌으면 한 번 더 APPROACH 로 보정.
  - DWELL: 정지 publish 한 채 dwell_duration_sec 만큼 안정화.
  - DONE: control_success 세팅.
  y 오차는 의도적으로 무시 (홀로노믹 아님).
  */
  const float abs_ex = std::abs(se2_error.x);
  const float abs_eth = std::abs(se2_error.degree_theta);
  geometry_msgs::msg::Twist cmd_vel;

  switch (state_) {
    case ApproachState::IDLE:
      // 알고리즘 시작 시 APPROACH 로 전이되므로 정상 흐름에선 도달 불가.
      return;

    case ApproachState::APPROACH: {
      // 시작 거리(initial_dist_) 기반으로 감속 구간 길이를 한 번 산정.
      // 가까이서 시작하면 짧은 ramp, 멀리서 시작해도 최대 decel_dist_max 까지만.
      const float effective_decel = std::clamp(
          initial_dist_ * decel_ratio_, decel_dist_min_, decel_dist_max_);
      const float v_scale = (effective_decel > 0.0F)
                                ? std::clamp(abs_ex / effective_decel, 0.0F, 1.0F)
                                : 1.0F;
      cmd_vel = pid_controller_->compute_control(se2_error, dt, v_scale);

      if (abs_ex < tol_x_) {
        if (abs_eth < tol_theta_) {
          state_ = ApproachState::DWELL;
          dwell_start_time_ = current_time;
          RCLCPP_INFO(this->get_logger(),
                      "APPROACH → DWELL (ex=%.3f, eth=%.3f)", abs_ex, abs_eth);
        } else {
          state_ = ApproachState::ALIGN_THETA;
          align_start_time_ = current_time;
          pid_controller_->reset();
          RCLCPP_INFO(this->get_logger(),
                      "APPROACH → ALIGN_THETA (ex=%.3f, eth=%.3f)",
                      abs_ex, abs_eth);
        }
      }
      break;
    }

    case ApproachState::ALIGN_THETA: {
      cmd_vel = pid_controller_->compute_align_only(se2_error.degree_theta);

      if (abs_eth < tol_theta_) {
        if (!post_align_approach_done_ && abs_ex >= tol_x_) {
          post_align_approach_done_ = true;
          state_ = ApproachState::APPROACH;
          pid_controller_->reset();
          RCLCPP_INFO(this->get_logger(),
                      "ALIGN_THETA → APPROACH recheck (ex=%.3f, eth=%.3f)",
                      abs_ex, abs_eth);
        } else {
          state_ = ApproachState::DWELL;
          dwell_start_time_ = current_time;
          RCLCPP_INFO(this->get_logger(),
                      "ALIGN_THETA → DWELL (ex=%.3f, eth=%.3f)",
                      abs_ex, abs_eth);
        }
      } else if ((current_time - align_start_time_).seconds() >
                 align_timeout_sec_) {
        if (!post_align_approach_done_ && abs_ex >= tol_x_) {
          post_align_approach_done_ = true;
          state_ = ApproachState::APPROACH;
          pid_controller_->reset();
          RCLCPP_WARN(this->get_logger(),
                      "ALIGN_THETA timeout (%.1fs) → APPROACH recheck "
                      "(ex=%.3f, eth=%.3f)",
                      align_timeout_sec_, abs_ex, abs_eth);
        } else {
          RCLCPP_WARN(this->get_logger(),
                      "ALIGN_THETA timeout (%.1fs) → DWELL anyway "
                      "(ex=%.3f, eth=%.3f)",
                      align_timeout_sec_, abs_ex, abs_eth);
          state_ = ApproachState::DWELL;
          dwell_start_time_ = current_time;
        }
      }
      break;
    }

    case ApproachState::DWELL: {
      cmd_vel.linear.x = 0.0;
      cmd_vel.angular.z = 0.0;
      if ((current_time - dwell_start_time_).seconds() >
          dwell_duration_sec_) {
        state_ = ApproachState::DONE;
      }
      break;
    }

    case ApproachState::DONE: {
      cmd_vel.linear.x = 0.0;
      cmd_vel.angular.z = 0.0;
      control_success = true;
      break;
    }
  }

  if (selection.lost_count > 0) {
    const float lost_ratio = static_cast<float>(selection.lost_count) /
                             static_cast<float>(target_selector_params_.max_lost_frames);
    const float lost_decel_factor = std::max(0.0F, 1.0F - lost_ratio);
    cmd_vel.linear.x *= lost_decel_factor;
    cmd_vel.angular.z *= lost_decel_factor;
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                         "Target lost! Decelerating: lost_count=%d/%d, factor=%.2f",
                         selection.lost_count, target_selector_params_.max_lost_frames,
                         lost_decel_factor);
  }

  cmd_vel_publisher_->publish(cmd_vel);

  if (debug_enabled_) {
    const double proc_ms = elapsed_ms(t_cb);
    const double total_delay_ms = sensor_latency_ms + proc_ms;
    std::ostringstream line;
    line << "[MEAS] #" << measure_cycle_++
         << " | sensor=" << std::fixed << std::setprecision(1)
         << sensor_latency_ms << "ms | proc=" << proc_ms
         << "ms | total=" << total_delay_ms << "ms | dt=" << dt * 1000.0
         << "ms | ex=" << std::setprecision(4) << se2_error.x
         << " | ey=" << se2_error.y << " | eth=" << se2_error.degree_theta
         << " | vx=" << cmd_vel.linear.x << " | wz=" << cmd_vel.angular.z;
    writeMeasureLog(line.str());
  }
}

void ApproachNode::lidarCallback(
    const PointCloudMsg::ConstSharedPtr &lidar_msg) {
  if (!algorithm_start_flag) {
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*lidar_msg, *lidar_cloud);
  if (lidar_cloud->empty()) {
    return;
  }

  // 카메라와 동일한 전처리 단계 통과 (downsample → outlier → ground removal)
  roi_filter_->voxel_downsampling(lidar_cloud);
  roi_filter_->remove_outliers(lidar_cloud);

  // base→lidar 는 본래 동적(torso 움직임)이지만, 이 approach 노드가 도는 동안
  // 은 torso 고정이라 사실상 static. TimePointZero(=latest)로 lookup 해서 stamp
  // 기준 lookup 의 릴레이 지연 실패를 회피.
  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_.lookupTransform(target_frame_, lidar_msg->header.frame_id,
                                    tf2::TimePointZero,
                                    tf2::durationFromSec(0.2));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "lidar TF lookup failed: %s", ex.what());
    return;
  }
  roi_filter_->remove_ground(lidar_cloud, tf.transform);

  // 공간 ROI는 LiDAR에만 적용. LiDAR는 360°/넓은 영역을 보고 로봇 본체 자기반사도
  // 잡히므로 전방 박스로 잘라낸다. 카메라는 FOV 자체가 좁고, 가까이 갔을 때
  // roi_x_min 에 의해 객체가 통째로 잘려나가는 문제 때문에 ROI 미적용.
  applySpatialRoi(lidar_cloud);

  // base 프레임 + ground 제거 + ROI 적용된 상태로 캐싱. 카메라 콜백에서 concat.
  std::lock_guard<std::mutex> lock(lidar_cloud_mutex_);
  latest_lidar_cloud_ = lidar_cloud;
  latest_lidar_stamp_ = lidar_msg->header.stamp;
}

void ApproachNode::cameraInfoCallback(const CameraInfoMsg::SharedPtr msg) {
  if (!algorithm_start_flag) {
    return;
  }

  if (received_camera_info_) {
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Received CameraInfo message");
  roi_filter_->setCameraInfo(msg->k[0], // fx
                             msg->k[4], // fy
                             msg->k[2], // cx
                             msg->k[5]  // cy
  );
  received_camera_info_ = true;
}

bool ApproachNode::getTransform(const std::string &target_frame,
                                const std::string &source_frame,
                                geometry_msgs::msg::TransformStamped &transform,
                                const rclcpp::Time &stamp) {
  try {
    /*
    Real World -> stamp
    Bag files or Simulation -> rclcpp::Time(0, 0, RCL_ROS_TIME)
    */
    transform = tf_buffer_.lookupTransform(target_frame, source_frame,
                                           stamp, // Here
                                           rclcpp::Duration::from_seconds(0.1));
    return true;
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(this->get_logger(), "Failed to get TF from %s to %s: %s, %f",
                source_frame.c_str(), target_frame.c_str(), ex.what(),
                stamp.seconds());
    return false;
  }
}

void ApproachNode::publish3Dpointcloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
  pcl::toROSMsg(*cloud, filtered_cloud_msg);
  filtered_cloud_msg.header.stamp = this->now();
  filtered_cloud_msg.header.frame_id =
      target_frame_; // Points are now in target_frame_
  filtered_pointcloud_publisher_->publish(filtered_cloud_msg);
}

void ApproachNode::publish2DOBB(const Eigen::Vector2f &center,
                                const Eigen::Vector2f &axis1,
                                const Eigen::Vector2f &axis2,
                                const float length1, const float length2) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = target_frame_;
  marker.header.stamp = this->now();
  marker.ns = "obb";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;

  marker.scale.x = 0.05; // Line width

  marker.color.r = 0.0f;
  marker.color.g = 1.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0f;

  Eigen::Vector2f p1 =
      center + (length1 / 2.0f) * axis1 + (length2 / 2.0f) * axis2;
  Eigen::Vector2f p2 =
      center - (length1 / 2.0f) * axis1 + (length2 / 2.0f) * axis2;
  Eigen::Vector2f p3 =
      center - (length1 / 2.0f) * axis1 - (length2 / 2.0f) * axis2;
  Eigen::Vector2f p4 =
      center + (length1 / 2.0f) * axis1 - (length2 / 2.0f) * axis2;

  geometry_msgs::msg::Point pt;
  pt.z = ground_height_; // Place it on the ground height

  pt.x = p1.x();
  pt.y = p1.y();
  marker.points.push_back(pt);
  pt.x = p2.x();
  pt.y = p2.y();
  marker.points.push_back(pt);
  pt.x = p3.x();
  pt.y = p3.y();
  marker.points.push_back(pt);
  pt.x = p4.x();
  pt.y = p4.y();
  marker.points.push_back(pt);
  pt.x = p1.x();
  pt.y = p1.y();
  marker.points.push_back(pt); // Close the loop

  obb_publisher_->publish(marker);
}

void ApproachNode::publishTargetEdge(const TargetEdge &target_edge) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = target_frame_;
  marker.header.stamp = this->now();
  marker.ns = "target_edge";
  marker.id = 0;

  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.scale.x = 0.05;
  marker.color.r = 1.0f;
  marker.color.g = 0.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0f;
  geometry_msgs::msg::Point pt;
  pt.z = ground_height_;
  Eigen::Vector2f p1 =
      target_edge.target_center +
      (target_edge.target_length / 2.0f) * target_edge.target_axis;
  Eigen::Vector2f p2 =
      target_edge.target_center -
      (target_edge.target_length / 2.0f) * target_edge.target_axis;
  pt.x = p1.x();
  pt.y = p1.y();
  marker.points.push_back(pt);
  pt.x = p2.x();
  pt.y = p2.y();
  marker.points.push_back(pt);
  Eigen::Vector2f p3_normal_end =
      target_edge.target_center -
      target_edge.normal_axis * target_standoff_distance_;
  pt.x = target_edge.target_center.x();
  pt.y = target_edge.target_center.y();
  marker.points.push_back(pt);

  pt.x = p3_normal_end.x();
  pt.y = p3_normal_end.y();
  marker.points.push_back(pt);
  target_edge_publisher_->publish(marker);
}

void ApproachNode::saveDebugOverlay(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &roi_cloud,
    const TargetSelectorResult *selection) {
  if (!debug_enabled_ || !roi_cloud || roi_cloud->empty()) {
    return;
  }

  const std::string stage = "debug_overlay";
  const auto now = Clock::now();
  const auto last_save = last_debug_save_time_by_stage_.find(stage);
  if (last_save != last_debug_save_time_by_stage_.end() &&
      std::chrono::duration<double>(now - last_save->second).count() <
          debug_save_period_sec_) {
    return;
  }
  last_debug_save_time_by_stage_[stage] = now;

  pcl::PointCloud<pcl::PointXYZRGB> overlay;
  overlay.points.reserve(roi_cloud->points.size() + 600);

  std::vector<bool> assigned(roi_cloud->points.size(), false);
  auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
  tree->setInputCloud(roi_cloud);

  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(max_cluster_size_);
  ec.setSearchMethod(tree);
  ec.setInputCloud(roi_cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  ec.extract(cluster_indices);

  for (std::size_t cluster_idx = 0; cluster_idx < cluster_indices.size();
       ++cluster_idx) {
    const DebugColor color = debugPaletteColor(cluster_idx);
    for (const int point_idx : cluster_indices[cluster_idx].indices) {
      if (point_idx < 0 ||
          static_cast<std::size_t>(point_idx) >= roi_cloud->points.size()) {
        continue;
      }
      assigned[static_cast<std::size_t>(point_idx)] = true;
      const auto &src = roi_cloud->points[point_idx];
      pcl::PointXYZRGB pt;
      pt.x = src.x;
      pt.y = src.y;
      pt.z = src.z;
      pt.r = color.r;
      pt.g = color.g;
      pt.b = color.b;
      overlay.points.push_back(pt);
    }
  }

  for (std::size_t i = 0; i < roi_cloud->points.size(); ++i) {
    if (assigned[i]) {
      continue;
    }
    const auto &src = roi_cloud->points[i];
    pcl::PointXYZRGB pt;
    pt.x = src.x;
    pt.y = src.y;
    pt.z = src.z;
    pt.r = 90;
    pt.g = 90;
    pt.b = 90;
    overlay.points.push_back(pt);
  }

  const float marker_z = ground_height_ + 0.06F;
  addDebugAxes(overlay, marker_z + 0.10F, 1.2F);

  if (selection && selection->valid) {
    addDebugObb(overlay, selection->obb, marker_z, {0, 255, 0});

    const TargetEdge &edge = selection->edge;
    const Eigen::Vector2f edge_a =
        edge.target_center + edge.target_axis * (edge.target_length * 0.5F);
    const Eigen::Vector2f edge_b =
        edge.target_center - edge.target_axis * (edge.target_length * 0.5F);
    addDebugSegment(overlay, edge_a, edge_b, marker_z + 0.04F,
                    {0, 220, 255}, 120);
    addDebugCross(overlay, selection->anchor_base, marker_z + 0.08F,
                  {255, 0, 0}, 0.045F);
    addDebugCross(overlay, selection->hit_base, marker_z + 0.10F,
                  {255, 255, 255}, 0.04F);
    addDebugSegment(overlay, selection->anchor_base, selection->hit_base,
                    marker_z + 0.09F, {255, 0, 0}, 80);

    const Eigen::Vector2f standoff_point =
        edge.target_center - edge.normal_axis * target_standoff_distance_;
    addDebugSegment(overlay, edge.target_center, standoff_point,
                    marker_z + 0.06F, {255, 128, 0}, 80);
    addDebugCross(overlay, standoff_point, marker_z + 0.09F,
                  {255, 128, 0}, 0.035F);
  }

  if (overlay.points.empty()) {
    return;
  }
  overlay.width = static_cast<std::uint32_t>(overlay.points.size());
  overlay.height = 1;
  overlay.is_dense = false;

  std::ostringstream filename;
  filename << currentTimeForFilename() << "_" << std::setw(6)
           << std::setfill('0') << debug_cloud_index_ << "_" << stage
           << ".pcd";

  const auto &output_dir =
      debug_action_output_dir_.empty() ? debug_output_dir_ : debug_action_output_dir_;
  const auto file_path = output_dir / filename.str();
  const int result = pcl::io::savePCDFileBinary(file_path.string(), overlay);
  if (result == 0) {
    ++debug_cloud_index_;
    RCLCPP_DEBUG(this->get_logger(), "Saved debug overlay: %s",
                 file_path.c_str());
  } else {
    RCLCPP_WARN(this->get_logger(), "Failed to save debug overlay: %s",
                file_path.c_str());
  }
}

void ApproachNode::openDebugActionDirectory() {
  if (!debug_enabled_) {
    return;
  }

  debug_action_output_dir_ =
      debug_output_dir_ / ("action_" + currentTimeForFilename());
  std::filesystem::create_directories(debug_action_output_dir_);
  debug_cloud_index_ = 0;
  last_debug_save_time_by_stage_.clear();

  RCLCPP_INFO(this->get_logger(), "Debug action files will be saved to %s",
              debug_action_output_dir_.c_str());
}

void ApproachNode::openMeasureLog() {
  if (!debug_enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(measure_log_mutex_);
  if (measure_log_file_.is_open()) {
    measure_log_file_.close();
  }

  const auto &output_dir =
      debug_action_output_dir_.empty() ? debug_output_dir_ : debug_action_output_dir_;
  measure_log_path_ = output_dir / ("measure_" + currentTimeForFilename() + ".txt");
  measure_log_file_.open(measure_log_path_, std::ios::out | std::ios::trunc);
  if (!measure_log_file_.is_open()) {
    RCLCPP_WARN(this->get_logger(), "Failed to open measure log: %s",
                measure_log_path_.c_str());
    return;
  }

  measure_log_file_
      << "[MEAS] cycle | sensor_lat_ms | proc_ms | total_delay_ms | dt_ms | "
         "e_x | e_y | e_theta | v_x | w_z\n";
  measure_log_file_.flush();
  RCLCPP_INFO(this->get_logger(), "Measure log will be saved to %s",
              measure_log_path_.c_str());
}

void ApproachNode::closeMeasureLog() {
  std::lock_guard<std::mutex> lock(measure_log_mutex_);
  if (measure_log_file_.is_open()) {
    measure_log_file_.flush();
    measure_log_file_.close();
  }
}

void ApproachNode::writeMeasureLog(const std::string &line) {
  if (!debug_enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(measure_log_mutex_);
  if (!measure_log_file_.is_open()) {
    return;
  }
  measure_log_file_ << line << '\n';
}

void ApproachNode::stopAlgorithm() {
  // trail은 이번 approach가 끝난 시점에 latched로 한 번 발행해서
  // RetreatNode가 후진 액션 시 사용할 수 있도록 한다.
  publishTrail();

  point_cloud_subscriber_.reset();
  lidar_subscriber_.reset();

  {
    std::lock_guard<std::mutex> lock(lidar_cloud_mutex_);
    latest_lidar_cloud_.reset();
  }

  start_time_flag = false;
  control_success = false;
  control_failure = false;
  algorithm_start_flag = false;
  closeMeasureLog();
}

void ApproachNode::startAlgorithm() {
  measure_cycle_ = 0;
  openDebugActionDirectory();
  openMeasureLog();

  {
    std::lock_guard<std::mutex> lock(trail_mutex_);
    trail_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(lidar_cloud_mutex_);
    latest_lidar_cloud_.reset();
  }

  point_cloud_subscriber_ = this->create_subscription<PointCloudMsg>(
      pointcloud_topic_name_, qos_best_effort_,
      std::bind(&ApproachNode::pointCloudCallback, this,
                std::placeholders::_1));

  lidar_subscriber_ = this->create_subscription<PointCloudMsg>(
      lidar_topic_name_, qos_best_effort_,
      std::bind(&ApproachNode::lidarCallback, this, std::placeholders::_1));

  start_time_flag = false;
  control_success = false;
  control_failure = false;

  // 상태머신/anchor/PID 초기화
  state_ = ApproachState::APPROACH;
  post_align_approach_done_ = false;
  aim_anchor_captured_ = false;
  initial_dist_ = 0.0F;
  se2_error_initialized_ = false;
  consecutive_outliers_ = 0;
  if (pid_controller_) pid_controller_->reset();
  if (target_selector_) target_selector_->reset();

  algorithm_start_flag = true;
}

void ApproachNode::recordTrailPose() {
  if (!algorithm_start_flag) {
    return;
  }

  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_.lookupTransform(odom_frame_, target_frame_,
                                    tf2::TimePointZero,
                                    tf2::durationFromSec(0.05));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_DEBUG(this->get_logger(), "trail TF lookup failed: %s", ex.what());
    return;
  }

  geometry_msgs::msg::PoseStamped pose;
  pose.header = tf.header;
  pose.pose.position.x = tf.transform.translation.x;
  pose.pose.position.y = tf.transform.translation.y;
  pose.pose.position.z = tf.transform.translation.z;
  pose.pose.orientation = tf.transform.rotation;

  std::lock_guard<std::mutex> lock(trail_mutex_);
  if (trail_.empty()) {
    trail_.push_back(pose);
    return;
  }
  const auto &last = trail_.back().pose;
  const float dx = static_cast<float>(pose.pose.position.x - last.position.x);
  const float dy = static_cast<float>(pose.pose.position.y - last.position.y);
  const float dist = std::sqrt(dx * dx + dy * dy);
  const double yaw_now = tf2::getYaw(pose.pose.orientation);
  const double yaw_last = tf2::getYaw(last.orientation);
  double dyaw = yaw_now - yaw_last;
  while (dyaw > M_PI) dyaw -= 2.0 * M_PI;
  while (dyaw < -M_PI) dyaw += 2.0 * M_PI;

  if (dist >= trail_min_dist_ ||
      std::abs(dyaw) >= static_cast<double>(trail_min_yaw_)) {
    trail_.push_back(pose);
  }
}

void ApproachNode::publishTrail() {
  nav_msgs::msg::Path path;
  {
    std::lock_guard<std::mutex> lock(trail_mutex_);
    path.poses.assign(trail_.begin(), trail_.end());
  }
  path.header.stamp = this->now();
  path.header.frame_id = odom_frame_;
  trail_publisher_->publish(path);
  RCLCPP_INFO(this->get_logger(), "Published approach trail: %zu poses",
              path.poses.size());
}

bool ApproachNode::captureAimAnchor(const TargetEdge &target_edge,
                                     const rclcpp::Time &stamp) {
  /*
  base 프레임에서 로봇 정면 ray (origin=0, dir=+x)와 target edge 직선의 교차점.
    line: P(t) = c + t * a   (c=target_center, a=target_axis)
    ray:  Q(s) = (s, 0)
    교차: c.y + t * a.y = 0  →  t = -c.y / a.y
  axis가 robot x 와 거의 평행이면 (|a.y| ~ 0) 교차점 정의 안되므로 target_center
  로 fallback.
  */
  const Eigen::Vector2f &c = target_edge.target_center;
  const Eigen::Vector2f &a = target_edge.target_axis;
  const float half_L = target_edge.target_length * 0.5F;

  Eigen::Vector2f aim_base;
  const float eps = 1e-3F;
  if (std::abs(a.y()) < eps) {
    aim_base = c;
  } else {
    float t = -c.y() / a.y();
    t = std::clamp(t, -half_L, half_L);
    aim_base = c + t * a;
  }

  // base → odom 변환
  geometry_msgs::msg::TransformStamped tf;
  if (!getTransform(odom_frame_, target_frame_, tf, stamp)) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "captureAimAnchor: odom←base TF lookup failed");
    return false;
  }

  const double yaw = tf2::getYaw(tf.transform.rotation);
  const float cy_v = static_cast<float>(std::cos(yaw));
  const float sy_v = static_cast<float>(std::sin(yaw));
  const auto &tr = tf.transform.translation;
  aim_anchor_odom_.x() =
      cy_v * aim_base.x() - sy_v * aim_base.y() + static_cast<float>(tr.x);
  aim_anchor_odom_.y() =
      sy_v * aim_base.x() + cy_v * aim_base.y() + static_cast<float>(tr.y);

  // 감속 ramp 산정용 초기 종방향 거리: anchor 기준 normal-projection - standoff.
  const Eigen::Vector2f &n = target_edge.normal_axis;
  initial_dist_ = std::abs(aim_base.dot(n) - target_standoff_distance_);

  aim_anchor_captured_ = true;
  RCLCPP_INFO(this->get_logger(),
              "Aim anchor captured: base=(%.3f, %.3f) odom=(%.3f, %.3f) "
              "init_dist=%.3f",
              aim_base.x(), aim_base.y(), aim_anchor_odom_.x(),
              aim_anchor_odom_.y(), initial_dist_);
  return true;
}

bool ApproachNode::anchorInBase(const rclcpp::Time &stamp,
                                 Eigen::Vector2f &out_xy) {
  geometry_msgs::msg::TransformStamped tf;
  if (!getTransform(target_frame_, odom_frame_, tf, stamp)) {
    RCLCPP_DEBUG(this->get_logger(),
                 "anchorInBase: base←odom TF lookup failed");
    return false;
  }
  const double yaw = tf2::getYaw(tf.transform.rotation);
  const float cy_v = static_cast<float>(std::cos(yaw));
  const float sy_v = static_cast<float>(std::sin(yaw));
  const auto &tr = tf.transform.translation;
  out_xy.x() = cy_v * aim_anchor_odom_.x() - sy_v * aim_anchor_odom_.y() +
               static_cast<float>(tr.x);
  out_xy.y() = sy_v * aim_anchor_odom_.x() + cy_v * aim_anchor_odom_.y() +
               static_cast<float>(tr.y);
  return true;
}

bool ApproachNode::projectAnchorOnEdge(const TargetEdge &target_edge,
                                        const rclcpp::Time &stamp,
                                        Eigen::Vector2f &out_center) {
  Eigen::Vector2f anchor_base;
  if (!anchorInBase(stamp, anchor_base)) {
    return false;
  }
  // 현재 edge 직선에 수직 투영 (부호 flip에 무관)
  const Eigen::Vector2f &c = target_edge.target_center;
  const Eigen::Vector2f a_unit = target_edge.target_axis.normalized();
  float s = (anchor_base - c).dot(a_unit);
  const float half_L = target_edge.target_length * 0.5F;
  s = std::clamp(s, -half_L, half_L);
  out_center = c + s * a_unit;
  return true;
}

void ApproachNode::applySpatialRoi(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  applySpatialRoiBounds(cloud, roi_x_min_, roi_x_max_, roi_y_abs_max_,
                        roi_z_max_);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ApproachNode>());
  rclcpp::shutdown();
  return 0;
}
