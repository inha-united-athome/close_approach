#include "close_approach/main.hpp"

#include <filesystem>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>

namespace {

bool hasDebugArg(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "debug" || std::string(argv[i]) == "--debug")
      return true;
  }
  return false;
}

bool hasMeasureArg(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "measure" || std::string(argv[i]) == "--measure")
      return true;
  }
  return false;
}

std::vector<char *> filterKnownArgs(int argc, char **argv) {
  std::vector<char *> filtered_args;
  filtered_args.reserve(static_cast<std::size_t>(argc));
  if (argc > 0)
    filtered_args.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "debug" || arg == "--debug" || arg == "measure" || arg == "--measure")
      continue;
    filtered_args.push_back(argv[i]);
  }
  return filtered_args;
}

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

double elapsed_ms(const Clock::time_point &from) {
  return std::chrono::duration_cast<Ms>(Clock::now() - from).count();
}

} // namespace

ApproachNode::ApproachNode(bool debug_enabled, bool measure_enabled)
    : Node("approach_node"),
      qos_best_effort_(rclcpp::QoS(rclcpp::KeepLast(10)).best_effort()),
      qos_reliable_(rclcpp::QoS(rclcpp::KeepLast(10)).reliable()),
      tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_),
      debug_enabled_(debug_enabled), measure_enabled_(measure_enabled) {

  if (measure_enabled_) {
    RCLCPP_INFO(this->get_logger(),
                "[MEAS] measure mode ON  -- header format:");
    RCLCPP_INFO(this->get_logger(),
                "[MEAS] cycle | sensor_lat_ms | proc_ms | total_delay_ms | "
                "dt_ms | e_x | e_y | e_theta | v_x | w_z");
  }

  if (debug_enabled_) {
    debug_output_dir_ = std::filesystem::current_path() / "debug" / "main";
    std::filesystem::create_directories(debug_output_dir_);
    RCLCPP_INFO(this->get_logger(), "Debug point clouds will be saved to %s",
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

  /*
  파라미터 선언
  */
  this->declare_parameter<std::string>(
      "pointcloud_topic_name", "/camera/camera_head/depth/color/points");
  this->declare_parameter<std::string>("lidar_topic_name", "/livox/lidar");
  this->declare_parameter<std::string>("info_topic_name",
                                       "/camera/camera_head/color/camera_info");
  this->declare_parameter<std::string>("target_frame", "base");
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
  this->declare_parameter<float>("base_to_rotationcore", 0.2F);
  this->declare_parameter<float>("target_standoff_distance", 0.5F); // 로봇과 목표 사이의 간격
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
  this->get_parameter("base_to_rotationcore", base_to_rotationcore_);
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
                                 ki_theta_, kd_x_, kd_y_, kd_theta_,
                                 base_to_rotationcore_);
  pid_controller_->setLimits(max_v_, max_w_);
}
/*
Ros2 Action 관련 함수들
*/
rclcpp_action::GoalResponse
ApproachNode::handle_goal(const rclcpp_action::GoalUUID &uuid,
                          std::shared_ptr<const ApproachAction::Goal> goal) {
  (void)uuid;
  (void)goal;
  RCLCPP_INFO(this->get_logger(), "Received approach goal");
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
      const double age =
          (this->now() - latest_lidar_stamp_).seconds();
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
  카메라는 ROI 미적용: FOV 자체가 좁아 자연스럽게 전방만 보고, 가까이 접근했을
  때 roi_x_min 에 의해 객체가 통째로 잘려나가 충돌하는 문제를 피한다.
  LiDAR cloud 는 lidarCallback 에서 이미 applySpatialRoi 통과한 상태로 합쳐져
  있으므로 로봇 본체 자기반사 / 360° 잡음은 그쪽에서 걸러진다.
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
  saveDebugCloud(cloud, "roi_filtered");

  /*
  BBOX를 통해 관심영역을 설정하여서 뒤에 다른 물체들 혹은 여러 물체들이 잡혔을
  수도 있기에 클러스터링을 통해 가장 큰 클러스터만 남긴다.
  */
  const auto t3 = Clock::now();
  kdtree->setInputCloud(cloud);
  // anchor 가 잡혀있으면 그 위치 기준으로 가까운 클러스터 선택, 아니면 로봇 원점.
  // 첫 프레임 한정 anchor 없음 → 가장 가까운 클러스터.
  Eigen::Vector2f anchor_base_for_cluster;
  const Eigen::Vector2f *anchor_ptr = nullptr;
  if (aim_anchor_captured_ &&
      anchorInBase(pointcloud_msg->header.stamp, anchor_base_for_cluster)) {
    anchor_ptr = &anchor_base_for_cluster;
  }
  roi_filter_->cluster_points(cloud, kdtree, min_cluster_area_, anchor_ptr);
  const double t_cluster = elapsed_ms(t3);
  saveDebugCloud(cloud, "clustered");

  publish3Dpointcloud(cloud);

  const auto t4 = Clock::now();
  roi_filter_->projection_filter(cloud);
  roi_filter_->front_slicing(cloud);
  const double t_proj = elapsed_ms(t4);
  saveDebugCloud(cloud, "final");

  const auto t5 = Clock::now();
  obb = plane_filter_->compute_OBB(cloud);
  const double t_obb = elapsed_ms(t5);
  publish2DOBB(obb.center, obb.axis1, obb.axis2, obb.length1, obb.length2);

  /*
  OBB에서 구한 사각형 중 로봇과 가장 가까우면서도 수평방향을 띄고 있는 축을
  내적을 통해 구한다.
  */
  TargetEdge target_edge = edge_extractor_->extract_edges(
      obb.center, obb.axis1, obb.axis2, obb.length1, obb.length2);
  this->get_parameter("target_standoff_distance", target_standoff_distance_);

  /*
  Aim anchor: 첫 유효 프레임에서 로봇 정면 ray ∩ target edge 직선 교차점을
  odom 프레임에 박아두고, 이후 매 프레임 현재 OBB edge 직선에 수직 투영해서
  target_center 로 사용한다. → 시작 시 본 지점을 향해 계속 접근.
  */
  if (!aim_anchor_captured_ && target_edge.target_length > 0.1F) {
    captureAimAnchor(target_edge, pointcloud_msg->header.stamp);
  }
  if (aim_anchor_captured_) {
    Eigen::Vector2f projected;
    if (projectAnchorOnEdge(target_edge, pointcloud_msg->header.stamp,
                            projected)) {
      target_edge.target_center = projected;
    }
  }

  publishTargetEdge(target_edge);

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

  if (measure_enabled_) {
    RCLCPP_INFO(
        this->get_logger(),
        "[MEAS-STAGE] #%zu  voxel=%.1fms  outlier=%.1fms  ground=%.1fms  "
        "cluster=%.1fms  proj+slice=%.1fms  obb=%.1fms  |  "
        "sensor_lat=%.1fms",
        measure_cycle_, t_voxel, t_outlier, t_ground, t_cluster, t_proj,
        t_obb, sensor_latency_ms);
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
  상태머신: APPROACH → ALIGN_THETA → DWELL → DONE (단방향).
  - APPROACH: 종/횡/각도 PID 동시 제어 + 시작거리 기반 동적 감속 ramp.
  - ALIGN_THETA: v_x=0, w_z만 kp_theta * e_theta 로 제자리 회전 정렬.
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
        state_ = ApproachState::DWELL;
        dwell_start_time_ = current_time;
        RCLCPP_INFO(this->get_logger(),
                    "ALIGN_THETA → DWELL (eth=%.3f)", abs_eth);
      } else if ((current_time - align_start_time_).seconds() >
                 align_timeout_sec_) {
        RCLCPP_WARN(this->get_logger(),
                    "ALIGN_THETA timeout (%.1fs) → DWELL anyway",
                    align_timeout_sec_);
        state_ = ApproachState::DWELL;
        dwell_start_time_ = current_time;
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

  cmd_vel_publisher_->publish(cmd_vel);

  if (measure_enabled_) {
    const double proc_ms = elapsed_ms(t_cb);
    const double total_delay_ms = sensor_latency_ms + proc_ms;
    RCLCPP_INFO(
        this->get_logger(),
        "[MEAS] #%zu | sensor=%.1fms | proc=%.1fms | total=%.1fms | "
        "dt=%.1fms | ex=%.4f | ey=%.4f | eth=%.4f | vx=%.4f | wz=%.4f",
        measure_cycle_++, sensor_latency_ms, proc_ms, total_delay_ms,
        dt * 1000.0, se2_error.x, se2_error.y, se2_error.degree_theta,
        cmd_vel.linear.x, cmd_vel.angular.z);
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

void ApproachNode::saveDebugCloud(
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
}

void ApproachNode::startAlgorithm() {
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
  aim_anchor_captured_ = false;
  initial_dist_ = 0.0F;
  se2_error_initialized_ = false;
  consecutive_outliers_ = 0;
  if (pid_controller_) pid_controller_->reset();

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
  auto roi_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  roi_cloud->points.reserve(cloud->points.size());
  for (const auto &pt : cloud->points) {
    if (pt.x < roi_x_min_ || pt.x > roi_x_max_) {
      continue;
    }
    if (std::abs(pt.y) > roi_y_abs_max_) {
      continue;
    }
    if (pt.z > roi_z_max_) {
      continue;
    }
    roi_cloud->points.push_back(pt);
  }
  roi_cloud->width = static_cast<std::uint32_t>(roi_cloud->points.size());
  roi_cloud->height = 1;
  roi_cloud->is_dense = false;
  cloud = roi_cloud;
}

int main(int argc, char **argv) {
  const bool debug_enabled = hasDebugArg(argc, argv);
  const bool measure_enabled = hasMeasureArg(argc, argv);
  std::vector<char *> filtered_args = filterKnownArgs(argc, argv);
  rclcpp::init(static_cast<int>(filtered_args.size()), filtered_args.data());
  rclcpp::spin(std::make_shared<ApproachNode>(debug_enabled, measure_enabled));
  rclcpp::shutdown();
  return 0;
}
