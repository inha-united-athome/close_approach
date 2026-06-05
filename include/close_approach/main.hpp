#pragma once

#include "close_approach/convexhull.hpp"
#include "close_approach/edge_extractor.hpp"
#include "close_approach/error_estimator.hpp"
#include "close_approach/pid_controller.hpp"
#include "close_approach/plane_filter.hpp"
#include "close_approach/roi_filter.hpp"
#include "close_approach/target_selector.hpp"

#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "inha_interfaces/action/approach.hpp"
#include <Eigen/Dense>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <visualization_msgs/msg/marker.hpp>

enum class ApproachState {
  IDLE,
  APPROACH,
  ALIGN_THETA,
  DWELL,
  DONE
};

class ApproachNode : public rclcpp::Node {
public:
  ApproachNode();

private:
  using PointCloudMsg = sensor_msgs::msg::PointCloud2;
  using CameraInfoMsg = sensor_msgs::msg::CameraInfo;
  using ApproachAction = inha_interfaces::action::Approach;
  using GoalHandleApproach = rclcpp_action::ServerGoalHandle<ApproachAction>;

  std::shared_ptr<Filter> roi_filter_;
  std::shared_ptr<ConvexHull> convex_hull_;
  std::shared_ptr<PlaneFilter> plane_filter_;
  std::shared_ptr<PIDController> pid_controller_;
  std::shared_ptr<ErrorEstimator> error_estimator_;
  std::shared_ptr<EdgeExtractor> edge_extractor_;
  std::shared_ptr<TargetSelector> target_selector_;

  rclcpp_action::Server<ApproachAction>::SharedPtr approach_action_server_;

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const ApproachAction::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleApproach>);
  void handle_accepted(const std::shared_ptr<GoalHandleApproach> goal_handle);
  void execute(const std::shared_ptr<GoalHandleApproach> goal_handle);

  void stopAlgorithm();
  void startAlgorithm();

  rclcpp::Subscription<PointCloudMsg>::SharedPtr point_cloud_subscriber_;
  rclcpp::Subscription<PointCloudMsg>::SharedPtr lidar_subscriber_;
  rclcpp::Subscription<CameraInfoMsg>::SharedPtr camera_info_subscriber_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      filtered_pointcloud_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      debugging_pointcloud_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr obb_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      target_edge_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trail_publisher_;
  rclcpp::TimerBase::SharedPtr trail_timer_;

  rclcpp::QoS qos_best_effort_;
  rclcpp::QoS qos_reliable_;

  std::string pointcloud_topic_name_;
  std::string lidar_topic_name_;
  std::string info_topic_name_;
  std::string target_frame_;
  std::string odom_frame_;
  bool received_camera_info_ = false;
  float roi_x_min_ = 0.1F;
  float roi_x_max_ = 2.0F;
  float roi_y_abs_max_ = 0.8F;
  float roi_z_max_ = 1.5F;
  float leaf_size_ = 0.02F;
  int mean_k_ = 50;
  float stddev_mul_thresh_ = 1.0F;
  float ground_height_ = 0.0F;
  float cluster_tolerance_ = 0.02F;
  int min_cluster_size_ = 100;
  int max_cluster_size_ = 10000;
  float min_cluster_area_ = 0.09F;  // 30cm x 30cm. 물병/소품 컷, 박스/사람 통과.
  TargetSelectorParams target_selector_params_;
  float fx_ = 0.0F;
  float fy_ = 0.0F;
  float cx_ = 0.0F;
  float cy_ = 0.0F;
  float kp_x_ = 0.25F;
  float kp_y_ = 2.0F;
  float kp_theta_ = 1.2F;
  float ki_x_ = 0.0F;
  float ki_y_ = 0.0F;
  float ki_theta_ = 0.0F;
  float kd_x_ = 0.0F;
  float kd_y_ = 0.0F;
  float kd_theta_ = 0.0F;
  float tol_x_ = 0.05F;
  float tol_y_ = 0.02F;
  float tol_theta_ = 0.08F;
  float target_standoff_distance_ = 0.35F;

  // 속도/감속 한계
  float max_v_ = 0.10F;
  float max_w_ = 0.2F;
  float decel_dist_max_ = 0.20F;  // 멀리서 시작해도 감속 구간 최대치
  float decel_dist_min_ = 0.05F;  // 너무 가까이서 시작해도 최소 ramp 확보
  float decel_ratio_ = 0.5F;      // 시작 거리 대비 감속 구간 비율

  // 상태머신 / 도착 안정화
  float align_timeout_sec_ = 5.0F;
  float dwell_duration_sec_ = 2.0F;

  bool debug_enabled_ = false;
  std::filesystem::path debug_output_dir_;
  std::filesystem::path debug_action_output_dir_;
  std::filesystem::path measure_log_path_;
  std::ofstream measure_log_file_;
  std::mutex measure_log_mutex_;
  std::size_t debug_cloud_index_ = 0;
  double debug_save_period_sec_ = 1.0;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      last_debug_save_time_by_stage_;
  std::size_t measure_cycle_ = 0;

  bool start_time_flag = false;
  rclcpp::Time start_time;
  rclcpp::Time previous_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

  std::atomic<bool> control_success = false;
  std::atomic<bool> control_failure = false;
  std::string failure_message;
  std::atomic<bool> algorithm_start_flag = false;

  OBB obb;
  SE2Error se2_error;
  SE2Error se2_error_prev;

  // 상태머신
  ApproachState state_ = ApproachState::IDLE;
  rclcpp::Time dwell_start_time_;
  rclcpp::Time align_start_time_;
  bool post_align_approach_done_ = false;

  // Aim anchor: 첫 프레임 시선 교차점을 odom 프레임에 고정 → 이후 매 프레임
  // 현재 OBB edge 직선에 투영해서 target_center 로 사용.
  bool aim_anchor_captured_ = false;
  Eigen::Vector2f aim_anchor_odom_{0.0F, 0.0F};
  float initial_dist_ = 0.0F;  // 첫 anchor 캡처 시점의 종방향 거리(감속 ramp 길이 산정용)

  // 스파이크 필터 상태 (덜덜 떨림 방지)
  bool se2_error_initialized_ = false;
  int consecutive_outliers_ = 0;
  std::size_t spike_count_ = 0;
  float spike_dy_max_ = 0.25F;      // 한 사이클당 허용 y 변화 (m)
  float spike_dtheta_max_ = 0.25F;  // 한 사이클당 허용 theta 변화 (rad)
  int max_consecutive_outliers_ = 5; // 연속으로 이만큼 튀면 받아들임(씬 변경)

  // LiDAR 융합: 최신 LiDAR 클라우드를 base 프레임으로 변환·전처리해 캐싱.
  // 카메라 콜백에서 ground 제거 후 concat 한다.
  pcl::PointCloud<pcl::PointXYZ>::Ptr latest_lidar_cloud_;
  rclcpp::Time latest_lidar_stamp_;
  std::mutex lidar_cloud_mutex_;
  float lidar_max_age_sec_ = 0.3F;  // 릴레이 끊김 안전장치 (PTP 동기 가정)

  // 후진용 trail 기록 (odom 프레임 기준 base 위치 시퀀스)
  std::deque<geometry_msgs::msg::PoseStamped> trail_;
  std::mutex trail_mutex_;
  float trail_min_dist_ = 0.03F;   // arc-length 다운샘플 거리 임계 (m)
  float trail_min_yaw_ = 0.052F;   // arc-length 다운샘플 yaw 임계 (rad, ~3deg)

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  void pointCloudCallback(const PointCloudMsg::ConstSharedPtr &pointcloud_msg);
  void lidarCallback(const PointCloudMsg::ConstSharedPtr &lidar_msg);
  void cameraInfoCallback(const CameraInfoMsg::SharedPtr msg);
  void applySpatialRoi(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  bool getTransform(const std::string &target_frame,
                    const std::string &source_frame,
                    geometry_msgs::msg::TransformStamped &transform,
                    const rclcpp::Time &stamp = rclcpp::Time(0));

  void publish3Dpointcloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  void publish2DOBB(const Eigen::Vector2f &center, const Eigen::Vector2f &axis1,
                    const Eigen::Vector2f &axis2, const float length1,
                    const float length2);
  void publishTargetEdge(const TargetEdge &target_edge);
  void saveDebugCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                      const std::string &stage);
  void openDebugActionDirectory();
  void openMeasureLog();
  void closeMeasureLog();
  void writeMeasureLog(const std::string &line);

  void recordTrailPose();
  void publishTrail();

  // Aim anchor 관련
  bool captureAimAnchor(const TargetEdge &target_edge,
                        const rclcpp::Time &stamp);
  bool projectAnchorOnEdge(const TargetEdge &target_edge,
                           const rclcpp::Time &stamp,
                           Eigen::Vector2f &out_center);
  // odom 에 박힌 anchor 를 stamp 시점의 base 프레임으로 변환.
  bool anchorInBase(const rclcpp::Time &stamp, Eigen::Vector2f &out_xy);
};
