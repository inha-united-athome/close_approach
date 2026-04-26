#pragma once

#include <rclcpp/rclcpp.hpp>


#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

#include "inha_interfaces/srv/accumulation.hpp"

struct Trajectory3D{
  float x;
  float y;
  float z;
  float roll;
  float pitch;
  float yaw;
  int index;
};

struct Trajectory{
  float x;
  float y;
  float theta;
  int index;
};

class AccumulationAndIcp : public rclcpp::Node {
public:
  explicit AccumulationAndIcp(bool debug_enabled = false);

private:
  using AccumService = inha_interfaces::srv::Accumulation;

  // --- Service Server ---
  rclcpp::CallbackGroup::SharedPtr service_callback_group_;
  rclcpp::Service<AccumService>::SharedPtr accum_service_server_;

  void handle_accumulate_request(
      const std::shared_ptr<AccumService::Request> request,
      std::shared_ptr<AccumService::Response> response);

  // --- Subscribers ---
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      pointcloud_subscriber_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;

  // --- Publisher ---
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      accumulated_cloud_publisher_;

  // --- QoS ---
  rclcpp::QoS qos_best_effort_;
  rclcpp::QoS qos_reliable_;

  // --- TF ---
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // --- Timer ---
  rclcpp::TimerBase::SharedPtr accumulation_timer_;

  // --- Parameters ---
  std::string lidar_topic_name_;
  std::string odom_topic_name_;
  std::string lidar_frame_;
  std::string base_frame_;
  std::string map_frame_;
  float voxel_leaf_size_;
  int icp_max_iterations_;
  double icp_transform_epsilon_;
  double icp_max_correspondence_distance_;
  float roi_x_min_;
  float roi_x_max_;
  float roi_y_min_;
  float roi_y_max_;
  float roi_z_min_;
  float roi_z_max_;
  bool debug_enabled_{false};
  std::filesystem::path debug_output_dir_;
  std::size_t debug_cloud_index_{0};

  // --- Trajectory & State ---
  Trajectory current_trajectory_;
  std::vector<Trajectory> trajectory_history_;
  int trajectory_index_{0};

  // --- State ---
  std::atomic<bool> algorithm_running_{false};
  std::atomic<int32_t> max_scans_{0};
  std::atomic<int32_t> accumulated_count_{0};

  // 최신 스캔 버퍼 (subscriber가 계속 갱신, timer가 소비)
  std::mutex scan_mutex_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr latest_scan_;
  rclcpp::Time latest_scan_stamp_;
  bool has_new_scan_{false};

  // 누적 클라우드
  pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_cloud_;

  // Odom 상태 (ICP 초기값 계산용)
  std::mutex odom_mutex_;
  Eigen::Matrix4f prev_odom_pose_;
  Eigen::Matrix4f curr_odom_pose_;
  bool has_prev_odom_{false};

  // --- Callbacks ---
  void
  pointcloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr &msg);
  void accumulationTimerCallback();

  // --- Core ---
  Eigen::Matrix4f runICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr &target,
                         const pcl::PointCloud<pcl::PointXYZ>::Ptr &source,
                         const Eigen::Matrix4f &initial_guess);
  void downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  void publishAccumulatedCloud();
  void SetROI(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

  // --- Helpers ---
  Eigen::Matrix4f odomMsgToMatrix(const nav_msgs::msg::Odometry &odom);
  void startAlgorithm(double hz);
  void stopAlgorithm();
  void resetState();
  void saveDebugCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                      const std::string &stage);
};
