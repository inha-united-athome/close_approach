#pragma once

#include "close_approach/convexhull.hpp"
#include "close_approach/edge_extractor.hpp"
#include "close_approach/error_estimator.hpp"
#include "close_approach/plane_filter.hpp"
#include "close_approach/roi_filter.hpp"
#include "close_approach/msg/approach_error.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>

class PCDetector : public rclcpp::Node {
public:
  PCDetector();

private:
  using CloudMsg    = sensor_msgs::msg::PointCloud2;
  using CamInfoMsg  = sensor_msgs::msg::CameraInfo;
  using ApproachError = close_approach::msg::ApproachError;

  // Processing helpers
  std::shared_ptr<Filter>         roi_filter_;
  std::shared_ptr<ConvexHull>     convex_hull_;
  std::shared_ptr<PlaneFilter>    plane_filter_;
  std::shared_ptr<ErrorEstimator> error_estimator_;
  std::shared_ptr<EdgeExtractor>  edge_extractor_;

  // Subscribers
  rclcpp::Subscription<CloudMsg>::SharedPtr    cloud_sub_;
  rclcpp::Subscription<CloudMsg>::SharedPtr    lidar_sub_;
  rclcpp::Subscription<CamInfoMsg>::SharedPtr  cam_info_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr active_sub_;

  // Publishers
  rclcpp::Publisher<ApproachError>::SharedPtr              pc_error_pub_;
  rclcpp::Publisher<CloudMsg>::SharedPtr                   filtered_cloud_pub_;
  rclcpp::Publisher<CloudMsg>::SharedPtr                   debug_cloud_pub_;
  rclcpp::Publisher<CloudMsg>::SharedPtr                   colored_cloud_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr      pc_debug_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr obb_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr edge_pub_;

  rclcpp::QoS qos_be_, qos_rel_;

  // Params
  std::string cloud_topic_, lidar_topic_, info_topic_, target_frame_, odom_frame_;
  float roi_x_min_, roi_x_max_, roi_y_abs_near_, roi_y_abs_max_, roi_z_max_;
  float leaf_size_, stddev_mul_thresh_, ground_height_;
  int   mean_k_;
  float cluster_tolerance_, min_cluster_area_;
  int   min_cluster_size_, max_cluster_size_;
  float target_standoff_distance_;
  float front_slice_ratio_ = 0.05F;
  int   front_min_points_ = 5;
  float x_ema_alpha_ = 0.80F;
  float spike_dx_max_ = 0.15F;
  int   max_consecutive_outliers_;
  float lidar_max_age_sec_;
  bool  debug_log_ = true;

  // Target selection: when false, skip Euclidean clustering and run the front
  // slice directly on the ROI + self-filtered cloud (nearest-obstacle = target).
  bool  use_clustering_ = true;

  // When false, ignore the LiDAR entirely (no subscription, no fusion).
  bool  use_lidar_ = true;

  // Robot self filter (TF-driven box exclusion). Each volume is a box defined
  // in a robot link frame; points inside any box (after live TF lookup) are
  // dropped so the nearest-point logic never locks onto the robot itself.
  bool  self_filter_enabled_ = false;
  float self_filter_padding_ = 0.0F;
  std::vector<std::string> self_filter_frames_;
  std::vector<double> self_filter_size_x_, self_filter_size_y_, self_filter_size_z_;
  std::vector<double> self_filter_off_x_,  self_filter_off_y_,  self_filter_off_z_;

  // Plane-normal yaw fallback: for a vertical surface (wall/front panel) near the
  // target distance, fit a plane and derive yaw from its horizontal normal. Used
  // by the manager only when the image edge yaw is stale.
  float yaw_plane_band_         = 0.15F;  // m, depth band from the near surface
  float yaw_plane_dist_thresh_  = 0.02F;  // m, RANSAC inlier distance
  int   yaw_plane_min_inliers_  = 50;
  float yaw_plane_max_normal_z_ = 0.4F;   // |n_z| below this = vertical surface
  float yaw_plane_max_yaw_deg_  = 45.0F;  // |yaw| beyond this → treat as misdetect
  float yaw_plane_sign_         = 1.0F;   // flip to match edge yaw sign convention

  // State
  bool  received_camera_info_ = false;
  std::atomic<bool> is_active_{false};
  bool  aim_anchor_captured_  = false;
  float initial_dist_         = 0.0F;
  Eigen::Vector2f aim_anchor_odom_{0.0F, 0.0F};
  bool      se2_initialized_  = false;
  SE2Error  se2_cached_{0.0F, 0.0F, 0.0F};
  int       consecutive_outliers_ = 0;

  // LiDAR cache
  pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_cache_;
  rclcpp::Time                         lidar_stamp_;
  std::mutex                           lidar_mutex_;

  // PCL workspace
  pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_;

  // TF
  tf2_ros::Buffer           tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Callbacks
  void cloudCallback(const CloudMsg::ConstSharedPtr &msg);
  void lidarCallback(const CloudMsg::ConstSharedPtr &msg);
  void camInfoCallback(const CamInfoMsg::SharedPtr msg);
  void activeCallback(const std_msgs::msg::Bool::SharedPtr msg);

  // Helpers
  bool getTransform(const std::string &tgt, const std::string &src,
                    geometry_msgs::msg::TransformStamped &tf_out,
                    const rclcpp::Time &stamp = rclcpp::Time(0));
  void applySpatialRoi(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
  void applySelfFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                       const rclcpp::Time &stamp);
  // Fit a plane to the near surface (within yaw_plane_band of ref_x, so a far
  // wall behind the target is discarded) and, if it is vertical enough, return
  // the yaw needed to face it. False = not a usable vertical surface.
  bool computeSurfaceYaw(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                         float ref_x, float &yaw_out);
  bool captureAimAnchor(const TargetEdge &edge, const rclcpp::Time &stamp);
  bool anchorInBase(const rclcpp::Time &stamp, Eigen::Vector2f &out_xy);
  bool projectAnchorOnEdge(const TargetEdge &edge, const rclcpp::Time &stamp,
                           Eigen::Vector2f &out_center);
  void publishOBB(const OBB &obb);
  void publishTargetEdge(const TargetEdge &edge);
  void publishPcDebug(const std::string &status);
  void publishInvalid(const std::string &reason = "unspecified");
};
