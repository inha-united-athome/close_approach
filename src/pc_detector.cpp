#include "close_approach/pc_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>

namespace {
void applySpatialRoiBounds(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                           float x_min, float x_max, float y_abs_near,
                           float y_abs_far, float z_max) {
  auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  out->points.reserve(cloud->points.size());
  const float near_x = std::isfinite(x_min) ? x_min : 0.0F;
  const float x_span = std::max(1e-3F, x_max - near_x);
  for (const auto &p : cloud->points) {
    if (p.x < x_min || p.x > x_max) continue;
    const float t = std::clamp((p.x - near_x) / x_span, 0.0F, 1.0F);
    const float y_abs_limit = y_abs_near + t * (y_abs_far - y_abs_near);
    if (std::abs(p.y) > y_abs_limit) continue;
    if (p.z > z_max)                  continue;
    out->points.push_back(p);
  }
  out->width  = static_cast<uint32_t>(out->points.size());
  out->height = 1;
  out->is_dense = false;
  cloud = out;
}

bool keepFrontFraction(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                       float ratio, int min_points,
                       float &representative_x, float &representative_y) {
  if (!cloud || cloud->empty()) return false;

  std::vector<pcl::PointXYZ> sorted;
  sorted.reserve(cloud->size());
  for (const auto &point : cloud->points) {
    if (std::isfinite(point.x) && std::isfinite(point.y)) {
      sorted.push_back(point);
    }
  }
  if (sorted.empty()) return false;

  std::sort(sorted.begin(), sorted.end(),
            [](const pcl::PointXYZ &a, const pcl::PointXYZ &b) {
              return a.x < b.x;
            });
  const std::size_t ratio_count = static_cast<std::size_t>(
      std::ceil(sorted.size() * std::clamp(ratio, 0.01F, 1.0F)));
  const std::size_t keep_count = std::min(
      sorted.size(), std::max<std::size_t>(std::max(1, min_points),
                                           ratio_count));

  // x는 이미 정렬돼 있으므로 전면 slice의 중앙값을 바로 취한다.
  representative_x = sorted[keep_count / 2].x;
  std::vector<float> y_values;
  y_values.reserve(keep_count);
  for (std::size_t i = 0; i < keep_count; ++i) {
    y_values.push_back(sorted[i].y);
  }
  const auto y_mid = y_values.begin() +
                     static_cast<std::ptrdiff_t>(y_values.size() / 2);
  std::nth_element(y_values.begin(), y_mid, y_values.end());
  representative_y = *y_mid;

  auto front = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  front->points.assign(sorted.begin(), sorted.begin() +
                                        static_cast<std::ptrdiff_t>(keep_count));
  front->width = static_cast<std::uint32_t>(front->points.size());
  front->height = 1;
  front->is_dense = true;
  cloud = front;
  return true;
}
} // namespace

PCDetector::PCDetector()
    : Node("pc_detector"),
      qos_be_(rclcpp::QoS(rclcpp::KeepLast(10)).best_effort()),
      qos_rel_(rclcpp::QoS(rclcpp::KeepLast(10)).reliable()),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_) {

  roi_filter_      = std::make_shared<Filter>();
  convex_hull_     = std::make_shared<ConvexHull>();
  plane_filter_    = std::make_shared<PlaneFilter>();
  error_estimator_ = std::make_shared<ErrorEstimator>();
  edge_extractor_  = std::make_shared<EdgeExtractor>();

  // Params
  this->declare_parameter<std::string>("cloud_topic",  "/camera/camera_head/depth/color/points");
  this->declare_parameter<std::string>("lidar_topic",  "/livox/lidar");
  this->declare_parameter<std::string>("info_topic",   "/camera/camera_head/color/camera_info");
  this->declare_parameter<std::string>("target_frame", "base_nav");
  this->declare_parameter<std::string>("odom_frame",   "odom");
  this->declare_parameter<float>("roi_x_min",              0.1F);
  this->declare_parameter<float>("roi_x_max",              2.0F);
  this->declare_parameter<float>("roi_y_abs_near",         0.3F);
  this->declare_parameter<float>("roi_y_abs_max",          0.8F);
  this->declare_parameter<float>("roi_z_max",              1.5F);
  this->declare_parameter<float>("leaf_size",              0.03F);
  this->declare_parameter<int>  ("mean_k",                 50);
  this->declare_parameter<float>("stddev_mul_thresh",      0.5F);
  this->declare_parameter<float>("ground_height",          0.1F);
  this->declare_parameter<float>("cluster_tolerance",      0.05F);
  this->declare_parameter<int>  ("min_cluster_size",       100);
  this->declare_parameter<int>  ("max_cluster_size",       10000);
  this->declare_parameter<float>("min_cluster_area",       0.09F);
  this->declare_parameter<float>("target_standoff_distance", 0.3F);
  this->declare_parameter<float>("front_slice_ratio",       0.05F);
  this->declare_parameter<int>  ("front_min_points",        5);
  this->declare_parameter<float>("x_ema_alpha",             0.35F);
  this->declare_parameter<float>("spike_dx_max",            0.15F);
  this->declare_parameter<int>  ("max_consecutive_outliers", 5);
  this->declare_parameter<float>("lidar_max_age_sec",      0.3F);
  this->declare_parameter<bool> ("debug_log",              true);

  this->get_parameter("cloud_topic",  cloud_topic_);
  this->get_parameter("lidar_topic",  lidar_topic_);
  this->get_parameter("info_topic",   info_topic_);
  this->get_parameter("target_frame", target_frame_);
  this->get_parameter("odom_frame",   odom_frame_);
  this->get_parameter("roi_x_min",            roi_x_min_);
  this->get_parameter("roi_x_max",            roi_x_max_);
  this->get_parameter("roi_y_abs_near",       roi_y_abs_near_);
  this->get_parameter("roi_y_abs_max",        roi_y_abs_max_);
  this->get_parameter("roi_z_max",            roi_z_max_);
  roi_y_abs_near_ = std::max(0.0F, roi_y_abs_near_);
  roi_y_abs_max_ = std::max(roi_y_abs_near_, roi_y_abs_max_);
  this->get_parameter("leaf_size",            leaf_size_);
  this->get_parameter("mean_k",               mean_k_);
  this->get_parameter("stddev_mul_thresh",    stddev_mul_thresh_);
  this->get_parameter("ground_height",        ground_height_);
  this->get_parameter("cluster_tolerance",    cluster_tolerance_);
  this->get_parameter("min_cluster_size",     min_cluster_size_);
  this->get_parameter("max_cluster_size",     max_cluster_size_);
  this->get_parameter("min_cluster_area",     min_cluster_area_);
  this->get_parameter("target_standoff_distance", target_standoff_distance_);
  this->get_parameter("front_slice_ratio",     front_slice_ratio_);
  this->get_parameter("front_min_points",      front_min_points_);
  this->get_parameter("x_ema_alpha",           x_ema_alpha_);
  this->get_parameter("spike_dx_max",          spike_dx_max_);
  this->get_parameter("max_consecutive_outliers", max_consecutive_outliers_);
  this->get_parameter("lidar_max_age_sec",    lidar_max_age_sec_);
  this->get_parameter("debug_log",            debug_log_);
  front_slice_ratio_ = std::clamp(front_slice_ratio_, 0.01F, 1.0F);
  front_min_points_ = std::max(1, front_min_points_);
  x_ema_alpha_ = std::clamp(x_ema_alpha_, 0.01F, 1.0F);
  spike_dx_max_ = std::max(0.0F, spike_dx_max_);

  roi_filter_->setParameters(leaf_size_, mean_k_, stddev_mul_thresh_,
                              ground_height_, cluster_tolerance_,
                              min_cluster_size_, max_cluster_size_);

  // Subscribers
  cloud_sub_   = this->create_subscription<CloudMsg>(
      cloud_topic_, qos_be_,
      std::bind(&PCDetector::cloudCallback, this, std::placeholders::_1));
  lidar_sub_   = this->create_subscription<CloudMsg>(
      lidar_topic_, qos_be_,
      std::bind(&PCDetector::lidarCallback, this, std::placeholders::_1));
  cam_info_sub_= this->create_subscription<CamInfoMsg>(
      info_topic_, qos_be_,
      std::bind(&PCDetector::camInfoCallback, this, std::placeholders::_1));
  active_sub_  = this->create_subscription<std_msgs::msg::Bool>(
      "/approach/active", qos_rel_,
      std::bind(&PCDetector::activeCallback, this, std::placeholders::_1));

  // Publishers
  pc_error_pub_      = this->create_publisher<ApproachError>("/approach/pc_error", qos_rel_);
  filtered_cloud_pub_= this->create_publisher<CloudMsg>("/approach/filtered_cloud", qos_be_);
  debug_cloud_pub_   = this->create_publisher<CloudMsg>("/approach/debug_cloud", qos_be_);
  obb_pub_           = this->create_publisher<visualization_msgs::msg::Marker>(
      "/approach/obb_marker", qos_rel_);
  edge_pub_          = this->create_publisher<visualization_msgs::msg::Marker>(
      "/approach/target_edge_marker", qos_rel_);

  RCLCPP_INFO(this->get_logger(), "PCDetector ready. cloud_topic=%s", cloud_topic_.c_str());
}

void PCDetector::activeCallback(const std_msgs::msg::Bool::SharedPtr msg) {
  const bool was_active = is_active_.exchange(msg->data,
                                               std::memory_order_acq_rel);
  if (!was_active && msg->data) {
    // Approach just started → reset anchor and spike filter
    aim_anchor_captured_ = false;
    initial_dist_        = 0.0F;
    se2_initialized_     = false;
    consecutive_outliers_= 0;
    RCLCPP_INFO(this->get_logger(), "Active: aim anchor reset");
  } else if (was_active && !msg->data) {
    std::lock_guard<std::mutex> lk(lidar_mutex_);
    lidar_cache_.reset();
    RCLCPP_INFO(this->get_logger(), "Inactive: point-cloud processing stopped");
  }
}

void PCDetector::camInfoCallback(const CamInfoMsg::SharedPtr msg) {
  if (received_camera_info_) return;
  roi_filter_->setCameraInfo(msg->k[0], msg->k[4], msg->k[2], msg->k[5]);
  received_camera_info_ = true;
  RCLCPP_INFO(this->get_logger(), "CameraInfo received");
}

void PCDetector::lidarCallback(const CloudMsg::ConstSharedPtr &msg) {
  if (!is_active_.load(std::memory_order_acquire)) return;

  auto lidar_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::fromROSMsg(*msg, *lidar_cloud);
  if (lidar_cloud->empty()) return;

  roi_filter_->voxel_downsampling(lidar_cloud);
  roi_filter_->remove_outliers(lidar_cloud);

  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_.lookupTransform(target_frame_, msg->header.frame_id,
                                    tf2::TimePointZero, tf2::durationFromSec(0.2));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "lidar TF failed: %s", ex.what());
    return;
  }
  roi_filter_->remove_ground(lidar_cloud, tf.transform);
  applySpatialRoi(lidar_cloud);

  std::lock_guard<std::mutex> lk(lidar_mutex_);
  lidar_cache_ = lidar_cloud;
  lidar_stamp_ = msg->header.stamp;
}

void PCDetector::cloudCallback(const CloudMsg::ConstSharedPtr &msg) {
  if (!is_active_.load(std::memory_order_acquire)) return;

  cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  kdtree_.reset(new pcl::search::KdTree<pcl::PointXYZ>);

  pcl::fromROSMsg(*msg, *cloud_);
  if (cloud_->empty()) {
    publishInvalid();
    return;
  }

  roi_filter_->voxel_downsampling(cloud_);
  roi_filter_->remove_outliers(cloud_);

  geometry_msgs::msg::TransformStamped tf;
  if (!getTransform(target_frame_, msg->header.frame_id, tf, msg->header.stamp)) {
    publishInvalid();
    return;
  }

  roi_filter_->remove_ground(cloud_, tf.transform);
  applySpatialRoiBounds(cloud_, roi_x_min_,
                        roi_x_max_, roi_y_abs_near_, roi_y_abs_max_,
                        roi_z_max_);

  // LiDAR fusion
  {
    std::lock_guard<std::mutex> lk(lidar_mutex_);
    if (lidar_cache_ && !lidar_cache_->empty()) {
      const double age = (this->now() - lidar_stamp_).seconds();
      if (age < static_cast<double>(lidar_max_age_sec_)) {
        *cloud_ += *lidar_cache_;
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Stale lidar cloud (%.2fs), skipping", age);
      }
    }
  }

  if (cloud_->empty()) {
    publishInvalid();
    return;
  }

  // Debug cloud
  {
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*cloud_, out);
    out.header.stamp    = this->now();
    out.header.frame_id = target_frame_;
    debug_cloud_pub_->publish(out);
  }

  // Clustering
  kdtree_->setInputCloud(cloud_);
  Eigen::Vector2f anchor_base;
  const Eigen::Vector2f *anchor_ptr = nullptr;
  if (aim_anchor_captured_ && anchorInBase(msg->header.stamp, anchor_base)) {
    anchor_ptr = &anchor_base;
  }
  roi_filter_->cluster_points(cloud_, kdtree_, min_cluster_area_, anchor_ptr);
  if (cloud_->empty()) {
    publishInvalid();
    return;
  }

  // XY projection. OBB/anchor is retained only for target association and RViz;
  // longitudinal control below no longer depends on its normal/yaw estimate.
  roi_filter_->projection_filter(cloud_);
  OBB        obb         = plane_filter_->compute_OBB(cloud_);
  TargetEdge target_edge = edge_extractor_->extract_edges(
      obb.center, obb.axis1, obb.axis2, obb.length1, obb.length2);

  publishOBB(obb);

  // Aim anchor
  if (!aim_anchor_captured_ && target_edge.target_length > 0.1F) {
    captureAimAnchor(target_edge, msg->header.stamp);
  }
  if (aim_anchor_captured_) {
    Eigen::Vector2f projected;
    if (projectAnchorOnEdge(target_edge, msg->header.stamp, projected)) {
      target_edge.target_center = projected;
    }
  }
  publishTargetEdge(target_edge);

  // Keep only the nearest X fraction of the selected target cluster. The
  // median of that front slice is a robust representative surface distance.
  float representative_x = 0.0F;
  float representative_y = 0.0F;
  if (!keepFrontFraction(cloud_, front_slice_ratio_, front_min_points_,
                         representative_x, representative_y)) {
    publishInvalid();
    return;
  }

  // Filtered cloud now visualizes exactly the points used for longitudinal x.
  {
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*cloud_, out);
    out.header.stamp    = this->now();
    out.header.frame_id = target_frame_;
    filtered_cloud_pub_->publish(out);
  }

  SE2Error candidate;
  candidate.x = representative_x - target_standoff_distance_;
  candidate.y = 0.0F;
  candidate.degree_theta = 0.0F;

  // X-only spike rejection + EMA. Y and theta belong to neither the
  // longitudinal detector nor its controller path.
  if (!se2_initialized_) {
    se2_cached_      = candidate;
    se2_initialized_ = true;
    consecutive_outliers_ = 0;
    initial_dist_ = std::abs(candidate.x);
  } else {
    const float dx = std::abs(candidate.x - se2_cached_.x);
    const bool spike = dx > spike_dx_max_;
    if (spike) {
      if (++consecutive_outliers_ >= max_consecutive_outliers_) {
        se2_cached_           = candidate;
        consecutive_outliers_ = 0;
      }
    } else {
      se2_cached_.x = x_ema_alpha_ * candidate.x +
                      (1.0F - x_ema_alpha_) * se2_cached_.x;
      se2_cached_.y = 0.0F;
      se2_cached_.degree_theta = 0.0F;
      consecutive_outliers_ = 0;
    }
  }

  // Publish x only. Theta comes from depth edge and y is intentionally unused.
  ApproachError err;
  err.header.stamp    = msg->header.stamp;
  err.valid           = true;
  err.x_error         = se2_cached_.x;
  err.y_error         = 0.0f;
  err.theta_error     = 0.0f;   // theta comes from edge_detector
  err.initial_dist_m  = initial_dist_;
  err.mean_y_px       = 0.0f;
  pc_error_pub_->publish(err);

  if (debug_log_) {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 300,
        "pc_error valid=1 x=%.4fm surface_x=%.4fm surface_y=%.4fm "
        "init=%.4fm front=%zu(%.1f%%)",
        err.x_error, representative_x, representative_y, err.initial_dist_m,
        cloud_->size(), front_slice_ratio_ * 100.0F);
  }
}

void PCDetector::publishInvalid() {
  ApproachError err;
  err.header.stamp = this->now();
  err.valid = false;
  pc_error_pub_->publish(err);
  if (debug_log_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "pc_error valid=0 (no usable longitudinal target)");
  }
}

bool PCDetector::getTransform(const std::string &tgt, const std::string &src,
                               geometry_msgs::msg::TransformStamped &tf_out,
                               const rclcpp::Time &stamp) {
  try {
    tf_out = tf_buffer_.lookupTransform(tgt, src, stamp,
                                        rclcpp::Duration::from_seconds(0.1));
    return true;
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "TF %s→%s failed: %s", src.c_str(), tgt.c_str(), ex.what());
    return false;
  }
}

void PCDetector::applySpatialRoi(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  applySpatialRoiBounds(cloud, roi_x_min_, roi_x_max_, roi_y_abs_near_,
                        roi_y_abs_max_, roi_z_max_);
}

bool PCDetector::captureAimAnchor(const TargetEdge &edge, const rclcpp::Time &stamp) {
  const Eigen::Vector2f &c = edge.target_center;
  const Eigen::Vector2f &a = edge.target_axis;
  const float half_L = edge.target_length * 0.5F;

  Eigen::Vector2f aim_base;
  if (std::abs(a.y()) < 1e-3F) {
    aim_base = c;
  } else {
    float t = std::clamp(-c.y() / a.y(), -half_L, half_L);
    aim_base = c + t * a;
  }

  geometry_msgs::msg::TransformStamped tf;
  if (!getTransform(odom_frame_, target_frame_, tf, stamp)) return false;

  const double yaw = tf2::getYaw(tf.transform.rotation);
  const float cy_v = static_cast<float>(std::cos(yaw));
  const float sy_v = static_cast<float>(std::sin(yaw));
  const auto &tr   = tf.transform.translation;
  aim_anchor_odom_.x() = cy_v * aim_base.x() - sy_v * aim_base.y() + (float)tr.x;
  aim_anchor_odom_.y() = sy_v * aim_base.x() + cy_v * aim_base.y() + (float)tr.y;

  const Eigen::Vector2f &n = edge.normal_axis;
  initial_dist_ = std::abs(aim_base.dot(n) - target_standoff_distance_);

  aim_anchor_captured_ = true;
  RCLCPP_INFO(this->get_logger(),
              "Aim anchor captured base=(%.3f,%.3f) odom=(%.3f,%.3f) init_dist=%.3f",
              aim_base.x(), aim_base.y(),
              aim_anchor_odom_.x(), aim_anchor_odom_.y(), initial_dist_);
  return true;
}

bool PCDetector::anchorInBase(const rclcpp::Time &stamp, Eigen::Vector2f &out_xy) {
  geometry_msgs::msg::TransformStamped tf;
  if (!getTransform(target_frame_, odom_frame_, tf, stamp)) return false;

  const double yaw = tf2::getYaw(tf.transform.rotation);
  const float cy_v = static_cast<float>(std::cos(yaw));
  const float sy_v = static_cast<float>(std::sin(yaw));
  const auto &tr   = tf.transform.translation;
  out_xy.x() = cy_v * aim_anchor_odom_.x() - sy_v * aim_anchor_odom_.y() + (float)tr.x;
  out_xy.y() = sy_v * aim_anchor_odom_.x() + cy_v * aim_anchor_odom_.y() + (float)tr.y;
  return true;
}

bool PCDetector::projectAnchorOnEdge(const TargetEdge &edge, const rclcpp::Time &stamp,
                                      Eigen::Vector2f &out_center) {
  Eigen::Vector2f anchor_base;
  if (!anchorInBase(stamp, anchor_base)) return false;

  const Eigen::Vector2f  a_unit = edge.target_axis.normalized();
  float s = (anchor_base - edge.target_center).dot(a_unit);
  s = std::clamp(s, -edge.target_length * 0.5F, edge.target_length * 0.5F);
  out_center = edge.target_center + s * a_unit;
  return true;
}

void PCDetector::publishOBB(const OBB &obb) {
  visualization_msgs::msg::Marker m;
  m.header.frame_id = target_frame_;
  m.header.stamp    = this->now();
  m.ns = "obb"; m.id = 0;
  m.type   = visualization_msgs::msg::Marker::LINE_STRIP;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.scale.x = 0.05;
  m.color.g = 1.0f; m.color.a = 1.0f;

  auto corner = [&](float s1, float s2) {
    Eigen::Vector2f p = obb.center + s1 * (obb.length1 / 2.0f) * obb.axis1
                                   + s2 * (obb.length2 / 2.0f) * obb.axis2;
    geometry_msgs::msg::Point pt;
    pt.x = p.x(); pt.y = p.y(); pt.z = 0.0;
    return pt;
  };
  m.points = {corner(1,1), corner(-1,1), corner(-1,-1), corner(1,-1), corner(1,1)};
  obb_pub_->publish(m);
}

void PCDetector::publishTargetEdge(const TargetEdge &edge) {
  visualization_msgs::msg::Marker m;
  m.header.frame_id = target_frame_;
  m.header.stamp    = this->now();
  m.ns = "target_edge"; m.id = 0;
  m.type   = visualization_msgs::msg::Marker::LINE_LIST;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.scale.x = 0.05;
  m.color.r = 1.0f; m.color.a = 1.0f;

  auto pt2 = [](float x, float y) {
    geometry_msgs::msg::Point p; p.x = x; p.y = y; p.z = 0.0; return p;
  };
  Eigen::Vector2f p1 = edge.target_center + (edge.target_length / 2.0f) * edge.target_axis;
  Eigen::Vector2f p2 = edge.target_center - (edge.target_length / 2.0f) * edge.target_axis;
  m.points = {pt2(p1.x(), p1.y()), pt2(p2.x(), p2.y())};
  edge_pub_->publish(m);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PCDetector>());
  rclcpp::shutdown();
  return 0;
}
