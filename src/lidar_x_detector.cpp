#include "close_approach/msg/approach_error.hpp"
#include "close_approach/roi_filter.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

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
    if (p.z > z_max) continue;
    out->points.push_back(p);
  }
  out->width = static_cast<std::uint32_t>(out->points.size());
  out->height = 1;
  out->is_dense = false;
  cloud = out;
}

bool keepFrontFraction(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                       float ratio, int min_points,
                       float &representative_x,
                       float &representative_y) {
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
  front->is_dense = false;
  cloud = front;
  return true;
}

}  // namespace

class LidarXDetector final : public rclcpp::Node {
public:
  using ApproachError = close_approach::msg::ApproachError;
  using CloudMsg = sensor_msgs::msg::PointCloud2;

  LidarXDetector()
      : Node("lidar_x_detector"),
        qos_be_(rclcpp::QoS(rclcpp::KeepLast(10)).best_effort()),
        qos_rel_(rclcpp::QoS(rclcpp::KeepLast(10)).reliable()),
        tf_buffer_(get_clock()),
        tf_listener_(tf_buffer_) {
    roi_filter_ = std::make_shared<Filter>();

    declare_parameter<std::string>("lidar_topic", "/livox/lidar");
    declare_parameter<std::string>("target_frame", "base_nav");
    declare_parameter<float>("roi_x_min", 0.1F);
    declare_parameter<float>("roi_x_max", 3.0F);
    declare_parameter<float>("roi_y_abs_near", 0.3F);
    declare_parameter<float>("roi_y_abs_max", 0.8F);
    declare_parameter<float>("roi_z_max", 1.5F);
    declare_parameter<float>("leaf_size", 0.06F);
    declare_parameter<int>("mean_k", 50);
    declare_parameter<float>("stddev_mul_thresh", 0.5F);
    declare_parameter<float>("ground_height", 0.1F);
    declare_parameter<float>("target_standoff_distance", 0.3F);
    declare_parameter<float>("front_slice_ratio", 0.05F);
    declare_parameter<int>("front_min_points", 5);
    declare_parameter<bool>("debug_log", true);

    declare_parameter<bool>("self_filter.enabled", false);
    declare_parameter<double>("self_filter.padding", 0.03);
    declare_parameter<std::vector<std::string>>(
        "self_filter.frames", std::vector<std::string>{});
    declare_parameter<std::vector<double>>(
        "self_filter.size_x", std::vector<double>{});
    declare_parameter<std::vector<double>>(
        "self_filter.size_y", std::vector<double>{});
    declare_parameter<std::vector<double>>(
        "self_filter.size_z", std::vector<double>{});
    declare_parameter<std::vector<double>>(
        "self_filter.offset_x", std::vector<double>{});
    declare_parameter<std::vector<double>>(
        "self_filter.offset_y", std::vector<double>{});
    declare_parameter<std::vector<double>>(
        "self_filter.offset_z", std::vector<double>{});

    get_parameter("lidar_topic", lidar_topic_);
    get_parameter("target_frame", target_frame_);
    get_parameter("roi_x_min", roi_x_min_);
    get_parameter("roi_x_max", roi_x_max_);
    get_parameter("roi_y_abs_near", roi_y_abs_near_);
    get_parameter("roi_y_abs_max", roi_y_abs_max_);
    get_parameter("roi_z_max", roi_z_max_);
    get_parameter("leaf_size", leaf_size_);
    get_parameter("mean_k", mean_k_);
    get_parameter("stddev_mul_thresh", stddev_mul_thresh_);
    get_parameter("ground_height", ground_height_);
    get_parameter("target_standoff_distance", target_standoff_distance_);
    get_parameter("front_slice_ratio", front_slice_ratio_);
    get_parameter("front_min_points", front_min_points_);
    get_parameter("debug_log", debug_log_);

    roi_y_abs_near_ = std::max(0.0F, roi_y_abs_near_);
    roi_y_abs_max_ = std::max(roi_y_abs_near_, roi_y_abs_max_);
    front_slice_ratio_ = std::clamp(front_slice_ratio_, 0.01F, 1.0F);
    front_min_points_ = std::max(1, front_min_points_);

    double pad = 0.03;
    get_parameter("self_filter.enabled", self_filter_enabled_);
    get_parameter("self_filter.padding", pad);
    self_filter_padding_ = static_cast<float>(std::max(0.0, pad));
    get_parameter("self_filter.frames", self_filter_frames_);
    get_parameter("self_filter.size_x", self_filter_size_x_);
    get_parameter("self_filter.size_y", self_filter_size_y_);
    get_parameter("self_filter.size_z", self_filter_size_z_);
    get_parameter("self_filter.offset_x", self_filter_off_x_);
    get_parameter("self_filter.offset_y", self_filter_off_y_);
    get_parameter("self_filter.offset_z", self_filter_off_z_);

    const std::size_t n = self_filter_frames_.size();
    if (self_filter_off_x_.empty()) self_filter_off_x_.assign(n, 0.0);
    if (self_filter_off_y_.empty()) self_filter_off_y_.assign(n, 0.0);
    if (self_filter_off_z_.empty()) self_filter_off_z_.assign(n, 0.0);
    const bool self_filter_lengths_ok =
        self_filter_size_x_.size() == n &&
        self_filter_size_y_.size() == n &&
        self_filter_size_z_.size() == n &&
        self_filter_off_x_.size() == n &&
        self_filter_off_y_.size() == n &&
        self_filter_off_z_.size() == n;
    if (self_filter_enabled_ && (n == 0 || !self_filter_lengths_ok)) {
      RCLCPP_ERROR(get_logger(),
                   "self_filter disabled: frames=%zu but size/offset lengths "
                   "do not match",
                   n);
      self_filter_enabled_ = false;
    }

    roi_filter_->setParameters(leaf_size_, mean_k_, stddev_mul_thresh_,
                               ground_height_, 0.05F, 1, 100000);

    error_pub_ =
        create_publisher<ApproachError>("/approach/lidar_error", qos_rel_);
    debug_pub_ =
        create_publisher<std_msgs::msg::String>("/approach/lidar_debug",
                                                qos_rel_);
    debug_cloud_pub_ =
        create_publisher<CloudMsg>("/approach/lidar_debug_cloud", qos_be_);
    front_cloud_pub_ =
        create_publisher<CloudMsg>("/approach/lidar_front_cloud", qos_be_);

    lidar_sub_ = create_subscription<CloudMsg>(
        lidar_topic_, qos_be_,
        std::bind(&LidarXDetector::lidarCallback, this,
                  std::placeholders::_1));

    const auto active_qos =
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    active_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/approach/active", active_qos,
        std::bind(&LidarXDetector::activeCallback, this,
                  std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "LidarXDetector ready. lidar_topic=%s",
                lidar_topic_.c_str());
  }

private:
  void activeCallback(const std_msgs::msg::Bool::SharedPtr msg) {
    is_active_.store(msg->data, std::memory_order_release);
    if (!msg->data) {
      publishInvalid("inactive");
    }
  }

  void lidarCallback(const CloudMsg::ConstSharedPtr &msg) {
    if (!is_active_.load(std::memory_order_acquire)) return;

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) {
      publishInvalid("empty input cloud");
      return;
    }

    roi_filter_->voxel_downsampling(cloud);

    geometry_msgs::msg::TransformStamped tf;
    if (!lookupTransform(target_frame_, msg->header.frame_id, msg->header.stamp,
                         tf)) {
      publishInvalid("input TF unavailable");
      return;
    }

    roi_filter_->remove_ground(cloud, tf.transform);
    applySpatialRoiBounds(cloud, roi_x_min_, roi_x_max_, roi_y_abs_near_,
                          roi_y_abs_max_, roi_z_max_);
    roi_filter_->remove_outliers(cloud);
    applySelfFilter(cloud, msg->header.stamp);
    if (cloud->empty()) {
      publishInvalid("empty after ROI/self_filter");
      return;
    }

    publishDebugCloud(cloud, debug_cloud_pub_);

    auto front_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*cloud);
    float representative_x = 0.0F;
    float representative_y = 0.0F;
    if (!keepFrontFraction(front_cloud, front_slice_ratio_, front_min_points_,
                           representative_x, representative_y)) {
      publishInvalid("front slice unavailable");
      return;
    }
    publishDebugCloud(front_cloud, front_cloud_pub_);

    ApproachError err;
    err.header.stamp = msg->header.stamp;
    err.valid = true;
    err.surface_distance_m = representative_x;
    err.x_error = representative_x - target_standoff_distance_;
    err.y_error = 0.0F;
    err.theta_error = 0.0F;
    err.yaw_valid = false;
    err.initial_dist_m = std::abs(err.x_error);
    err.mean_y_px = 0.0F;
    error_pub_->publish(err);

    std::ostringstream status;
    status << "valid source=lidar surface_x=" << representative_x
           << " legacy_x=" << err.x_error
           << " fallback_standoff=" << target_standoff_distance_
           << " front_points=" << front_cloud->size()
           << " debug_points=" << cloud->size();
    publishDebug(status.str());

    if (debug_log_) {
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 300,
          "lidar_error valid=1 surface_x=%.4fm legacy_x=%.4fm front=%zu",
          err.surface_distance_m, err.x_error, front_cloud->size());
    }
  }

  bool lookupTransform(const std::string &target, const std::string &source,
                       const builtin_interfaces::msg::Time &stamp,
                       geometry_msgs::msg::TransformStamped &tf_out) {
    try {
      tf_out = tf_buffer_.lookupTransform(
          target, source, rclcpp::Time(stamp),
          rclcpp::Duration::from_seconds(0.1));
      return true;
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "TF %s->%s failed: %s", source.c_str(),
                           target.c_str(), ex.what());
      return false;
    }
  }

  void applySelfFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                       const builtin_interfaces::msg::Time &stamp) {
    if (!self_filter_enabled_ || self_filter_frames_.empty() || !cloud ||
        cloud->empty()) {
      return;
    }

    struct Vol {
      Eigen::Affine3f t_link_base;
      Eigen::Vector3f half;
      Eigen::Vector3f center;
    };
    std::vector<Vol> vols;
    vols.reserve(self_filter_frames_.size());
    std::size_t skipped_tf = 0;
    for (std::size_t i = 0; i < self_filter_frames_.size(); ++i) {
      geometry_msgs::msg::TransformStamped tf;
      try {
        tf = tf_buffer_.lookupTransform(
            self_filter_frames_[i], target_frame_, rclcpp::Time(stamp),
            rclcpp::Duration::from_seconds(0.0));
      } catch (const tf2::TransformException &) {
        try {
          tf = tf_buffer_.lookupTransform(
              self_filter_frames_[i], target_frame_, tf2::TimePointZero,
              tf2::durationFromSec(0.0));
        } catch (const tf2::TransformException &) {
          ++skipped_tf;
          continue;
        }
      }
      Eigen::Affine3f t = Eigen::Affine3f::Identity();
      const auto &tr = tf.transform.translation;
      const auto &q = tf.transform.rotation;
      t.translation() << static_cast<float>(tr.x), static_cast<float>(tr.y),
          static_cast<float>(tr.z);
      t.linear() = Eigen::Quaternionf(
                       static_cast<float>(q.w), static_cast<float>(q.x),
                       static_cast<float>(q.y), static_cast<float>(q.z))
                       .toRotationMatrix();

      Vol v;
      v.t_link_base = t;
      v.half = Eigen::Vector3f(static_cast<float>(self_filter_size_x_[i]),
                               static_cast<float>(self_filter_size_y_[i]),
                               static_cast<float>(self_filter_size_z_[i])) *
                   0.5F +
               Eigen::Vector3f::Constant(self_filter_padding_);
      v.center = Eigen::Vector3f(static_cast<float>(self_filter_off_x_[i]),
                                 static_cast<float>(self_filter_off_y_[i]),
                                 static_cast<float>(self_filter_off_z_[i]));
      vols.push_back(v);
    }
    if (skipped_tf > 0 && debug_log_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "self_filter: skipped %zu/%zu volume TFs",
                           skipped_tf, self_filter_frames_.size());
    }
    if (vols.empty()) return;

    auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    out->points.reserve(cloud->points.size());
    for (const auto &p : cloud->points) {
      const Eigen::Vector3f pb(p.x, p.y, p.z);
      bool inside_any = false;
      for (const auto &v : vols) {
        const Eigen::Vector3f pl = v.t_link_base * pb - v.center;
        if (std::abs(pl.x()) <= v.half.x() &&
            std::abs(pl.y()) <= v.half.y() &&
            std::abs(pl.z()) <= v.half.z()) {
          inside_any = true;
          break;
        }
      }
      if (!inside_any) out->points.push_back(p);
    }
    out->width = static_cast<std::uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = false;
    cloud = out;
  }

  void publishDebugCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
      const rclcpp::Publisher<CloudMsg>::SharedPtr &publisher) {
    if (!publisher || publisher->get_subscription_count() == 0 || !cloud) {
      return;
    }
    CloudMsg out;
    pcl::toROSMsg(*cloud, out);
    out.header.stamp = now();
    out.header.frame_id = target_frame_;
    publisher->publish(out);
  }

  void publishInvalid(const std::string &reason) {
    ApproachError err;
    err.header.stamp = now();
    err.valid = false;
    error_pub_->publish(err);
    publishDebug("invalid:" + reason);
    if (debug_log_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "lidar_error valid=0 reason=%s", reason.c_str());
    }
  }

  void publishDebug(const std::string &status) {
    std_msgs::msg::String msg;
    msg.data = status;
    debug_pub_->publish(msg);
  }

  std::shared_ptr<Filter> roi_filter_;
  rclcpp::QoS qos_be_;
  rclcpp::QoS qos_rel_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  rclcpp::Subscription<CloudMsg>::SharedPtr lidar_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr active_sub_;
  rclcpp::Publisher<ApproachError>::SharedPtr error_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub_;
  rclcpp::Publisher<CloudMsg>::SharedPtr debug_cloud_pub_;
  rclcpp::Publisher<CloudMsg>::SharedPtr front_cloud_pub_;

  std::string lidar_topic_;
  std::string target_frame_;
  float roi_x_min_ = 0.1F;
  float roi_x_max_ = 3.0F;
  float roi_y_abs_near_ = 0.3F;
  float roi_y_abs_max_ = 0.8F;
  float roi_z_max_ = 1.5F;
  float leaf_size_ = 0.06F;
  int mean_k_ = 50;
  float stddev_mul_thresh_ = 0.5F;
  float ground_height_ = 0.1F;
  float target_standoff_distance_ = 0.3F;
  float front_slice_ratio_ = 0.05F;
  int front_min_points_ = 5;
  bool debug_log_ = true;
  std::atomic<bool> is_active_{false};

  bool self_filter_enabled_ = false;
  float self_filter_padding_ = 0.0F;
  std::vector<std::string> self_filter_frames_;
  std::vector<double> self_filter_size_x_;
  std::vector<double> self_filter_size_y_;
  std::vector<double> self_filter_size_z_;
  std::vector<double> self_filter_off_x_;
  std::vector<double> self_filter_off_y_;
  std::vector<double> self_filter_off_z_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarXDetector>());
  rclcpp::shutdown();
  return 0;
}
