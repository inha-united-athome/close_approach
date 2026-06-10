#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <pcl/common/centroid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include "close_approach/msg/close_approach_error.hpp"

namespace close_approach
{
namespace
{
using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

struct ClusterCandidate
{
  int id = -1;
  std::vector<int> indices;
  geometry_msgs::msg::Point centroid;
  geometry_msgs::msg::Point front_point;
  geometry_msgs::msg::Point align_point;
  float x_min = 0.0F;
  float x_max = 0.0F;
  float y_min = 0.0F;
  float y_max = 0.0F;
  float score = std::numeric_limits<float>::max();
  float confidence = 0.0F;
};

float clampFloat(const float value, const float min_value, const float max_value)
{
  return std::max(min_value, std::min(max_value, value));
}

float percentile(std::vector<float> values, const float ratio)
{
  if (values.empty()) {
    return 0.0F;
  }
  std::sort(values.begin(), values.end());
  const float clamped_ratio = clampFloat(ratio, 0.0F, 1.0F);
  const auto index = static_cast<std::size_t>(
    std::round(clamped_ratio * static_cast<float>(values.size() - 1)));
  return values[index];
}

geometry_msgs::msg::Point lerpPoint(
  const geometry_msgs::msg::Point & a,
  const geometry_msgs::msg::Point & b,
  const float alpha)
{
  geometry_msgs::msg::Point out;
  out.x = (1.0F - alpha) * a.x + alpha * b.x;
  out.y = (1.0F - alpha) * a.y + alpha * b.y;
  out.z = (1.0F - alpha) * a.z + alpha * b.z;
  return out;
}

float pointDistance2D(const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b)
{
  const auto dx = static_cast<float>(a.x - b.x);
  const auto dy = static_cast<float>(a.y - b.y);
  return std::hypot(dx, dy);
}

visualization_msgs::msg::Marker makeSphereMarker(
  const std_msgs::msg::Header & header,
  const std::string & ns,
  const int id,
  const geometry_msgs::msg::Point & p,
  const float r,
  const float g,
  const float b)
{
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position = p;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.08;
  marker.scale.y = 0.08;
  marker.scale.z = 0.08;
  marker.color.r = r;
  marker.color.g = g;
  marker.color.b = b;
  marker.color.a = 1.0;
  marker.lifetime = rclcpp::Duration::from_seconds(0.3);
  return marker;
}

visualization_msgs::msg::Marker makeTextMarker(
  const std_msgs::msg::Header & header,
  const std::string & text)
{
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.ns = "submap_close_approach_state";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = 0.6;
  marker.pose.position.y = 0.0;
  marker.pose.position.z = 1.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.16;
  marker.color.r = 1.0;
  marker.color.g = 1.0;
  marker.color.b = 1.0;
  marker.color.a = 1.0;
  marker.text = text;
  marker.lifetime = rclcpp::Duration::from_seconds(0.3);
  return marker;
}
}  // namespace

class SubmapCloseApproachPerceptionNode : public rclcpp::Node
{
public:
  SubmapCloseApproachPerceptionNode()
  : Node("submap_close_approach_perception_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    declareParameters();
    loadParameters();

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_cloud_topic_, rclcpp::SensorDataQoS(),
      std::bind(&SubmapCloseApproachPerceptionNode::cloudCallback, this, std::placeholders::_1));

    error_pub_ = create_publisher<close_approach::msg::CloseApproachError>(error_topic_, 10);
    preprocessed_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(preprocessed_cloud_topic_, 10);
    selected_cluster_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(selected_cluster_topic_, 10);
    front_band_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(front_band_topic_, 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);

    RCLCPP_INFO(
      get_logger(), "Submap close approach perception started. input=%s target_frame=%s",
      input_cloud_topic_.c_str(), target_frame_.c_str());
  }

private:
  enum class State
  {
    ACQUIRE,
    TRACK,
    LOST
  };

  void declareParameters()
  {
    declare_parameter<std::string>("input_cloud_topic", "/approach/submap_point");
    declare_parameter<std::string>("target_frame", "base_nav");
    declare_parameter<std::string>("error_topic", "/close_approach/error");
    declare_parameter<std::string>("preprocessed_cloud_topic", "/close_approach/preprocessed_cloud");
    declare_parameter<std::string>("selected_cluster_topic", "/close_approach/selected_cluster");
    declare_parameter<std::string>("front_band_topic", "/close_approach/front_band_cloud");
    declare_parameter<std::string>("marker_topic", "/close_approach/markers");

    declare_parameter<double>("max_cloud_age_sec", 0.6);
    declare_parameter<double>("roi_x_min", 0.20);
    declare_parameter<double>("roi_x_max", 2.00);
    declare_parameter<double>("roi_y_abs_max_acquire", 0.80);
    declare_parameter<double>("roi_y_abs_max_track", 0.45);
    declare_parameter<double>("roi_z_min", -0.20);
    declare_parameter<double>("roi_z_max", 1.50);

    declare_parameter<bool>("ground_remove_enable", true);
    declare_parameter<double>("ground_z_max", 0.08);
    declare_parameter<bool>("statistical_remove_enable", true);
    declare_parameter<int>("mean_k", 20);
    declare_parameter<double>("stddev_mul_thresh", 1.0);
    declare_parameter<bool>("radius_remove_enable", false);
    declare_parameter<double>("radius_search", 0.08);
    declare_parameter<int>("min_neighbors", 2);

    declare_parameter<double>("cluster_tolerance_xy", 0.15);
    declare_parameter<int>("min_cluster_size", 20);
    declare_parameter<int>("max_cluster_size", 8000);
    declare_parameter<double>("min_cluster_width", 0.08);
    declare_parameter<double>("min_cluster_depth", 0.04);

    declare_parameter<double>("front_percentile", 0.07);
    declare_parameter<double>("front_band_width", 0.15);
    declare_parameter<double>("desired_distance", 0.45);
    declare_parameter<double>("align_front_weight", 0.60);

    declare_parameter<double>("score_weight_y", 2.0);
    declare_parameter<double>("score_weight_x", 0.5);
    declare_parameter<double>("score_weight_area", 0.3);
    declare_parameter<double>("score_weight_locked_jump", 2.0);

    declare_parameter<double>("filter_alpha_far", 0.25);
    declare_parameter<double>("filter_alpha_near", 0.08);
    declare_parameter<double>("freeze_distance_margin", 0.10);
    declare_parameter<double>("jump_reject_distance", 0.30);
    declare_parameter<int>("lost_frame_threshold", 5);
    declare_parameter<int>("lock_frame_threshold", 2);
  }

  void loadParameters()
  {
    input_cloud_topic_ = get_parameter("input_cloud_topic").as_string();
    target_frame_ = get_parameter("target_frame").as_string();
    error_topic_ = get_parameter("error_topic").as_string();
    preprocessed_cloud_topic_ = get_parameter("preprocessed_cloud_topic").as_string();
    selected_cluster_topic_ = get_parameter("selected_cluster_topic").as_string();
    front_band_topic_ = get_parameter("front_band_topic").as_string();
    marker_topic_ = get_parameter("marker_topic").as_string();

    max_cloud_age_sec_ = get_parameter("max_cloud_age_sec").as_double();
    roi_x_min_ = get_parameter("roi_x_min").as_double();
    roi_x_max_ = get_parameter("roi_x_max").as_double();
    roi_y_abs_max_acquire_ = get_parameter("roi_y_abs_max_acquire").as_double();
    roi_y_abs_max_track_ = get_parameter("roi_y_abs_max_track").as_double();
    roi_z_min_ = get_parameter("roi_z_min").as_double();
    roi_z_max_ = get_parameter("roi_z_max").as_double();

    ground_remove_enable_ = get_parameter("ground_remove_enable").as_bool();
    ground_z_max_ = get_parameter("ground_z_max").as_double();
    statistical_remove_enable_ = get_parameter("statistical_remove_enable").as_bool();
    mean_k_ = get_parameter("mean_k").as_int();
    stddev_mul_thresh_ = get_parameter("stddev_mul_thresh").as_double();
    radius_remove_enable_ = get_parameter("radius_remove_enable").as_bool();
    radius_search_ = get_parameter("radius_search").as_double();
    min_neighbors_ = get_parameter("min_neighbors").as_int();

    cluster_tolerance_xy_ = get_parameter("cluster_tolerance_xy").as_double();
    min_cluster_size_ = get_parameter("min_cluster_size").as_int();
    max_cluster_size_ = get_parameter("max_cluster_size").as_int();
    min_cluster_width_ = get_parameter("min_cluster_width").as_double();
    min_cluster_depth_ = get_parameter("min_cluster_depth").as_double();

    front_percentile_ = get_parameter("front_percentile").as_double();
    front_band_width_ = get_parameter("front_band_width").as_double();
    desired_distance_ = get_parameter("desired_distance").as_double();
    align_front_weight_ = get_parameter("align_front_weight").as_double();

    score_weight_y_ = get_parameter("score_weight_y").as_double();
    score_weight_x_ = get_parameter("score_weight_x").as_double();
    score_weight_area_ = get_parameter("score_weight_area").as_double();
    score_weight_locked_jump_ = get_parameter("score_weight_locked_jump").as_double();

    filter_alpha_far_ = get_parameter("filter_alpha_far").as_double();
    filter_alpha_near_ = get_parameter("filter_alpha_near").as_double();
    freeze_distance_margin_ = get_parameter("freeze_distance_margin").as_double();
    jump_reject_distance_ = get_parameter("jump_reject_distance").as_double();
    lost_frame_threshold_ = get_parameter("lost_frame_threshold").as_int();
    lock_frame_threshold_ = get_parameter("lock_frame_threshold").as_int();
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0) {
      const double age = (now() - msg->header.stamp).seconds();
      if (age > max_cloud_age_sec_) {
        publishInvalid(msg->header, "STALE", "cloud is too old");
        return;
      }
    }

    sensor_msgs::msg::PointCloud2 transformed_msg;
    try {
      transformed_msg = tf_buffer_.transform(*msg, target_frame_, tf2::durationFromSec(0.05));
    } catch (const tf2::TransformException & ex) {
      publishInvalid(msg->header, "TF_FAIL", ex.what());
      return;
    }

    auto cloud = std::make_shared<CloudT>();
    pcl::fromROSMsg(transformed_msg, *cloud);
    auto filtered = preprocess(cloud);
    publishCloud(filtered, transformed_msg.header, preprocessed_pub_);

    std::vector<ClusterCandidate> candidates = buildClusters(filtered);
    if (candidates.empty()) {
      lost_count_++;
      if (lost_count_ >= lost_frame_threshold_) {
        state_ = State::LOST;
        has_lock_ = false;
      }
      publishInvalid(transformed_msg.header, stateName(), "no valid cluster");
      return;
    }

    auto selected = selectCluster(candidates);
    if (!selected.has_value()) {
      lost_count_++;
      publishInvalid(transformed_msg.header, stateName(), "no selected cluster");
      return;
    }

    updateTracking(*selected);
    publishResult(transformed_msg.header, filtered, *selected);
  }

  std::shared_ptr<CloudT> preprocess(const std::shared_ptr<CloudT> & input)
  {
    auto roi = std::make_shared<CloudT>();
    const double y_abs = state_ == State::TRACK && has_lock_ ? roi_y_abs_max_track_ : roi_y_abs_max_acquire_;

    for (const auto & p : input->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
        continue;
      }
      if (p.x < roi_x_min_ || p.x > roi_x_max_) {
        continue;
      }
      if (std::abs(p.y) > y_abs) {
        continue;
      }
      if (p.z < roi_z_min_ || p.z > roi_z_max_) {
        continue;
      }
      if (ground_remove_enable_ && p.z < ground_z_max_) {
        continue;
      }
      roi->points.push_back(p);
    }
    roi->width = static_cast<std::uint32_t>(roi->points.size());
    roi->height = 1;
    roi->is_dense = false;

    auto out = roi;
    if (statistical_remove_enable_ && static_cast<int>(out->size()) > mean_k_) {
      auto sor_out = std::make_shared<CloudT>();
      pcl::StatisticalOutlierRemoval<PointT> sor;
      sor.setInputCloud(out);
      sor.setMeanK(mean_k_);
      sor.setStddevMulThresh(stddev_mul_thresh_);
      sor.filter(*sor_out);
      out = sor_out;
    }

    if (radius_remove_enable_ && !out->empty()) {
      auto radius_out = std::make_shared<CloudT>();
      pcl::RadiusOutlierRemoval<PointT> radius;
      radius.setInputCloud(out);
      radius.setRadiusSearch(radius_search_);
      radius.setMinNeighborsInRadius(min_neighbors_);
      radius.filter(*radius_out);
      out = radius_out;
    }

    return out;
  }

  std::vector<ClusterCandidate> buildClusters(const std::shared_ptr<CloudT> & cloud)
  {
    std::vector<ClusterCandidate> clusters;
    const int n = static_cast<int>(cloud->points.size());
    if (n < min_cluster_size_) {
      return clusters;
    }

    std::vector<bool> visited(n, false);
    int cluster_id = 0;
    const float tol2 = static_cast<float>(cluster_tolerance_xy_ * cluster_tolerance_xy_);

    for (int i = 0; i < n; ++i) {
      if (visited[i]) {
        continue;
      }

      std::vector<int> queue;
      std::vector<int> indices;
      queue.push_back(i);
      visited[i] = true;

      for (std::size_t q = 0; q < queue.size(); ++q) {
        const int current = queue[q];
        indices.push_back(current);
        const auto & p = cloud->points[current];

        for (int j = 0; j < n; ++j) {
          if (visited[j]) {
            continue;
          }
          const auto & other = cloud->points[j];
          const float dx = p.x - other.x;
          const float dy = p.y - other.y;
          if ((dx * dx + dy * dy) <= tol2) {
            visited[j] = true;
            queue.push_back(j);
          }
        }
      }

      if (static_cast<int>(indices.size()) < min_cluster_size_ ||
        static_cast<int>(indices.size()) > max_cluster_size_)
      {
        continue;
      }

      auto candidate = makeCandidate(*cloud, indices, cluster_id++);
      const float width = candidate.y_max - candidate.y_min;
      const float depth = candidate.x_max - candidate.x_min;
      if (width < min_cluster_width_ && depth < min_cluster_depth_) {
        continue;
      }
      clusters.push_back(candidate);
    }

    return clusters;
  }

  ClusterCandidate makeCandidate(
    const CloudT & cloud,
    const std::vector<int> & indices,
    const int cluster_id)
  {
    ClusterCandidate c;
    c.id = cluster_id;
    c.indices = indices;
    c.x_min = std::numeric_limits<float>::max();
    c.y_min = std::numeric_limits<float>::max();
    c.x_max = std::numeric_limits<float>::lowest();
    c.y_max = std::numeric_limits<float>::lowest();

    std::vector<float> xs;
    xs.reserve(indices.size());
    geometry_msgs::msg::Point centroid;

    for (const int idx : indices) {
      const auto & p = cloud.points[idx];
      centroid.x += p.x;
      centroid.y += p.y;
      centroid.z += p.z;
      xs.push_back(p.x);
      c.x_min = std::min(c.x_min, p.x);
      c.x_max = std::max(c.x_max, p.x);
      c.y_min = std::min(c.y_min, p.y);
      c.y_max = std::max(c.y_max, p.y);
    }

    const double inv = 1.0 / static_cast<double>(indices.size());
    centroid.x *= inv;
    centroid.y *= inv;
    centroid.z *= inv;
    c.centroid = centroid;

    const float x_front = percentile(xs, static_cast<float>(front_percentile_));
    std::vector<float> front_xs;
    std::vector<float> front_ys;
    std::vector<float> front_zs;
    for (const int idx : indices) {
      const auto & p = cloud.points[idx];
      if (p.x >= x_front && p.x <= x_front + front_band_width_) {
        front_xs.push_back(p.x);
        front_ys.push_back(p.y);
        front_zs.push_back(p.z);
      }
    }
    if (front_xs.empty()) {
      front_xs = xs;
      for (const int idx : indices) {
        const auto & p = cloud.points[idx];
        front_ys.push_back(p.y);
        front_zs.push_back(p.z);
      }
    }

    c.front_point.x = percentile(front_xs, 0.5F);
    c.front_point.y = percentile(front_ys, 0.5F);
    c.front_point.z = percentile(front_zs, 0.5F);

    c.align_point = lerpPoint(c.centroid, c.front_point, static_cast<float>(align_front_weight_));

    const float area_proxy = std::max(0.01F, (c.x_max - c.x_min) * (c.y_max - c.y_min));
    c.score = static_cast<float>(score_weight_x_) * c.x_min +
      static_cast<float>(score_weight_y_) * std::abs(static_cast<float>(c.align_point.y)) -
      static_cast<float>(score_weight_area_) * area_proxy;
    if (has_lock_) {
      c.score += static_cast<float>(score_weight_locked_jump_) * pointDistance2D(c.align_point, locked_align_point_);
    }
    c.confidence = clampFloat(
      static_cast<float>(indices.size()) / static_cast<float>(std::max(1, min_cluster_size_ * 5)),
      0.0F, 1.0F);
    return c;
  }

  std::optional<ClusterCandidate> selectCluster(const std::vector<ClusterCandidate> & candidates)
  {
    if (candidates.empty()) {
      return std::nullopt;
    }

    std::optional<ClusterCandidate> best;
    for (const auto & c : candidates) {
      if (has_lock_ && pointDistance2D(c.align_point, locked_align_point_) > jump_reject_distance_) {
        continue;
      }
      if (!best || c.score < best->score) {
        best = c;
      }
    }
    return best;
  }

  void updateTracking(const ClusterCandidate & candidate)
  {
    lost_count_ = 0;
    lock_count_++;

    const double distance_margin = candidate.front_point.x - desired_distance_;
    const double alpha = distance_margin < freeze_distance_margin_ ? filter_alpha_near_ : filter_alpha_far_;

    if (!has_lock_ || lock_count_ <= lock_frame_threshold_) {
      locked_align_point_ = candidate.align_point;
      locked_front_point_ = candidate.front_point;
    } else {
      locked_align_point_ = lerpPoint(locked_align_point_, candidate.align_point, static_cast<float>(alpha));
      locked_front_point_ = lerpPoint(locked_front_point_, candidate.front_point, static_cast<float>(alpha));
    }

    has_lock_ = true;
    state_ = State::TRACK;
  }

  void publishResult(
    const std_msgs::msg::Header & header,
    const std::shared_ptr<CloudT> & filtered,
    const ClusterCandidate & selected)
  {
    auto cluster_cloud = std::make_shared<CloudT>();
    auto front_cloud = std::make_shared<CloudT>();

    const float x_front = static_cast<float>(selected.front_point.x);
    for (const int idx : selected.indices) {
      cluster_cloud->points.push_back(filtered->points[idx]);
      const auto & p = filtered->points[idx];
      if (p.x >= x_front && p.x <= x_front + front_band_width_) {
        front_cloud->points.push_back(p);
      }
    }
    cluster_cloud->width = static_cast<std::uint32_t>(cluster_cloud->points.size());
    cluster_cloud->height = 1;
    cluster_cloud->is_dense = false;
    front_cloud->width = static_cast<std::uint32_t>(front_cloud->points.size());
    front_cloud->height = 1;
    front_cloud->is_dense = false;

    publishCloud(cluster_cloud, header, selected_cluster_pub_);
    publishCloud(front_cloud, header, front_band_pub_);

    close_approach::msg::CloseApproachError error;
    error.header = header;
    error.valid = true;
    error.state = stateName();
    error.reason = "tracking";
    error.align_point = locked_align_point_;
    error.front_point = locked_front_point_;
    error.cluster_centroid = selected.centroid;
    error.x_error = static_cast<float>(locked_front_point_.x - desired_distance_);
    error.y_error = static_cast<float>(locked_align_point_.y);
    error.theta_error = static_cast<float>(std::atan2(locked_align_point_.y, locked_align_point_.x));
    error.confidence = selected.confidence;
    error.cluster_id = selected.id;
    error.cluster_size = static_cast<int>(selected.indices.size());
    error_pub_->publish(error);

    visualization_msgs::msg::MarkerArray markers;
    markers.markers.push_back(makeSphereMarker(header, "p_align", 0, locked_align_point_, 0.0F, 1.0F, 0.0F));
    markers.markers.push_back(makeSphereMarker(header, "p_front", 1, locked_front_point_, 1.0F, 0.3F, 0.0F));
    markers.markers.push_back(makeTextMarker(
      header, stateName() + " x=" + std::to_string(error.x_error) +
      " y=" + std::to_string(error.y_error) +
      " th=" + std::to_string(error.theta_error)));
    marker_pub_->publish(markers);
  }

  void publishInvalid(
    const std_msgs::msg::Header & input_header,
    const std::string & state,
    const std::string & reason)
  {
    close_approach::msg::CloseApproachError error;
    error.header = input_header;
    error.header.frame_id = target_frame_;
    error.valid = false;
    error.state = state;
    error.reason = reason;
    error_pub_->publish(error);
  }

  void publishCloud(
    const std::shared_ptr<CloudT> & cloud,
    const std_msgs::msg::Header & header,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub)
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header = header;
    pub->publish(msg);
  }

  std::string stateName() const
  {
    switch (state_) {
      case State::ACQUIRE:
        return "ACQUIRE";
      case State::TRACK:
        return "TRACK";
      case State::LOST:
        return "LOST";
    }
    return "UNKNOWN";
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<close_approach::msg::CloseApproachError>::SharedPtr error_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr preprocessed_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr selected_cluster_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr front_band_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string input_cloud_topic_;
  std::string target_frame_;
  std::string error_topic_;
  std::string preprocessed_cloud_topic_;
  std::string selected_cluster_topic_;
  std::string front_band_topic_;
  std::string marker_topic_;

  double max_cloud_age_sec_ = 0.6;
  double roi_x_min_ = 0.2;
  double roi_x_max_ = 2.0;
  double roi_y_abs_max_acquire_ = 0.8;
  double roi_y_abs_max_track_ = 0.45;
  double roi_z_min_ = -0.2;
  double roi_z_max_ = 1.5;

  bool ground_remove_enable_ = true;
  double ground_z_max_ = 0.08;
  bool statistical_remove_enable_ = true;
  int mean_k_ = 20;
  double stddev_mul_thresh_ = 1.0;
  bool radius_remove_enable_ = false;
  double radius_search_ = 0.08;
  int min_neighbors_ = 2;

  double cluster_tolerance_xy_ = 0.15;
  int min_cluster_size_ = 20;
  int max_cluster_size_ = 8000;
  double min_cluster_width_ = 0.08;
  double min_cluster_depth_ = 0.04;

  double front_percentile_ = 0.07;
  double front_band_width_ = 0.15;
  double desired_distance_ = 0.45;
  double align_front_weight_ = 0.6;

  double score_weight_y_ = 2.0;
  double score_weight_x_ = 0.5;
  double score_weight_area_ = 0.3;
  double score_weight_locked_jump_ = 2.0;

  double filter_alpha_far_ = 0.25;
  double filter_alpha_near_ = 0.08;
  double freeze_distance_margin_ = 0.10;
  double jump_reject_distance_ = 0.30;
  int lost_frame_threshold_ = 5;
  int lock_frame_threshold_ = 2;

  State state_ = State::ACQUIRE;
  bool has_lock_ = false;
  int lock_count_ = 0;
  int lost_count_ = 0;
  geometry_msgs::msg::Point locked_align_point_;
  geometry_msgs::msg::Point locked_front_point_;
};
}  // namespace close_approach

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<close_approach::SubmapCloseApproachPerceptionNode>());
  rclcpp::shutdown();
  return 0;
}
