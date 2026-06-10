#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include "close_approach/msg/close_approach_error.hpp"

namespace close_approach
{
namespace
{
using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

struct Candidate
{
  int id = -1;
  std::vector<int> indices;
  geometry_msgs::msg::Point centroid;
  geometry_msgs::msg::Point front;
  geometry_msgs::msg::Point align;
  float xmin = 0.0F;
  float xmax = 0.0F;
  float ymin = 0.0F;
  float ymax = 0.0F;
  float score = std::numeric_limits<float>::max();
  float confidence = 0.0F;
};

float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }

float percentile(std::vector<float> values, float ratio)
{
  if (values.empty()) { return 0.0F; }
  std::sort(values.begin(), values.end());
  const auto idx = static_cast<std::size_t>(
    std::round(clampf(ratio, 0.0F, 1.0F) * static_cast<float>(values.size() - 1)));
  return values[idx];
}

geometry_msgs::msg::Point blend(
  const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b, float alpha)
{
  geometry_msgs::msg::Point p;
  p.x = (1.0F - alpha) * a.x + alpha * b.x;
  p.y = (1.0F - alpha) * a.y + alpha * b.y;
  p.z = (1.0F - alpha) * a.z + alpha * b.z;
  return p;
}

float dist2d(const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b)
{
  return std::hypot(static_cast<float>(a.x - b.x), static_cast<float>(a.y - b.y));
}

visualization_msgs::msg::Marker sphere(
  const std_msgs::msg::Header & h, const std::string & ns, int id,
  const geometry_msgs::msg::Point & p, float r, float g, float b)
{
  visualization_msgs::msg::Marker m;
  m.header = h;
  m.ns = ns;
  m.id = id;
  m.type = visualization_msgs::msg::Marker::SPHERE;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.pose.position = p;
  m.pose.orientation.w = 1.0;
  m.scale.x = 0.08;
  m.scale.y = 0.08;
  m.scale.z = 0.08;
  m.color.r = r;
  m.color.g = g;
  m.color.b = b;
  m.color.a = 1.0;
  m.lifetime = rclcpp::Duration::from_seconds(0.3);
  return m;
}
}  // namespace

class SubmapCloseApproachPerceptionNode : public rclcpp::Node
{
public:
  SubmapCloseApproachPerceptionNode()
  : Node("submap_close_approach_perception_node"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_)
  {
    declareParams();
    loadParams();

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_cloud_topic_, rclcpp::SensorDataQoS(),
      std::bind(&SubmapCloseApproachPerceptionNode::onCloud, this, std::placeholders::_1));
    error_pub_ = create_publisher<close_approach::msg::CloseApproachError>(error_topic_, 10);
    preprocessed_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(preprocessed_cloud_topic_, 10);
    selected_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(selected_cluster_topic_, 10);
    front_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(front_band_topic_, 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);

    RCLCPP_INFO(get_logger(), "submap close approach perception: %s", input_cloud_topic_.c_str());
  }

private:
  void declareParams()
  {
    declare_parameter<std::string>("input_cloud_topic", "/approach/submap_point");
    declare_parameter<std::string>("target_frame", "base_nav");
    declare_parameter<std::string>("error_topic", "/close_approach/error");
    declare_parameter<std::string>("preprocessed_cloud_topic", "/close_approach/preprocessed_cloud");
    declare_parameter<std::string>("selected_cluster_topic", "/close_approach/selected_cluster");
    declare_parameter<std::string>("front_band_topic", "/close_approach/front_band_cloud");
    declare_parameter<std::string>("marker_topic", "/close_approach/markers");
    declare_parameter<double>("max_cloud_age_sec", 0.6);
    declare_parameter<double>("roi_x_min", 0.2);
    declare_parameter<double>("roi_x_max", 2.0);
    declare_parameter<double>("roi_y_abs_max_acquire", 0.8);
    declare_parameter<double>("roi_y_abs_max_track", 0.45);
    declare_parameter<double>("roi_z_min", -0.2);
    declare_parameter<double>("roi_z_max", 1.5);
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
    declare_parameter<double>("align_front_weight", 0.6);
    declare_parameter<double>("score_weight_y", 2.0);
    declare_parameter<double>("score_weight_x", 0.5);
    declare_parameter<double>("score_weight_area", 0.3);
    declare_parameter<double>("score_weight_locked_jump", 2.0);
    declare_parameter<double>("filter_alpha_far", 0.25);
    declare_parameter<double>("filter_alpha_near", 0.08);
    declare_parameter<double>("freeze_distance_margin", 0.1);
    declare_parameter<double>("jump_reject_distance", 0.3);
    declare_parameter<int>("lost_frame_threshold", 5);
    declare_parameter<int>("lock_frame_threshold", 2);
  }

  void loadParams()
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

  void onCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if ((msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0) &&
      (now() - msg->header.stamp).seconds() > max_cloud_age_sec_)
    {
      publishInvalid(msg->header, "STALE", "cloud is too old");
      return;
    }

    sensor_msgs::msg::PointCloud2 transformed;
    try {
      transformed = tf_buffer_.transform(*msg, target_frame_, tf2::durationFromSec(0.05));
    } catch (const tf2::TransformException & ex) {
      publishInvalid(msg->header, "TF_FAIL", ex.what());
      return;
    }

    auto cloud = std::make_shared<CloudT>();
    pcl::fromROSMsg(transformed, *cloud);
    auto filtered = preprocess(cloud);
    publishCloud(filtered, transformed.header, preprocessed_pub_);

    auto candidates = cluster(filtered);
    auto selected = select(candidates);
    if (!selected) {
      lost_count_++;
      if (lost_count_ >= lost_frame_threshold_) {
        state_ = "LOST";
        has_lock_ = false;
      }
      publishInvalid(transformed.header, state_, "no selected cluster");
      return;
    }

    updateLock(*selected);
    publishResult(transformed.header, filtered, *selected);
  }

  std::shared_ptr<CloudT> preprocess(const std::shared_ptr<CloudT> & input)
  {
    auto out = std::make_shared<CloudT>();
    const double y_abs = has_lock_ ? roi_y_abs_max_track_ : roi_y_abs_max_acquire_;
    for (const auto & p : input->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) { continue; }
      if (p.x < roi_x_min_ || p.x > roi_x_max_ || std::abs(p.y) > y_abs) { continue; }
      if (p.z < roi_z_min_ || p.z > roi_z_max_) { continue; }
      if (ground_remove_enable_ && p.z < ground_z_max_) { continue; }
      out->points.push_back(p);
    }
    out->width = static_cast<std::uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = false;

    if (statistical_remove_enable_ && static_cast<int>(out->size()) > mean_k_) {
      auto tmp = std::make_shared<CloudT>();
      pcl::StatisticalOutlierRemoval<PointT> sor;
      sor.setInputCloud(out);
      sor.setMeanK(mean_k_);
      sor.setStddevMulThresh(stddev_mul_thresh_);
      sor.filter(*tmp);
      out = tmp;
    }
    if (radius_remove_enable_ && !out->empty()) {
      auto tmp = std::make_shared<CloudT>();
      pcl::RadiusOutlierRemoval<PointT> ror;
      ror.setInputCloud(out);
      ror.setRadiusSearch(radius_search_);
      ror.setMinNeighborsInRadius(min_neighbors_);
      ror.filter(*tmp);
      out = tmp;
    }
    return out;
  }

  std::vector<Candidate> cluster(const std::shared_ptr<CloudT> & cloud)
  {
    std::vector<Candidate> out;
    const int n = static_cast<int>(cloud->size());
    if (n < min_cluster_size_) { return out; }

    std::vector<bool> visited(n, false);
    const float tol2 = static_cast<float>(cluster_tolerance_xy_ * cluster_tolerance_xy_);
    int id = 0;

    for (int i = 0; i < n; ++i) {
      if (visited[i]) { continue; }
      std::queue<int> q;
      std::vector<int> indices;
      visited[i] = true;
      q.push(i);
      while (!q.empty()) {
        const int cur = q.front();
        q.pop();
        indices.push_back(cur);
        const auto & p = cloud->points[cur];
        for (int j = 0; j < n; ++j) {
          if (visited[j]) { continue; }
          const auto & o = cloud->points[j];
          const float dx = p.x - o.x;
          const float dy = p.y - o.y;
          if (dx * dx + dy * dy <= tol2) {
            visited[j] = true;
            q.push(j);
          }
        }
      }
      if (static_cast<int>(indices.size()) < min_cluster_size_ ||
        static_cast<int>(indices.size()) > max_cluster_size_)
      {
        continue;
      }
      Candidate c = makeCandidate(*cloud, indices, id++);
      if ((c.ymax - c.ymin) < min_cluster_width_ && (c.xmax - c.xmin) < min_cluster_depth_) {
        continue;
      }
      out.push_back(c);
    }
    return out;
  }

  Candidate makeCandidate(const CloudT & cloud, const std::vector<int> & indices, int id)
  {
    Candidate c;
    c.id = id;
    c.indices = indices;
    c.xmin = c.ymin = std::numeric_limits<float>::max();
    c.xmax = c.ymax = std::numeric_limits<float>::lowest();
    std::vector<float> xs;
    std::vector<float> fx;
    std::vector<float> fy;
    std::vector<float> fz;
    xs.reserve(indices.size());

    for (int idx : indices) {
      const auto & p = cloud.points[idx];
      c.centroid.x += p.x;
      c.centroid.y += p.y;
      c.centroid.z += p.z;
      xs.push_back(p.x);
      c.xmin = std::min(c.xmin, p.x);
      c.xmax = std::max(c.xmax, p.x);
      c.ymin = std::min(c.ymin, p.y);
      c.ymax = std::max(c.ymax, p.y);
    }
    const double inv = 1.0 / static_cast<double>(indices.size());
    c.centroid.x *= inv;
    c.centroid.y *= inv;
    c.centroid.z *= inv;

    const float x_front = percentile(xs, static_cast<float>(front_percentile_));
    for (int idx : indices) {
      const auto & p = cloud.points[idx];
      if (p.x >= x_front && p.x <= x_front + front_band_width_) {
        fx.push_back(p.x);
        fy.push_back(p.y);
        fz.push_back(p.z);
      }
    }
    if (fx.empty()) {
      for (int idx : indices) {
        const auto & p = cloud.points[idx];
        fx.push_back(p.x);
        fy.push_back(p.y);
        fz.push_back(p.z);
      }
    }
    c.front.x = percentile(fx, 0.5F);
    c.front.y = percentile(fy, 0.5F);
    c.front.z = percentile(fz, 0.5F);
    c.align = blend(c.centroid, c.front, static_cast<float>(align_front_weight_));

    const float area = std::max(0.01F, (c.xmax - c.xmin) * (c.ymax - c.ymin));
    c.score = static_cast<float>(score_weight_x_) * c.xmin +
      static_cast<float>(score_weight_y_) * std::abs(static_cast<float>(c.align.y)) -
      static_cast<float>(score_weight_area_) * area;
    if (has_lock_) {
      c.score += static_cast<float>(score_weight_locked_jump_) * dist2d(c.align, locked_align_);
    }
    c.confidence = clampf(
      static_cast<float>(indices.size()) / static_cast<float>(std::max(1, min_cluster_size_ * 5)),
      0.0F, 1.0F);
    return c;
  }

  std::optional<Candidate> select(const std::vector<Candidate> & candidates) const
  {
    std::optional<Candidate> best;
    for (const auto & c : candidates) {
      if (has_lock_ && dist2d(c.align, locked_align_) > jump_reject_distance_) { continue; }
      if (!best || c.score < best->score) { best = c; }
    }
    return best;
  }

  void updateLock(const Candidate & c)
  {
    lost_count_ = 0;
    lock_count_++;
    const double margin = c.front.x - desired_distance_;
    const float alpha = static_cast<float>(margin < freeze_distance_margin_ ? filter_alpha_near_ : filter_alpha_far_);
    if (!has_lock_ || lock_count_ <= lock_frame_threshold_) {
      locked_align_ = c.align;
      locked_front_ = c.front;
    } else {
      locked_align_ = blend(locked_align_, c.align, alpha);
      locked_front_ = blend(locked_front_, c.front, alpha);
    }
    has_lock_ = true;
    state_ = "TRACK";
  }

  void publishResult(
    const std_msgs::msg::Header & h, const std::shared_ptr<CloudT> & cloud, const Candidate & c)
  {
    auto selected = std::make_shared<CloudT>();
    auto front = std::make_shared<CloudT>();
    for (int idx : c.indices) {
      const auto & p = cloud->points[idx];
      selected->points.push_back(p);
      if (p.x >= c.front.x && p.x <= c.front.x + front_band_width_) { front->points.push_back(p); }
    }
    selected->width = static_cast<std::uint32_t>(selected->size());
    selected->height = 1;
    front->width = static_cast<std::uint32_t>(front->size());
    front->height = 1;
    publishCloud(selected, h, selected_pub_);
    publishCloud(front, h, front_pub_);

    close_approach::msg::CloseApproachError e;
    e.header = h;
    e.valid = true;
    e.state = state_;
    e.reason = "tracking";
    e.align_point = locked_align_;
    e.front_point = locked_front_;
    e.cluster_centroid = c.centroid;
    e.x_error = static_cast<float>(locked_front_.x - desired_distance_);
    e.y_error = static_cast<float>(locked_align_.y);
    e.theta_error = static_cast<float>(std::atan2(locked_align_.y, locked_align_.x));
    e.confidence = c.confidence;
    e.cluster_id = c.id;
    e.cluster_size = static_cast<int>(c.indices.size());
    error_pub_->publish(e);

    visualization_msgs::msg::MarkerArray markers;
    markers.markers.push_back(sphere(h, "p_align", 0, locked_align_, 0.0F, 1.0F, 0.0F));
    markers.markers.push_back(sphere(h, "p_front", 1, locked_front_, 1.0F, 0.3F, 0.0F));
    marker_pub_->publish(markers);
  }

  void publishInvalid(const std_msgs::msg::Header & h, const std::string & state, const std::string & reason)
  {
    close_approach::msg::CloseApproachError e;
    e.header = h;
    e.header.frame_id = target_frame_;
    e.valid = false;
    e.state = state;
    e.reason = reason;
    error_pub_->publish(e);
  }

  void publishCloud(
    const std::shared_ptr<CloudT> & cloud, const std_msgs::msg::Header & h,
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub) const
  {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header = h;
    pub->publish(msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<close_approach::msg::CloseApproachError>::SharedPtr error_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr preprocessed_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr selected_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr front_pub_;
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
  double freeze_distance_margin_ = 0.1;
  double jump_reject_distance_ = 0.3;
  int lost_frame_threshold_ = 5;
  int lock_frame_threshold_ = 2;
  std::string state_ = "ACQUIRE";
  bool has_lock_ = false;
  int lock_count_ = 0;
  int lost_count_ = 0;
  geometry_msgs::msg::Point locked_align_;
  geometry_msgs::msg::Point locked_front_;
};
}  // namespace close_approach

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<close_approach::SubmapCloseApproachPerceptionNode>());
  rclcpp::shutdown();
  return 0;
}
