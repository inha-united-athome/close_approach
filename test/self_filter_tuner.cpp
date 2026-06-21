// Live tuning helper for the pc_detector robot self-filter boxes.
//
// It mirrors PCDetector::applySelfFilter but is standalone so the boxes can be
// adjusted in real time without rebuilding or driving the robot:
//
//   1) run this node (it subscribes the same cloud_topic)
//   2) open RViz, add /self_filter_tuner/input, /self_filter_tuner/filtered,
//      and the /self_filter_tuner/boxes MarkerArray
//   3) tweak params live, e.g.
//        ros2 param set /self_filter_tuner self_filter.padding 0.05
//        ros2 param set /self_filter_tuner self_filter.size_x "[0.5, 0.44, ...]"
//   4) save with:  ros2 param dump /self_filter_tuner
//      then copy the self_filter.* block into config/pc_detector.yaml
//
// Boxes are drawn as CUBE markers in each link frame, so RViz places them with
// live TF (a moving arm is shown correctly).

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/msg/transform.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

class SelfFilterTuner : public rclcpp::Node {
public:
  SelfFilterTuner()
      : Node("self_filter_tuner"),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {
    cloud_topic_  = declare_parameter<std::string>(
        "cloud_topic", "/camera/camera_head/depth/color/points");
    target_frame_ = declare_parameter<std::string>("target_frame", "base_nav");
    leaf_size_    = declare_parameter<double>("leaf_size", 0.03);
    enabled_      = declare_parameter<bool>("self_filter.enabled", true);
    padding_      = declare_parameter<double>("self_filter.padding", 0.03);
    frames_   = declare_parameter<std::vector<std::string>>(
        "self_filter.frames", std::vector<std::string>{});
    size_x_   = declare_parameter<std::vector<double>>("self_filter.size_x", {});
    size_y_   = declare_parameter<std::vector<double>>("self_filter.size_y", {});
    size_z_   = declare_parameter<std::vector<double>>("self_filter.size_z", {});
    offset_x_ = declare_parameter<std::vector<double>>("self_filter.offset_x", {});
    offset_y_ = declare_parameter<std::vector<double>>("self_filter.offset_y", {});
    offset_z_ = declare_parameter<std::vector<double>>("self_filter.offset_z", {});
    fixupOffsets();

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&SelfFilterTuner::cloudCb, this, std::placeholders::_1));
    in_pub_  = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/self_filter_tuner/input", 1);
    out_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/self_filter_tuner/filtered", 1);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/self_filter_tuner/boxes", 1);

    param_cb_ = add_on_set_parameters_callback(
        std::bind(&SelfFilterTuner::onSetParam, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
                "self_filter_tuner ready. cloud=%s target=%s boxes=%zu",
                cloud_topic_.c_str(), target_frame_.c_str(), frames_.size());
  }

private:
  static Eigen::Affine3f toAffine(const geometry_msgs::msg::Transform &t) {
    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    T.translation() << static_cast<float>(t.translation.x),
        static_cast<float>(t.translation.y),
        static_cast<float>(t.translation.z);
    T.linear() = Eigen::Quaternionf(static_cast<float>(t.rotation.w),
                                    static_cast<float>(t.rotation.x),
                                    static_cast<float>(t.rotation.y),
                                    static_cast<float>(t.rotation.z))
                     .toRotationMatrix();
    return T;
  }

  void fixupOffsets() {
    const std::size_t n = frames_.size();
    if (offset_x_.size() != n) offset_x_.assign(n, 0.0);
    if (offset_y_.size() != n) offset_y_.assign(n, 0.0);
    if (offset_z_.size() != n) offset_z_.assign(n, 0.0);
  }

  bool lengthsOk() const {
    const std::size_t n = frames_.size();
    return size_x_.size() == n && size_y_.size() == n && size_z_.size() == n &&
           offset_x_.size() == n && offset_y_.size() == n &&
           offset_z_.size() == n;
  }

  rcl_interfaces::msg::SetParametersResult onSetParam(
      const std::vector<rclcpp::Parameter> &params) {
    for (const auto &p : params) {
      const std::string &name = p.get_name();
      if (name == "self_filter.enabled")      enabled_ = p.as_bool();
      else if (name == "self_filter.padding") padding_ = p.as_double();
      else if (name == "leaf_size")           leaf_size_ = p.as_double();
      else if (name == "self_filter.frames")   frames_   = p.as_string_array();
      else if (name == "self_filter.size_x")   size_x_   = p.as_double_array();
      else if (name == "self_filter.size_y")   size_y_   = p.as_double_array();
      else if (name == "self_filter.size_z")   size_z_   = p.as_double_array();
      else if (name == "self_filter.offset_x") offset_x_ = p.as_double_array();
      else if (name == "self_filter.offset_y") offset_y_ = p.as_double_array();
      else if (name == "self_filter.offset_z") offset_z_ = p.as_double_array();
    }
    fixupOffsets();
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    if (!lengthsOk()) {
      result.successful = false;  // reject inconsistent array edits
      result.reason = "self_filter size/offset array lengths must match frames";
    }
    return result;
  }

  void publishCloud(
      const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
      const rclcpp::Time &stamp) {
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*cloud, out);
    out.header.stamp = stamp;
    out.header.frame_id = target_frame_;
    pub->publish(out);
  }

  void publishMarkers(const rclcpp::Time &stamp) {
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker del;
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(del);
    if (lengthsOk()) {
      for (std::size_t i = 0; i < frames_.size(); ++i) {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = frames_[i];
        m.header.stamp = stamp;
        m.ns = "self_filter";
        m.id = static_cast<int>(i);
        m.type = visualization_msgs::msg::Marker::CUBE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = offset_x_[i];
        m.pose.position.y = offset_y_[i];
        m.pose.position.z = offset_z_[i];
        m.pose.orientation.w = 1.0;
        m.scale.x = std::max(1e-3, size_x_[i] + 2.0 * padding_);
        m.scale.y = std::max(1e-3, size_y_[i] + 2.0 * padding_);
        m.scale.z = std::max(1e-3, size_z_[i] + 2.0 * padding_);
        m.color.r = 0.2f;
        m.color.g = 0.6f;
        m.color.b = 1.0f;
        m.color.a = 0.3f;
        arr.markers.push_back(m);
      }
    }
    marker_pub_->publish(arr);
  }

  std::size_t applySelfFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr &kept,
                              const rclcpp::Time &stamp) {
    kept = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (!enabled_ || frames_.empty() || !lengthsOk()) {
      *kept = *in;
      return 0;
    }
    struct Vol {
      Eigen::Affine3f T_link_base;
      Eigen::Vector3f half;
      Eigen::Vector3f center;
    };
    std::vector<Vol> vols;
    vols.reserve(frames_.size());
    for (std::size_t i = 0; i < frames_.size(); ++i) {
      geometry_msgs::msg::TransformStamped tf;
      try {
        tf = tf_buffer_.lookupTransform(frames_[i], target_frame_, stamp,
                                        rclcpp::Duration::from_seconds(0.05));
      } catch (const tf2::TransformException &e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "self_filter TF %s<-%s unavailable: %s",
                             frames_[i].c_str(), target_frame_.c_str(),
                             e.what());
        continue;
      }
      Vol v;
      v.T_link_base = toAffine(tf.transform);
      v.half = Eigen::Vector3f(static_cast<float>(size_x_[i]),
                               static_cast<float>(size_y_[i]),
                               static_cast<float>(size_z_[i])) *
                   0.5F +
               Eigen::Vector3f::Constant(static_cast<float>(padding_));
      v.center = Eigen::Vector3f(static_cast<float>(offset_x_[i]),
                                 static_cast<float>(offset_y_[i]),
                                 static_cast<float>(offset_z_[i]));
      vols.push_back(v);
    }

    kept->points.reserve(in->points.size());
    std::size_t removed = 0;
    for (const auto &p : in->points) {
      const Eigen::Vector3f pb(p.x, p.y, p.z);
      bool inside = false;
      for (const auto &v : vols) {
        const Eigen::Vector3f pl = v.T_link_base * pb - v.center;
        if (std::abs(pl.x()) <= v.half.x() && std::abs(pl.y()) <= v.half.y() &&
            std::abs(pl.z()) <= v.half.z()) {
          inside = true;
          break;
        }
      }
      if (inside) ++removed;
      else kept->points.push_back(p);
    }
    kept->width = static_cast<std::uint32_t>(kept->points.size());
    kept->height = 1;
    kept->is_dense = false;
    return removed;
  }

  void cloudCb(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) return;

    if (leaf_size_ > 1e-4) {
      pcl::VoxelGrid<pcl::PointXYZ> vg;
      vg.setInputCloud(cloud);
      vg.setLeafSize(static_cast<float>(leaf_size_), static_cast<float>(leaf_size_),
                     static_cast<float>(leaf_size_));
      auto ds = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      vg.filter(*ds);
      cloud = ds;
    }

    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_.lookupTransform(target_frame_, msg->header.frame_id,
                                      msg->header.stamp,
                                      rclcpp::Duration::from_seconds(0.1));
    } catch (const tf2::TransformException &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "TF %s<-%s unavailable: %s", target_frame_.c_str(),
                           msg->header.frame_id.c_str(), e.what());
      return;
    }
    auto base_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::transformPointCloud(*cloud, *base_cloud, toAffine(tf.transform));

    publishCloud(in_pub_, base_cloud, msg->header.stamp);

    pcl::PointCloud<pcl::PointXYZ>::Ptr kept;
    const std::size_t removed = applySelfFilter(base_cloud, kept, msg->header.stamp);
    publishCloud(out_pub_, kept, msg->header.stamp);
    publishMarkers(msg->header.stamp);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "self_filter: removed %zu / %zu pts (%zu boxes)",
                         removed, base_cloud->points.size(), frames_.size());
  }

  std::string cloud_topic_, target_frame_;
  double leaf_size_ = 0.03;
  bool enabled_ = true;
  double padding_ = 0.03;
  std::vector<std::string> frames_;
  std::vector<double> size_x_, size_y_, size_z_;
  std::vector<double> offset_x_, offset_y_, offset_z_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr in_pub_, out_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SelfFilterTuner>());
  rclcpp::shutdown();
  return 0;
}
