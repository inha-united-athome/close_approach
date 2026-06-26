#include "inha_interfaces/action/y_decider.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>

namespace {

constexpr float kPi = 3.14159265358979323846F;

using CloudMsg = sensor_msgs::msg::PointCloud2;
using CameraInfoMsg = sensor_msgs::msg::CameraInfo;
using ImageMsg = sensor_msgs::msg::CompressedImage;

struct Edge2D {
  Eigen::Vector2f center{0.0F, 0.0F};
  Eigen::Vector2f axis{0.0F, 1.0F};
  float length = 0.0F;
};

struct OBB2D {
  Eigen::Vector2f center{0.0F, 0.0F};
  Eigen::Vector2f axis1{1.0F, 0.0F};
  Eigen::Vector2f axis2{0.0F, 1.0F};
  float length1 = 0.0F;
  float length2 = 0.0F;
};

struct FitResult {
  bool valid = false;
  OBB2D obb;
  std::array<Edge2D, 4> edges;
};

Eigen::Vector2f normalizedOr(const Eigen::Vector2f &v,
                             const Eigen::Vector2f &fallback) {
  const float n = v.norm();
  if (!std::isfinite(n) || n < 1e-6F) {
    return fallback;
  }
  return v / n;
}

OBB2D obbFromAxes(const std::vector<Eigen::Vector2f> &points,
                  Eigen::Vector2f axis1) {
  OBB2D obb;
  obb.axis1 = normalizedOr(axis1, Eigen::Vector2f(1.0F, 0.0F));
  obb.axis2 = Eigen::Vector2f(-obb.axis1.y(), obb.axis1.x());

  float min1 = std::numeric_limits<float>::max();
  float max1 = -std::numeric_limits<float>::max();
  float min2 = std::numeric_limits<float>::max();
  float max2 = -std::numeric_limits<float>::max();
  for (const auto &p : points) {
    const float s1 = p.dot(obb.axis1);
    const float s2 = p.dot(obb.axis2);
    min1 = std::min(min1, s1);
    max1 = std::max(max1, s1);
    min2 = std::min(min2, s2);
    max2 = std::max(max2, s2);
  }

  obb.length1 = std::max(0.0F, max1 - min1);
  obb.length2 = std::max(0.0F, max2 - min2);
  obb.center = ((min1 + max1) * 0.5F) * obb.axis1 +
               ((min2 + max2) * 0.5F) * obb.axis2;
  return obb;
}

std::array<Edge2D, 4> makeEdges(const OBB2D &obb) {
  return {
      Edge2D{obb.center + obb.axis1 * (obb.length1 * 0.5F),
             obb.axis2, obb.length2},
      Edge2D{obb.center - obb.axis1 * (obb.length1 * 0.5F),
             obb.axis2, obb.length2},
      Edge2D{obb.center + obb.axis2 * (obb.length2 * 0.5F),
             obb.axis1, obb.length1},
      Edge2D{obb.center - obb.axis2 * (obb.length2 * 0.5F),
             obb.axis1, obb.length1},
  };
}

FitResult fitLShape(const std::vector<Eigen::Vector2f> &points,
                    float angle_step_deg, float sigma) {
  FitResult result;
  if (points.size() < 3) {
    return result;
  }

  const float step = std::clamp(angle_step_deg, 0.1F, 10.0F) * kPi / 180.0F;
  sigma = std::max(sigma, 0.001F);
  const float sigma2 = sigma * sigma;
  float best_score = -std::numeric_limits<float>::max();
  float best_theta = 0.0F;

  for (float theta = 0.0F; theta < kPi; theta += step) {
    const Eigen::Vector2f u(std::cos(theta), std::sin(theta));
    const Eigen::Vector2f v(-u.y(), u.x());

    float min_u = std::numeric_limits<float>::max();
    float max_u = -std::numeric_limits<float>::max();
    float min_v = std::numeric_limits<float>::max();
    float max_v = -std::numeric_limits<float>::max();
    std::vector<std::array<float, 2>> projected;
    projected.reserve(points.size());
    for (const auto &p : points) {
      const float pu = p.dot(u);
      const float pv = p.dot(v);
      projected.push_back({pu, pv});
      min_u = std::min(min_u, pu);
      max_u = std::max(max_u, pu);
      min_v = std::min(min_v, pv);
      max_v = std::max(max_v, pv);
    }

    const std::array<float, 2> u_sides = {min_u, max_u};
    const std::array<float, 2> v_sides = {min_v, max_v};
    for (const float us : u_sides) {
      for (const float vs : v_sides) {
        float support_score = 0.0F;
        float mean_dist = 0.0F;
        for (const auto &pv : projected) {
          const float d = std::min(std::abs(pv[0] - us),
                                   std::abs(pv[1] - vs));
          support_score += std::exp(-(d * d) / (2.0F * sigma2));
          mean_dist += d;
        }
        mean_dist /= static_cast<float>(projected.size());
        const float score = support_score - 0.2F * mean_dist / sigma;
        if (score > best_score) {
          best_score = score;
          best_theta = theta;
        }
      }
    }
  }

  result.obb = obbFromAxes(points, Eigen::Vector2f(std::cos(best_theta),
                                                   std::sin(best_theta)));
  result.valid = result.obb.length1 > 1e-5F && result.obb.length2 > 1e-5F;
  if (result.valid) {
    result.edges = makeEdges(result.obb);
  }
  return result;
}

float distanceOriginToSegment(const Edge2D &edge) {
  const Eigen::Vector2f axis = normalizedOr(edge.axis,
                                            Eigen::Vector2f(0.0F, 1.0F));
  const float half = edge.length * 0.5F;
  const float s = std::clamp(-edge.center.dot(axis), -half, half);
  return (edge.center + s * axis).norm();
}

}  // namespace

class YDeciderNode : public rclcpp::Node {
public:
  using YDecider = inha_interfaces::action::YDecider;
  using GoalHandle = rclcpp_action::ServerGoalHandle<YDecider>;

  YDeciderNode()
      : Node("y_decider"),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {
    declare_parameter<std::string>("mask_topic",
                                   "/sam2/binary/image_raw/compressed");
    declare_parameter<std::string>("cloud_topic",
                                   "/camera/camera_head/depth/color/points");
    declare_parameter<std::string>("camera_info_topic",
                                   "/camera/camera_head/color/camera_info");
    declare_parameter<std::string>("target_frame", "base_nav");
    declare_parameter<double>("sync_tolerance_sec", 0.15);
    declare_parameter<double>("max_decision_age_sec", 0.5);
    declare_parameter<double>("decision_wait_timeout_sec", 1.0);
    declare_parameter<float>("fx", 0.0F);
    declare_parameter<float>("fy", 0.0F);
    declare_parameter<float>("cx", 0.0F);
    declare_parameter<float>("cy", 0.0F);
    declare_parameter<int>("mask_threshold", 127);
    declare_parameter<int>("min_mask_points", 100);
    declare_parameter<float>("roi_x_min", 0.05F);
    declare_parameter<float>("roi_x_max", 3.0F);
    declare_parameter<float>("roi_y_abs_max", 1.0F);
    declare_parameter<float>("roi_z_min", 0.03F);
    declare_parameter<float>("roi_z_max", 1.5F);
    declare_parameter<float>("leaf_size", 0.02F);
    declare_parameter<int>("mean_k", 30);
    declare_parameter<float>("stddev_mul_thresh", 1.0F);
    declare_parameter<float>("cluster_tolerance", 0.05F);
    declare_parameter<int>("min_cluster_size", 50);
    declare_parameter<int>("max_cluster_size", 20000);
    declare_parameter<float>("min_cluster_area", 0.01F);
    declare_parameter<float>("l_shape_angle_step_deg", 1.0F);
    declare_parameter<float>("l_shape_sigma", 0.03F);
    declare_parameter<bool>("debug_log", true);

    get_parameter("mask_topic", mask_topic_);
    get_parameter("cloud_topic", cloud_topic_);
    get_parameter("camera_info_topic", camera_info_topic_);
    get_parameter("target_frame", target_frame_);
    get_parameter("sync_tolerance_sec", sync_tolerance_sec_);
    get_parameter("max_decision_age_sec", max_decision_age_sec_);
    get_parameter("decision_wait_timeout_sec", decision_wait_timeout_sec_);
    float param_fx = 0.0F;
    float param_fy = 0.0F;
    float param_cx = 0.0F;
    float param_cy = 0.0F;
    get_parameter("fx", param_fx);
    get_parameter("fy", param_fy);
    get_parameter("cx", param_cx);
    get_parameter("cy", param_cy);
    get_parameter("mask_threshold", mask_threshold_);
    get_parameter("min_mask_points", min_mask_points_);
    get_parameter("roi_x_min", roi_x_min_);
    get_parameter("roi_x_max", roi_x_max_);
    get_parameter("roi_y_abs_max", roi_y_abs_max_);
    get_parameter("roi_z_min", roi_z_min_);
    get_parameter("roi_z_max", roi_z_max_);
    get_parameter("leaf_size", leaf_size_);
    get_parameter("mean_k", mean_k_);
    get_parameter("stddev_mul_thresh", stddev_mul_thresh_);
    get_parameter("cluster_tolerance", cluster_tolerance_);
    get_parameter("min_cluster_size", min_cluster_size_);
    get_parameter("max_cluster_size", max_cluster_size_);
    get_parameter("min_cluster_area", min_cluster_area_);
    get_parameter("l_shape_angle_step_deg", l_shape_angle_step_deg_);
    get_parameter("l_shape_sigma", l_shape_sigma_);
    get_parameter("debug_log", debug_log_);

    mask_threshold_ = std::clamp(mask_threshold_, 0, 255);
    min_mask_points_ = std::max(min_mask_points_, 1);
    roi_x_max_ = std::max(roi_x_max_, roi_x_min_);
    roi_y_abs_max_ = std::max(0.0F, roi_y_abs_max_);
    roi_z_max_ = std::max(roi_z_max_, roi_z_min_);
    leaf_size_ = std::max(0.0F, leaf_size_);
    mean_k_ = std::max(1, mean_k_);
    cluster_tolerance_ = std::max(0.001F, cluster_tolerance_);
    min_cluster_size_ = std::max(1, min_cluster_size_);
    max_cluster_size_ = std::max(max_cluster_size_, min_cluster_size_);
    min_cluster_area_ = std::max(0.0F, min_cluster_area_);

    const auto qos = rclcpp::SensorDataQoS();
    cloud_sub_ = create_subscription<CloudMsg>(
        cloud_topic_, qos,
        std::bind(&YDeciderNode::cloudCallback, this, std::placeholders::_1));
    mask_sub_ = create_subscription<ImageMsg>(
        mask_topic_, qos,
        std::bind(&YDeciderNode::maskCallback, this, std::placeholders::_1));
    camera_info_sub_ = create_subscription<CameraInfoMsg>(
        camera_info_topic_, qos,
        std::bind(&YDeciderNode::cameraInfoCallback, this,
                  std::placeholders::_1));

    if (param_fx > 0.0F && param_fy > 0.0F) {
      intrinsics_.valid = true;
      intrinsics_.fx = param_fx;
      intrinsics_.fy = param_fy;
      intrinsics_.cx = param_cx;
      intrinsics_.cy = param_cy;
    }

    selected_cloud_pub_ =
        create_publisher<CloudMsg>("/y_decider/selected_cloud", qos);
    edge_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(
        "/y_decider/closest_edge_marker", rclcpp::QoS(1).reliable());

    server_ = rclcpp_action::create_server<YDecider>(
        this, "y_decision",
        std::bind(&YDeciderNode::handleGoal, this, std::placeholders::_1,
                  std::placeholders::_2),
        std::bind(&YDeciderNode::handleCancel, this, std::placeholders::_1),
        std::bind(&YDeciderNode::handleAccepted, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
                "YDecider ready mask=%s cloud=%s info=%s target=%s",
                mask_topic_.c_str(), cloud_topic_.c_str(),
                camera_info_topic_.c_str(), target_frame_.c_str());
  }

private:
  struct Decision {
    bool valid = false;
    std::string message;
    rclcpp::Time stamp;
    std::string direction;
    float remain_distance = 0.0F;
    float turn_angle_deg = 0.0F;
    geometry_msgs::msg::Point midpoint;
  };

  struct CameraIntrinsics {
    bool valid = false;
    float fx = 0.0F;
    float fy = 0.0F;
    float cx = 0.0F;
    float cy = 0.0F;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::string frame_id;
  };

  rclcpp_action::GoalResponse
  handleGoal(const rclcpp_action::GoalUUID &,
             std::shared_ptr<const YDecider::Goal> goal) {
    if (!goal || !goal->start) {
      RCLCPP_WARN(get_logger(), "Rejecting y_decision goal: start=false");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handleCancel(std::shared_ptr<GoalHandle>) {
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccepted(std::shared_ptr<GoalHandle> gh) {
    std::thread(std::bind(&YDeciderNode::execute, this, gh)).detach();
  }

  void execute(std::shared_ptr<GoalHandle> gh) {
    const rclcpp::Time request_time = now();
    const double timeout = decision_wait_timeout_sec_;
    const rclcpp::Time deadline =
        request_time + rclcpp::Duration::from_seconds(std::max(0.1, timeout));
    auto feedback = std::make_shared<YDecider::Feedback>();
    auto result = std::make_shared<YDecider::Result>();
    rclcpp::Rate rate(20);

    while (rclcpp::ok()) {
      if (gh->is_canceling()) {
        result->success = false;
        result->message = "Cancelled";
        gh->canceled(result);
        return;
      }

      Decision decision;
      {
        std::lock_guard<std::mutex> lk(decision_mutex_);
        decision = latest_decision_;
      }
      const bool fresh_enough =
          decision.valid &&
          (now() - decision.stamp).seconds() <= max_decision_age_sec_;
      const bool is_new =
          decision.valid && (decision.stamp - request_time).seconds() > 0.0;
      if (fresh_enough && is_new) {
        fillResult(decision, *result);
        gh->succeed(result);
        return;
      }

      if ((now() - deadline).seconds() > 0.0) {
        result->success = false;
        result->message =
            decision.message.empty() ? "No valid y decision" : decision.message;
        gh->abort(result);
        return;
      }

      feedback->progress =
          static_cast<float>(std::clamp(
              1.0 - (deadline - now()).seconds() /
                        std::max(0.1, timeout),
              0.0, 1.0));
      gh->publish_feedback(feedback);
      rate.sleep();
    }

    result->success = false;
    result->message = "ROS shutdown";
    gh->abort(result);
  }

  void cloudCallback(const CloudMsg::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lk(cloud_mutex_);
    cloud_cache_.push_back(msg);
    while (cloud_cache_.size() > 10) {
      cloud_cache_.pop_front();
    }
  }

  void cameraInfoCallback(const CameraInfoMsg::SharedPtr msg) {
    if (!msg || msg->k[0] <= 0.0 || msg->k[4] <= 0.0) {
      return;
    }

    CameraIntrinsics intrinsics;
    intrinsics.valid = true;
    intrinsics.fx = static_cast<float>(msg->k[0]);
    intrinsics.fy = static_cast<float>(msg->k[4]);
    intrinsics.cx = static_cast<float>(msg->k[2]);
    intrinsics.cy = static_cast<float>(msg->k[5]);
    intrinsics.width = msg->width;
    intrinsics.height = msg->height;
    intrinsics.frame_id = msg->header.frame_id;
    {
      std::lock_guard<std::mutex> lk(intrinsics_mutex_);
      intrinsics_ = intrinsics;
    }
    RCLCPP_INFO_ONCE(get_logger(),
                     "CameraInfo received for y_decider projection");
  }

  void maskCallback(const ImageMsg::ConstSharedPtr msg) {
    CloudMsg::ConstSharedPtr cloud_msg;
    {
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      cloud_msg = findClosestCloud(msg->header.stamp);
    }
    if (!cloud_msg) {
      storeInvalid("No synchronized pointcloud for mask");
      return;
    }
    processFrame(msg, cloud_msg);
  }

  CloudMsg::ConstSharedPtr findClosestCloud(const rclcpp::Time &stamp) const {
    CloudMsg::ConstSharedPtr best;
    double best_dt = std::numeric_limits<double>::max();
    for (const auto &cloud : cloud_cache_) {
      const double dt = std::abs((rclcpp::Time(cloud->header.stamp) -
                                  stamp).seconds());
      if (dt < best_dt) {
        best_dt = dt;
        best = cloud;
      }
    }
    if (best && best_dt <= sync_tolerance_sec_) {
      return best;
    }
    return nullptr;
  }

  void processFrame(const ImageMsg::ConstSharedPtr &mask_msg,
                    const CloudMsg::ConstSharedPtr &cloud_msg) {
    const cv::Mat mask = decodeMask(*mask_msg);
    if (mask.empty()) {
      storeInvalid("Mask decode failed");
      return;
    }

    pcl::PointCloud<pcl::PointXYZ> source;
    pcl::fromROSMsg(*cloud_msg, source);
    if (source.empty()) {
      storeInvalid("Input pointcloud empty");
      return;
    }

    const std::size_t mask_points =
        static_cast<std::size_t>(mask.cols) * static_cast<std::size_t>(mask.rows);
    const bool index_aligned =
        source.points.size() >= mask_points &&
        ((source.width == static_cast<std::uint32_t>(mask.cols) &&
          source.height == static_cast<std::uint32_t>(mask.rows)) ||
         source.points.size() == mask_points);

    CameraIntrinsics intrinsics;
    if (!index_aligned) {
      {
        std::lock_guard<std::mutex> lk(intrinsics_mutex_);
        intrinsics = intrinsics_;
      }
      if (!intrinsics.valid) {
        std::ostringstream ss;
        ss << "Mask/cloud size mismatch mask=" << mask.cols << "x" << mask.rows
           << " cloud=" << source.width << "x" << source.height
           << " and no CameraInfo for projection";
        storeInvalid(ss.str());
        return;
      }
      intrinsics = scaledIntrinsicsForMask(intrinsics, mask);
    }

    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_.lookupTransform(target_frame_, cloud_msg->header.frame_id,
                                      rclcpp::Time(cloud_msg->header.stamp),
                                      rclcpp::Duration::from_seconds(0.2));
    } catch (const tf2::TransformException &ex) {
      storeInvalid(std::string("TF failed: ") + ex.what());
      return;
    }

    geometry_msgs::msg::TransformStamped projection_tf;
    bool use_projection_tf = false;
    if (!index_aligned && !intrinsics.frame_id.empty() &&
        intrinsics.frame_id != cloud_msg->header.frame_id) {
      try {
        projection_tf = tf_buffer_.lookupTransform(
            intrinsics.frame_id, cloud_msg->header.frame_id,
            rclcpp::Time(cloud_msg->header.stamp),
            rclcpp::Duration::from_seconds(0.2));
        use_projection_tf = true;
      } catch (const tf2::TransformException &ex) {
        storeInvalid(std::string("Projection TF failed: ") + ex.what());
        return;
      }
    }

    auto masked_cloud = index_aligned
                            ? maskAndTransformByIndex(source, mask, tf)
                            : maskAndTransformByProjection(source, mask, tf,
                                                           intrinsics,
                                                           projection_tf,
                                                           use_projection_tf);
    if (static_cast<int>(masked_cloud->points.size()) < min_mask_points_) {
      storeInvalid("Too few masked points after ROI");
      return;
    }
    filterCloud(masked_cloud);
    if (static_cast<int>(masked_cloud->points.size()) < min_cluster_size_) {
      storeInvalid("Too few points after filtering");
      return;
    }

    auto cluster = selectClosestCluster(masked_cloud);
    if (!cluster || cluster->empty()) {
      storeInvalid("No valid cluster");
      return;
    }

    std::vector<Eigen::Vector2f> points;
    points.reserve(cluster->points.size());
    for (const auto &p : cluster->points) {
      points.emplace_back(p.x, p.y);
    }
    const FitResult fit = fitLShape(points, l_shape_angle_step_deg_,
                                    l_shape_sigma_);
    if (!fit.valid) {
      storeInvalid("L-shape fit failed");
      return;
    }

    const Edge2D edge = chooseClosestEdge(fit.edges);
    Decision decision;
    decision.valid = true;
    decision.message = "ok";
    decision.stamp = now();
    decision.midpoint.x = edge.center.x();
    decision.midpoint.y = edge.center.y();
    decision.midpoint.z = 0.0;
    decision.direction = (edge.center.y() >= 0.0F) ? "left" : "right";
    decision.turn_angle_deg = (edge.center.y() >= 0.0F) ? 90.0F : -90.0F;
    decision.remain_distance = std::abs(edge.center.y());
    {
      std::lock_guard<std::mutex> lk(decision_mutex_);
      latest_decision_ = decision;
    }

    publishSelectedCloud(cluster, cloud_msg->header.stamp);
    publishEdgeMarker(edge, cloud_msg->header.stamp);

    if (debug_log_) {
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 300,
          "y_decision midpoint=(%.3f, %.3f) dir=%s dist=%.3f",
          decision.midpoint.x, decision.midpoint.y,
          decision.direction.c_str(), decision.remain_distance);
    }
  }

  cv::Mat decodeMask(const ImageMsg &msg) const {
    if (msg.data.empty()) {
      return {};
    }
    const cv::Mat encoded(1, static_cast<int>(msg.data.size()), CV_8UC1,
                          const_cast<std::uint8_t *>(msg.data.data()));
    cv::Mat mask = cv::imdecode(encoded, cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
      return mask;
    }
    cv::threshold(mask, mask, mask_threshold_, 255, cv::THRESH_BINARY);
    return mask;
  }

  CameraIntrinsics scaledIntrinsicsForMask(const CameraIntrinsics &intrinsics,
                                           const cv::Mat &mask) const {
    CameraIntrinsics scaled = intrinsics;
    if (intrinsics.width > 0 && intrinsics.height > 0) {
      const float sx = static_cast<float>(mask.cols) /
                       static_cast<float>(intrinsics.width);
      const float sy = static_cast<float>(mask.rows) /
                       static_cast<float>(intrinsics.height);
      scaled.fx *= sx;
      scaled.cx *= sx;
      scaled.fy *= sy;
      scaled.cy *= sy;
    }
    return scaled;
  }

  static pcl::PointXYZ transformPoint(
      const pcl::PointXYZ &p, const tf2::Matrix3x3 &r,
      const geometry_msgs::msg::Vector3 &tr) {
    pcl::PointXYZ out;
    out.x = static_cast<float>(r[0][0] * p.x + r[0][1] * p.y +
                               r[0][2] * p.z + tr.x);
    out.y = static_cast<float>(r[1][0] * p.x + r[1][1] * p.y +
                               r[1][2] * p.z + tr.y);
    out.z = static_cast<float>(r[2][0] * p.x + r[2][1] * p.y +
                               r[2][2] * p.z + tr.z);
    return out;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr maskAndTransformByIndex(
      const pcl::PointCloud<pcl::PointXYZ> &source, const cv::Mat &mask,
      const geometry_msgs::msg::TransformStamped &tf) const {
    auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    out->points.reserve(source.points.size() / 4);

    const auto &tr = tf.transform.translation;
    const auto &qr = tf.transform.rotation;
    tf2::Quaternion q(qr.x, qr.y, qr.z, qr.w);
    tf2::Matrix3x3 r(q);

    for (int v = 0; v < mask.rows; ++v) {
      const auto *row = mask.ptr<std::uint8_t>(v);
      for (int u = 0; u < mask.cols; ++u) {
        if (row[u] == 0) continue;
        const auto &p = source.points[static_cast<std::size_t>(v) *
                                      static_cast<std::size_t>(mask.cols) +
                                      static_cast<std::size_t>(u)];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
            !std::isfinite(p.z)) {
          continue;
        }

        const pcl::PointXYZ base = transformPoint(p, r, tr);
        if (base.x < roi_x_min_ || base.x > roi_x_max_) continue;
        if (std::abs(base.y) > roi_y_abs_max_) continue;
        if (base.z < roi_z_min_ || base.z > roi_z_max_) continue;
        out->points.push_back(base);
      }
    }

    out->width = static_cast<std::uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = false;
    return out;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr maskAndTransformByProjection(
      const pcl::PointCloud<pcl::PointXYZ> &source, const cv::Mat &mask,
      const geometry_msgs::msg::TransformStamped &tf,
      const CameraIntrinsics &intrinsics,
      const geometry_msgs::msg::TransformStamped &projection_tf,
      bool use_projection_tf) const {
    auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    out->points.reserve(source.points.size() / 4);

    const auto &tr = tf.transform.translation;
    const auto &qr = tf.transform.rotation;
    tf2::Quaternion q(qr.x, qr.y, qr.z, qr.w);
    tf2::Matrix3x3 r(q);

    geometry_msgs::msg::Vector3 projection_tr;
    tf2::Matrix3x3 projection_r;
    projection_r.setIdentity();
    if (use_projection_tf) {
      projection_tr = projection_tf.transform.translation;
      const auto &projection_qr = projection_tf.transform.rotation;
      tf2::Quaternion projection_q(projection_qr.x, projection_qr.y,
                                   projection_qr.z, projection_qr.w);
      projection_r = tf2::Matrix3x3(projection_q);
    }

    for (const auto &p : source.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
          !std::isfinite(p.z)) {
        continue;
      }

      const pcl::PointXYZ camera =
          use_projection_tf ? transformPoint(p, projection_r, projection_tr) : p;
      if (!std::isfinite(camera.x) || !std::isfinite(camera.y) ||
          !std::isfinite(camera.z) || camera.z <= 0.0F) {
        continue;
      }

      const float u_f = intrinsics.fx * camera.x / camera.z + intrinsics.cx;
      const float v_f = intrinsics.fy * camera.y / camera.z + intrinsics.cy;
      const int u = static_cast<int>(std::lround(u_f));
      const int v = static_cast<int>(std::lround(v_f));
      if (u < 0 || u >= mask.cols || v < 0 || v >= mask.rows) {
        continue;
      }
      if (mask.ptr<std::uint8_t>(v)[u] == 0) {
        continue;
      }

      const pcl::PointXYZ base = transformPoint(p, r, tr);
      if (base.x < roi_x_min_ || base.x > roi_x_max_) continue;
      if (std::abs(base.y) > roi_y_abs_max_) continue;
      if (base.z < roi_z_min_ || base.z > roi_z_max_) continue;
      out->points.push_back(base);
    }

    out->width = static_cast<std::uint32_t>(out->points.size());
    out->height = 1;
    out->is_dense = false;
    return out;
  }

  void filterCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) const {
    if (leaf_size_ > 0.0F) {
      pcl::VoxelGrid<pcl::PointXYZ> voxel;
      voxel.setInputCloud(cloud);
      voxel.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
      auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      voxel.filter(*filtered);
      cloud = filtered;
    }
    if (static_cast<int>(cloud->points.size()) > mean_k_) {
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      sor.setInputCloud(cloud);
      sor.setMeanK(mean_k_);
      sor.setStddevMulThresh(stddev_mul_thresh_);
      auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      sor.filter(*filtered);
      cloud = filtered;
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr selectClosestCluster(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) const {
    auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    tree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusters;
    ec.extract(clusters);
    if (clusters.empty()) {
      return {};
    }

    int best_idx = -1;
    float best_min_x = std::numeric_limits<float>::max();
    float best_abs_y = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < clusters.size(); ++i) {
      float min_x = std::numeric_limits<float>::max();
      float min_y = std::numeric_limits<float>::max();
      float max_y = -std::numeric_limits<float>::max();
      float min_area_x = std::numeric_limits<float>::max();
      float max_area_x = -std::numeric_limits<float>::max();
      float sum_y = 0.0F;
      for (const int idx : clusters[i].indices) {
        const auto &p = cloud->points[static_cast<std::size_t>(idx)];
        min_x = std::min(min_x, p.x);
        min_area_x = std::min(min_area_x, p.x);
        max_area_x = std::max(max_area_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
        sum_y += p.y;
      }
      const float area = std::max(0.0F, max_area_x - min_area_x) *
                         std::max(0.0F, max_y - min_y);
      if (area < min_cluster_area_) {
        continue;
      }
      const float abs_y = std::abs(sum_y /
                                   static_cast<float>(clusters[i].indices.size()));
      if (min_x < best_min_x ||
          (std::abs(min_x - best_min_x) < 1e-4F && abs_y < best_abs_y)) {
        best_idx = static_cast<int>(i);
        best_min_x = min_x;
        best_abs_y = abs_y;
      }
    }

    if (best_idx < 0) {
      return {};
    }

    auto selected = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    selected->points.reserve(clusters[static_cast<std::size_t>(best_idx)]
                                 .indices.size());
    for (const int idx : clusters[static_cast<std::size_t>(best_idx)].indices) {
      selected->points.push_back(cloud->points[static_cast<std::size_t>(idx)]);
    }
    selected->width = static_cast<std::uint32_t>(selected->points.size());
    selected->height = 1;
    selected->is_dense = false;
    return selected;
  }

  Edge2D chooseClosestEdge(const std::array<Edge2D, 4> &edges) const {
    const Edge2D *best = &edges[0];
    float best_dist = distanceOriginToSegment(edges[0]);
    for (std::size_t i = 1; i < edges.size(); ++i) {
      const float dist = distanceOriginToSegment(edges[i]);
      if (dist < best_dist) {
        best = &edges[i];
        best_dist = dist;
      }
    }
    return *best;
  }

  void publishSelectedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                            const builtin_interfaces::msg::Time &stamp) {
    if (selected_cloud_pub_->get_subscription_count() == 0) return;
    CloudMsg out;
    pcl::toROSMsg(*cloud, out);
    out.header.stamp = stamp;
    out.header.frame_id = target_frame_;
    selected_cloud_pub_->publish(out);
  }

  void publishEdgeMarker(const Edge2D &edge,
                         const builtin_interfaces::msg::Time &stamp) {
    if (edge_marker_pub_->get_subscription_count() == 0) return;
    visualization_msgs::msg::Marker m;
    m.header.stamp = stamp;
    m.header.frame_id = target_frame_;
    m.ns = "y_decider";
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = 0.025;
    m.color.r = 0.1F;
    m.color.g = 0.7F;
    m.color.b = 1.0F;
    m.color.a = 1.0F;
    const Eigen::Vector2f axis = normalizedOr(edge.axis,
                                              Eigen::Vector2f(0.0F, 1.0F));
    const Eigen::Vector2f p0 = edge.center - axis * (edge.length * 0.5F);
    const Eigen::Vector2f p1 = edge.center + axis * (edge.length * 0.5F);
    geometry_msgs::msg::Point a;
    a.x = p0.x();
    a.y = p0.y();
    a.z = 0.05;
    geometry_msgs::msg::Point b;
    b.x = p1.x();
    b.y = p1.y();
    b.z = 0.05;
    m.points = {a, b};
    edge_marker_pub_->publish(m);
  }

  void fillResult(const Decision &decision, YDecider::Result &result) const {
    result.success = true;
    result.message = decision.message;
    result.direction = decision.direction;
    result.remain_distance = decision.remain_distance;
    result.turn_angle_deg = decision.turn_angle_deg;
    result.midpoint = decision.midpoint;
  }

  void storeInvalid(const std::string &message) {
    {
      std::lock_guard<std::mutex> lk(decision_mutex_);
      latest_decision_.valid = false;
      latest_decision_.message = message;
      latest_decision_.stamp = now();
    }
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s",
                         message.c_str());
  }

  std::string mask_topic_;
  std::string cloud_topic_;
  std::string camera_info_topic_;
  std::string target_frame_;
  double sync_tolerance_sec_ = 0.15;
  double max_decision_age_sec_ = 0.5;
  double decision_wait_timeout_sec_ = 1.0;
  int mask_threshold_ = 127;
  int min_mask_points_ = 100;
  float roi_x_min_ = 0.05F;
  float roi_x_max_ = 3.0F;
  float roi_y_abs_max_ = 1.0F;
  float roi_z_min_ = 0.03F;
  float roi_z_max_ = 1.5F;
  float leaf_size_ = 0.02F;
  int mean_k_ = 30;
  float stddev_mul_thresh_ = 1.0F;
  float cluster_tolerance_ = 0.05F;
  int min_cluster_size_ = 50;
  int max_cluster_size_ = 20000;
  float min_cluster_area_ = 0.01F;
  float l_shape_angle_step_deg_ = 1.0F;
  float l_shape_sigma_ = 0.03F;
  bool debug_log_ = true;

  rclcpp::Subscription<ImageMsg>::SharedPtr mask_sub_;
  rclcpp::Subscription<CloudMsg>::SharedPtr cloud_sub_;
  rclcpp::Subscription<CameraInfoMsg>::SharedPtr camera_info_sub_;
  rclcpp::Publisher<CloudMsg>::SharedPtr selected_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr edge_marker_pub_;
  rclcpp_action::Server<YDecider>::SharedPtr server_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  mutable std::mutex cloud_mutex_;
  std::deque<CloudMsg::ConstSharedPtr> cloud_cache_;
  std::mutex intrinsics_mutex_;
  CameraIntrinsics intrinsics_;
  std::mutex decision_mutex_;
  Decision latest_decision_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<YDeciderNode>());
  rclcpp::shutdown();
  return 0;
}
