// align_measure: 도착 후 재정렬용 측정 노드.
//
// 액션을 받으면
//   1) /approach/active 를 올리고 depth_edge_detector 를 SetEnable 서비스로 켜고,
//   2) /approach/edge_error 의 theta_error 를 짧게 누적해 평탄화(yaw),
<<<<<<< HEAD
//   3) 최신 포인트클라우드를 base 프레임으로 변환한 뒤 RANSAC 평면 inlier의
//      평균 y 로 y_error 를 측정,
=======
	//   3) 세그멘테이션 마스크 + 포인트클라우드로 테이블 점을 bas
//   4) edge_detector 와 /approach/active 를 끄고 결과로 반환.
#include "inha_interfaces/action/align_measure.hpp"
#include "close_approach/msg/approach_error.hpp"
#include "inha_interfaces/srv/set_enable.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
<<<<<<< HEAD
#include <functional>
#include <future>
=======
#include <limits>
>>>>>>> a25352d (0626)
#include <memory>
#include <mutex>
#include <string>
#include <thread>
<<<<<<< HEAD
#include <vector>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
=======
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
>>>>>>> a25352d (0626)
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
<<<<<<< HEAD
=======
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace {

using AlignMeasure = inha_interfaces::action::AlignMeasure;
using ApproachError = close_approach::msg::ApproachError;
using SetEnable = inha_interfaces::srv::SetEnable;
using CloudMsg = sensor_msgs::msg::PointCloud2;


}  // namespace

class AlignMeasureNode : public rclcpp::Node {
public:
  using GoalHandle = rclcpp_action::ServerGoalHandle<AlignMeasure>;

  AlignMeasureNode()
      : Node("align_measure"),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {
    declare_parameter<std::string>("action_name", "align_measure");
    declare_parameter<std::string>("cloud_topic",
                                   "/camera/camera_head/depth/color/points");

    declare_parameter<std::string>("edge_enable_service_name",
                                   "/approach/edge_detector/set_enable");
    declare_parameter<std::string>("edge_error_topic", "/approach/edge_error");
    declare_parameter<std::string>("target_frame", "base_nav");

    declare_parameter<double>("measure_duration_sec", 0.8);
    declare_parameter<double>("edge_enable_timeout_sec", 1.0);
    declare_parameter<double>("tf_timeout_sec", 0.2);

    declare_parameter<int>("min_total_points", 200);
    declare_parameter<int>("ransac_min_inliers", 200);
    declare_parameter<int>("ransac_max_iterations", 100);
    declare_parameter<double>("ransac_distance_threshold", 0.02);

    declare_parameter<double>("theta_trim_band_rad", 0.10);

    declare_parameter<bool>("debug_log", true);

    get_parameter("action_name", action_name_);
    get_parameter("cloud_topic", cloud_topic_);

    get_parameter("edge_enable_service_name", edge_enable_service_name_);
    get_parameter("edge_error_topic", edge_error_topic_);
    get_parameter("target_frame", target_frame_);
    get_parameter("measure_duration_sec", measure_duration_sec_);
    get_parameter("edge_enable_timeout_sec", edge_enable_timeout_sec_);
    get_parameter("tf_timeout_sec", tf_timeout_sec_);
    get_parameter("min_total_points", min_total_points_);
    get_parameter("ransac_min_inliers", ransac_min_inliers_);
    get_parameter("ransac_max_iterations", ransac_max_iterations_);
    get_parameter("ransac_distance_threshold", ransac_distance_threshold_);

    get_parameter("theta_trim_band_rad", theta_trim_band_rad_);
    get_parameter("debug_log", debug_log_);

    measure_duration_sec_ = std::max(0.1, measure_duration_sec_);
    min_total_points_ = std::max(3, min_total_points_);
    ransac_min_inliers_ = std::max(3, ransac_min_inliers_);
    ransac_max_iterations_ = std::max(1, ransac_max_iterations_);
    ransac_distance_threshold_ = std::max(0.001, ransac_distance_threshold_);


    const auto qos = rclcpp::SensorDataQoS();
    cloud_sub_ = create_subscription<CloudMsg>(
        cloud_topic_, qos,
        std::bind(&AlignMeasureNode::cloudCallback, this,
                  std::placeholders::_1));

    edge_error_sub_ = create_subscription<ApproachError>(
        edge_error_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
        std::bind(&AlignMeasureNode::edgeErrorCallback, this,
                  std::placeholders::_1));

    rclcpp::QoS active_qos(rclcpp::KeepLast(1));
    active_qos.reliable();
    active_qos.transient_local();
    active_pub_ = create_publisher<std_msgs::msg::Bool>(
        "/approach/active", active_qos);

    edge_enable_client_ = create_client<SetEnable>(edge_enable_service_name_);

    server_ = rclcpp_action::create_server<AlignMeasure>(
        this, action_name_,
        std::bind(&AlignMeasureNode::handleGoal, this, std::placeholders::_1,
                  std::placeholders::_2),
        std::bind(&AlignMeasureNode::handleCancel, this, std::placeholders::_1),
        std::bind(&AlignMeasureNode::handleAccepted, this,
                  std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
                "AlignMeasure ready action=%s cloud=%s edge_error=%s "
                "enable=%s target=%s",
                action_name_.c_str(), cloud_topic_.c_str(),

                edge_error_topic_.c_str(), edge_enable_service_name_.c_str(),
                target_frame_.c_str());
  }

private:
  struct PlaneMeasure {
    bool success = false;
    std::string message;
    float y_error = 0.0F;
    int input_points = 0;
    int inlier_points = 0;
    float a = 0.0F;
    float b = 0.0F;
    float c = 0.0F;
    float d = 0.0F;

  };

  static std::chrono::nanoseconds secondsToNanoseconds(double seconds) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(seconds));
  }

  rclcpp_action::GoalResponse
  handleGoal(const rclcpp_action::GoalUUID &,
             std::shared_ptr<const AlignMeasure::Goal> goal) {
    if (!goal || !goal->start) {
      RCLCPP_WARN(get_logger(), "Rejecting align_measure: start=false");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handleCancel(std::shared_ptr<GoalHandle>) {
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccepted(std::shared_ptr<GoalHandle> gh) {
    std::thread(std::bind(&AlignMeasureNode::execute, this, gh)).detach();
  }

  void cloudCallback(const CloudMsg::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lk(cloud_mutex_);
    cloud_cache_.push_back(msg);
    while (cloud_cache_.size() > 10) cloud_cache_.pop_front();
  }

  void edgeErrorCallback(const ApproachError::ConstSharedPtr msg) {
    if (!collecting_.load(std::memory_order_acquire)) return;
    if (!msg->valid) return;
    std::lock_guard<std::mutex> lk(theta_mutex_);
    theta_samples_.push_back(msg->theta_error);
  }

  void execute(std::shared_ptr<GoalHandle> gh) {
    auto result = std::make_shared<AlignMeasure::Result>();
    auto feedback = std::make_shared<AlignMeasure::Feedback>();

    // 누적 버퍼 초기화
    {
      std::lock_guard<std::mutex> lk(theta_mutex_);
      theta_samples_.clear();
    }

    feedback->progress = 0.1F;
    feedback->state = "enabling_edge_detector";
    gh->publish_feedback(feedback);

    publishApproachActive(true);
    std::string msg;
    const bool edge_on = setEdgeEnabled(true, msg);
    auto cleanup = [this, edge_on]() {
      collecting_.store(false, std::memory_order_release);
      if (edge_on) {
        std::string ignored;
        setEdgeEnabled(false, ignored);
      }
      publishApproachActive(false);
    };
    if (!edge_on) {
      RCLCPP_WARN(get_logger(), "edge enable failed: %s (yaw 평탄화 생략)",
                  msg.c_str());
    }
    collecting_.store(true, std::memory_order_release);

    feedback->progress = 0.3F;
    feedback->state = "measuring";
    gh->publish_feedback(feedback);

    const rclcpp::Time start = now();
    const rclcpp::Time deadline =
        start + rclcpp::Duration::from_seconds(measure_duration_sec_);


    while (rclcpp::ok() && (deadline - now()).seconds() > 0.0) {
      if (gh->is_canceling()) {
        cleanup();
        result->success = false;
        result->message = "Cancelled";
        gh->canceled(result);
        return;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    cleanup();


    feedback->progress = 0.85F;
    feedback->state = "computing";
    gh->publish_feedback(feedback);

    const CloudMsg::ConstSharedPtr cloud_msg = latestCloud();
    if (!cloud_msg) {
      result->success = false;
      result->message = "No pointcloud received";

      gh->abort(result);
      return;
    }
    const PlaneMeasure plane = measurePlaneY(cloud_msg);
    if (!plane.success) {
      result->success = false;
      result->message = plane.message;
      gh->abort(result);
      return;
    }


    // ---- yaw: 트림 평균 (지터 강건) ----
    int theta_n = 0;
    const float theta = robustThetaMean(theta_n);

    result->success = true;
    result->message = "ok";
    result->theta_error = theta;
    result->y_error = plane.y_error;

    if (debug_log_) {
      RCLCPP_INFO(get_logger(),
                  "align_measure: y_error=%.3f plane_inliers=%d/%d "
                  "plane=(%.3f, %.3f, %.3f, %.3f) theta=%.2fdeg(n=%d)",
                  plane.y_error, plane.inlier_points, plane.input_points,
                  plane.a, plane.b, plane.c, plane.d,
                  theta * 180.0 / M_PI, theta_n);

    }

    feedback->progress = 1.0F;
    feedback->state = "done";
    gh->publish_feedback(feedback);
    gh->succeed(result);
  }

  CloudMsg::ConstSharedPtr latestCloud() {
    std::lock_guard<std::mutex> lk(cloud_mutex_);
    if (cloud_cache_.empty()) return nullptr;
    return cloud_cache_.back();

  }

  static pcl::PointXYZ transformPoint(const pcl::PointXYZ &p,
                                      const tf2::Matrix3x3 &r,
                                      const geometry_msgs::msg::Vector3 &tr) {
    pcl::PointXYZ out;
    out.x = static_cast<float>(r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z +
                               tr.x);
    out.y = static_cast<float>(r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z +
                               tr.y);
    out.z = static_cast<float>(r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z +
                               tr.z);
    return out;
  }

  PlaneMeasure measurePlaneY(const CloudMsg::ConstSharedPtr &cloud_msg) {
    PlaneMeasure out;
    if (!cloud_msg) {
      out.message = "No pointcloud received";
      return out;
    }

    pcl::PointCloud<pcl::PointXYZ> source;
    pcl::fromROSMsg(*cloud_msg, source);
    if (source.empty()) {
      out.message = "Empty pointcloud";
      return out;
    }

    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_.lookupTransform(
          target_frame_, cloud_msg->header.frame_id,
          rclcpp::Time(cloud_msg->header.stamp),
          rclcpp::Duration::from_seconds(tf_timeout_sec_));
    } catch (const tf2::TransformException &ex) {
      out.message = std::string("TF lookup failed: ") + ex.what();
      return out;
    }


    const auto &tr = tf.transform.translation;
    const auto &qr = tf.transform.rotation;
    tf2::Matrix3x3 r(tf2::Quaternion(qr.x, qr.y, qr.z, qr.w));

    auto base_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    base_cloud->points.reserve(source.points.size());


    for (const auto &p : source.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        continue;
      base_cloud->points.push_back(transformPoint(p, r, tr));
    }
    base_cloud->width = static_cast<std::uint32_t>(base_cloud->points.size());
    base_cloud->height = 1;
    base_cloud->is_dense = false;
    out.input_points = static_cast<int>(base_cloud->points.size());

    if (out.input_points < min_total_points_) {
      out.message = "Too few finite cloud points: " +
                    std::to_string(out.input_points);
      return out;
    }

    pcl::ModelCoefficients coeff;
    pcl::PointIndices inliers;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(ransac_distance_threshold_);
    seg.setMaxIterations(ransac_max_iterations_);
    seg.setInputCloud(base_cloud);
    seg.segment(inliers, coeff);

    out.inlier_points = static_cast<int>(inliers.indices.size());
    if (coeff.values.size() < 4 || out.inlier_points < ransac_min_inliers_) {
      out.message = "RANSAC plane failed: inliers=" +
                    std::to_string(out.inlier_points) + "/" +
                    std::to_string(out.input_points);
      return out;
    }

    double sum_y = 0.0;
    int used = 0;
    for (const int idx : inliers.indices) {
      if (idx < 0 ||
          static_cast<std::size_t>(idx) >= base_cloud->points.size()) {
        continue;
      }
      sum_y += base_cloud->points[static_cast<std::size_t>(idx)].y;
      ++used;
    }
    if (used < ransac_min_inliers_) {
      out.message = "RANSAC plane valid indices too few: " +
                    std::to_string(used);
      return out;
    }

    out.inlier_points = used;
    out.y_error = static_cast<float>(sum_y / static_cast<double>(used));
    out.a = coeff.values[0];
    out.b = coeff.values[1];
    out.c = coeff.values[2];
    out.d = coeff.values[3];
    const float norm =
        std::sqrt(out.a * out.a + out.b * out.b + out.c * out.c);
    if (norm > 1e-6F) {
      out.a /= norm;
      out.b /= norm;
      out.c /= norm;
      out.d /= norm;
    }
    out.success = true;
    out.message = "ok";
    return out;

  }

  // 트림 평균: median 근처 band 안의 샘플만 평균 (지터 강건).
  float robustThetaMean(int &used) const {
    std::vector<float> s;
    {
      std::lock_guard<std::mutex> lk(theta_mutex_);
      s = theta_samples_;
    }
    used = 0;
    if (s.empty()) return 0.0F;
    std::sort(s.begin(), s.end());
    const float median = s[s.size() / 2];
    double sum = 0.0;
    int n = 0;
    for (const float v : s) {
      if (std::abs(v - median) <= theta_trim_band_rad_) {
        sum += v;
        ++n;
      }
    }
    if (n == 0) {
      used = static_cast<int>(s.size());
      return median;
    }
    used = n;
    return static_cast<float>(sum / n);
  }

  bool setEdgeEnabled(bool enable, std::string &message) {
    const auto timeout = secondsToNanoseconds(edge_enable_timeout_sec_);
    if (!edge_enable_client_->wait_for_service(timeout)) {
      message = "edge enable service unavailable";
      return false;
    }
    auto req = std::make_shared<SetEnable::Request>();
    req->enable = enable;
    auto future = edge_enable_client_->async_send_request(req);
    if (future.wait_for(timeout) != std::future_status::ready) {
      message = "edge enable timeout";
      return false;
    }
    const auto resp = future.get();
    if (!resp || !resp->success) {
      message = resp ? resp->message : "edge enable failed";
      return false;
    }
    return true;
  }

  void publishApproachActive(bool active) {
    std_msgs::msg::Bool msg;
    msg.data = active;
    active_pub_->publish(msg);
  }

  std::string action_name_;
  std::string cloud_topic_;

  std::string edge_enable_service_name_;
  std::string edge_error_topic_;
  std::string target_frame_;
  double measure_duration_sec_ = 0.8;
  double edge_enable_timeout_sec_ = 1.0;
  double tf_timeout_sec_ = 0.2;
  int min_total_points_ = 200;
  int ransac_min_inliers_ = 200;
  int ransac_max_iterations_ = 100;
  double ransac_distance_threshold_ = 0.02;

  double theta_trim_band_rad_ = 0.10;
  bool debug_log_ = true;

  rclcpp::Subscription<CloudMsg>::SharedPtr cloud_sub_;

  rclcpp::Subscription<ApproachError>::SharedPtr edge_error_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr active_pub_;
  rclcpp::Client<SetEnable>::SharedPtr edge_enable_client_;
  rclcpp_action::Server<AlignMeasure>::SharedPtr server_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex cloud_mutex_;
  std::deque<CloudMsg::ConstSharedPtr> cloud_cache_;

  mutable std::mutex theta_mutex_;
  std::vector<float> theta_samples_;
  std::atomic<bool> collecting_{false};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AlignMeasureNode>());
  rclcpp::shutdown();
  return 0;
}
