#include "inha_interfaces/action/y_nav_goal_decision.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace {

constexpr double kPi = 3.14159265358979323846;

double normalizeAngle(double angle) {
  while (angle > kPi) angle -= 2.0 * kPi;
  while (angle < -kPi) angle += 2.0 * kPi;
  return angle;
}

double yawFromQuaternion(const geometry_msgs::msg::Quaternion &q_msg) {
  tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  return yaw;
}

geometry_msgs::msg::Quaternion quaternionFromYaw(double yaw) {
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}

struct Transform2D {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
};

struct Pose2D {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
};

Pose2D transformPose(const Transform2D &tf, const Pose2D &pose) {
  const double c = std::cos(tf.yaw);
  const double s = std::sin(tf.yaw);
  Pose2D out;
  out.x = c * pose.x - s * pose.y + tf.x;
  out.y = s * pose.x + c * pose.y + tf.y;
  out.yaw = normalizeAngle(pose.yaw + tf.yaw);
  return out;
}

struct Candidate {
  Pose2D base_pose;
  Pose2D costmap_pose;
  Pose2D output_pose;
  double standoff = 0.0;
  double lateral = 0.0;
  double score = -std::numeric_limits<double>::infinity();
  int max_cost = 0;
};

using YNavGoalDecision = inha_interfaces::action::YNavGoalDecision;
using CostmapMsg = nav_msgs::msg::OccupancyGrid;
using PointStampedMsg = geometry_msgs::msg::PointStamped;
using SetBool = std_srvs::srv::SetBool;

}  // namespace

class YNavGoalDeciderNode : public rclcpp::Node {
public:
  using GoalHandle = rclcpp_action::ServerGoalHandle<YNavGoalDecision>;

  YNavGoalDeciderNode()
      : Node("y_nav_goal_decider"),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {
    declare_parameter<std::string>("action_name", "y_nav_goal_decision");
    declare_parameter<std::string>("y_decider_enable_service_name",
                                   "/y_decider/set_enable");
    declare_parameter<std::string>("y_decider_midpoint_topic",
                                   "/y_decider/midpoint");
    declare_parameter<std::string>("costmap_topic", "/local_costmap/costmap");
    declare_parameter<std::string>("base_frame", "base_nav");
    declare_parameter<std::string>("output_frame", "map");
    declare_parameter<double>("y_decider_enable_timeout_sec", 1.0);
    declare_parameter<double>("midpoint_timeout_sec", 2.0);
    declare_parameter<double>("costmap_timeout_sec", 1.0);
    declare_parameter<double>("tf_timeout_sec", 0.2);
    declare_parameter<double>("min_target_distance", 0.20);
    declare_parameter<double>("min_goal_forward_distance", 0.03);
    declare_parameter<double>("preferred_standoff_distance", 0.55);
    declare_parameter<std::vector<double>>(
        "standoff_candidates", std::vector<double>{0.35, 0.45, 0.55, 0.65, 0.75});
    declare_parameter<std::vector<double>>(
        "lateral_candidates", std::vector<double>{0.0, -0.10, 0.10, -0.20, 0.20});
    declare_parameter<double>("footprint_length", 0.70);
    declare_parameter<double>("footprint_width", 0.65);
    declare_parameter<double>("footprint_padding", 0.05);
    declare_parameter<int>("lethal_cost_threshold", 65);
    declare_parameter<bool>("unknown_is_blocked", true);
    declare_parameter<bool>("debug_log", true);

    get_parameter("action_name", action_name_);
    get_parameter("y_decider_enable_service_name",
                  y_decider_enable_service_name_);
    get_parameter("y_decider_midpoint_topic", y_decider_midpoint_topic_);
    get_parameter("costmap_topic", costmap_topic_);
    get_parameter("base_frame", base_frame_);
    get_parameter("output_frame", output_frame_);
    get_parameter("y_decider_enable_timeout_sec",
                  y_decider_enable_timeout_sec_);
    get_parameter("midpoint_timeout_sec", midpoint_timeout_sec_);
    get_parameter("costmap_timeout_sec", costmap_timeout_sec_);
    get_parameter("tf_timeout_sec", tf_timeout_sec_);
    get_parameter("min_target_distance", min_target_distance_);
    get_parameter("min_goal_forward_distance", min_goal_forward_distance_);
    get_parameter("preferred_standoff_distance", preferred_standoff_distance_);
    get_parameter("standoff_candidates", standoff_candidates_);
    get_parameter("lateral_candidates", lateral_candidates_);
    get_parameter("footprint_length", footprint_length_);
    get_parameter("footprint_width", footprint_width_);
    get_parameter("footprint_padding", footprint_padding_);
    get_parameter("lethal_cost_threshold", lethal_cost_threshold_);
    get_parameter("unknown_is_blocked", unknown_is_blocked_);
    get_parameter("debug_log", debug_log_);

    y_decider_enable_timeout_sec_ =
        std::max(0.1, y_decider_enable_timeout_sec_);
    midpoint_timeout_sec_ = std::max(0.1, midpoint_timeout_sec_);
    costmap_timeout_sec_ = std::max(0.1, costmap_timeout_sec_);
    tf_timeout_sec_ = std::max(0.01, tf_timeout_sec_);
    min_target_distance_ = std::max(0.01, min_target_distance_);
    min_goal_forward_distance_ = std::max(0.0, min_goal_forward_distance_);
    preferred_standoff_distance_ = std::max(0.01, preferred_standoff_distance_);
    footprint_length_ = std::max(0.05, footprint_length_);
    footprint_width_ = std::max(0.05, footprint_width_);
    footprint_padding_ = std::max(0.0, footprint_padding_);
    lethal_cost_threshold_ = std::clamp(lethal_cost_threshold_, 1, 100);
    sanitizeCandidates(standoff_candidates_, 0.01);
    sanitizeCandidates(lateral_candidates_, 0.0);

    const auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    const auto sensor_qos = rclcpp::SensorDataQoS();
    costmap_sub_ = create_subscription<CostmapMsg>(
        costmap_topic_, reliable_qos,
        std::bind(&YNavGoalDeciderNode::costmapCallback, this,
                  std::placeholders::_1));
    midpoint_sub_ = create_subscription<PointStampedMsg>(
        y_decider_midpoint_topic_, sensor_qos,
        std::bind(&YNavGoalDeciderNode::midpointCallback, this,
                  std::placeholders::_1));
    y_enable_client_ = create_client<SetBool>(y_decider_enable_service_name_);
    server_ = rclcpp_action::create_server<YNavGoalDecision>(
        this, action_name_,
        std::bind(&YNavGoalDeciderNode::handleGoal, this, std::placeholders::_1,
                  std::placeholders::_2),
        std::bind(&YNavGoalDeciderNode::handleCancel, this,
                  std::placeholders::_1),
        std::bind(&YNavGoalDeciderNode::handleAccepted, this,
                  std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
                "YNavGoalDecider ready action=%s enable_service=%s "
                "midpoint=%s costmap=%s output=%s",
                action_name_.c_str(), y_decider_enable_service_name_.c_str(),
                y_decider_midpoint_topic_.c_str(), costmap_topic_.c_str(),
                output_frame_.c_str());
  }

private:
  static void sanitizeCandidates(std::vector<double> &values, double min_abs) {
    values.erase(std::remove_if(values.begin(), values.end(),
                                [min_abs](double v) {
                                  return !std::isfinite(v) ||
                                         std::abs(v) < min_abs;
                                }),
                 values.end());
    if (values.empty()) {
      values.push_back(min_abs);
    }
  }

  static std::chrono::nanoseconds secondsToNanoseconds(double seconds) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(seconds));
  }

  rclcpp_action::GoalResponse
  handleGoal(const rclcpp_action::GoalUUID &,
             std::shared_ptr<const YNavGoalDecision::Goal> goal) {
    if (!goal || !goal->start) {
      RCLCPP_WARN(get_logger(), "Rejecting y_nav_goal_decision: start=false");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handleCancel(std::shared_ptr<GoalHandle>) {
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccepted(std::shared_ptr<GoalHandle> gh) {
    std::thread(std::bind(&YNavGoalDeciderNode::execute, this, gh)).detach();
  }

  void costmapCallback(const CostmapMsg::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lk(costmap_mutex_);
    latest_costmap_ = msg;
    latest_costmap_received_ = now();
  }

  void midpointCallback(const PointStampedMsg::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lk(midpoint_mutex_);
    latest_midpoint_ = msg;
    latest_midpoint_received_ = now();
  }

  void execute(std::shared_ptr<GoalHandle> gh) {
    auto result = std::make_shared<YNavGoalDecision::Result>();
    auto feedback = std::make_shared<YNavGoalDecision::Feedback>();
    std::string message;
    bool y_enabled = false;

    auto finish = [this, &y_enabled](bool disable) {
      if (disable && y_enabled) {
        std::string ignored;
        setYDeciderEnabled(false, ignored);
        y_enabled = false;
      }
    };

    clearMidpointCache();

    feedback->progress = 0.1F;
    feedback->state = "enabling_y_decider";
    gh->publish_feedback(feedback);
    if (!setYDeciderEnabled(true, message)) {
      result->success = false;
      result->message = message;
      gh->abort(result);
      return;
    }
    y_enabled = true;

    feedback->progress = 0.35F;
    feedback->state = "waiting_midpoint";
    gh->publish_feedback(feedback);
    const auto midpoint = waitForMidpoint(message);
    if (!midpoint) {
      finish(true);
      result->success = false;
      result->message = message;
      gh->abort(result);
      return;
    }
    if (gh->is_canceling()) {
      finish(true);
      result->success = false;
      result->message = "Cancelled";
      gh->canceled(result);
      return;
    }

    feedback->progress = 0.6F;
    feedback->state = "waiting_costmap";
    gh->publish_feedback(feedback);
    const auto costmap = waitForCostmap(message);
    if (!costmap) {
      finish(true);
      result->success = false;
      result->message = message;
      gh->abort(result);
      return;
    }

    feedback->progress = 0.8F;
    feedback->state = "checking_candidates";
    gh->publish_feedback(feedback);
    const auto best = chooseBestCandidate(*midpoint, **costmap, message);
    if (!best) {
      finish(true);
      result->success = false;
      result->message = message;
      gh->abort(result);
      return;
    }

    finish(true);

    result->success = true;
    result->message = "ok";
    result->nav_goal.header.stamp = now();
    result->nav_goal.header.frame_id = output_frame_;
    result->nav_goal.pose.position.x = best->output_pose.x;
    result->nav_goal.pose.position.y = best->output_pose.y;
    result->nav_goal.pose.position.z = 0.0;
    result->nav_goal.pose.orientation = quaternionFromYaw(best->output_pose.yaw);

    if (debug_log_) {
      RCLCPP_INFO(get_logger(),
                  "nav goal selected frame=%s x=%.3f y=%.3f yaw=%.2fdeg "
                  "standoff=%.2f lateral=%.2f cost=%d score=%.2f",
                  output_frame_.c_str(), best->output_pose.x, best->output_pose.y,
                  best->output_pose.yaw * 180.0 / kPi, best->standoff,
                  best->lateral, best->max_cost, best->score);
    }

    feedback->progress = 1.0F;
    feedback->state = "done";
    gh->publish_feedback(feedback);
    gh->succeed(result);
  }

  bool setYDeciderEnabled(bool enabled, std::string &message) {
    const auto timeout = secondsToNanoseconds(y_decider_enable_timeout_sec_);
    if (!y_enable_client_->wait_for_service(timeout)) {
      message = "y_decider enable service not available";
      return false;
    }

    auto req = std::make_shared<SetBool::Request>();
    req->data = enabled;
    auto future = y_enable_client_->async_send_request(req);
    if (future.wait_for(timeout) != std::future_status::ready) {
      message = enabled ? "y_decider enable timeout"
                        : "y_decider disable timeout";
      return false;
    }

    const auto resp = future.get();
    if (!resp || !resp->success) {
      message = resp ? resp->message : "y_decider enable service failed";
      return false;
    }
    return true;
  }

  void clearMidpointCache() {
    std::lock_guard<std::mutex> lk(midpoint_mutex_);
    latest_midpoint_.reset();
    latest_midpoint_received_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
  }

  std::optional<PointStampedMsg> waitForMidpoint(std::string &message) {
    const rclcpp::Time start = now();
    const rclcpp::Time deadline =
        start + rclcpp::Duration::from_seconds(midpoint_timeout_sec_);
    while (rclcpp::ok() && (deadline - now()).seconds() > 0.0) {
      {
        std::lock_guard<std::mutex> lk(midpoint_mutex_);
        if (latest_midpoint_ && latest_midpoint_received_ >= start) {
          return *latest_midpoint_;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    message = "No fresh y_decider midpoint";
    return std::nullopt;
  }

  std::optional<CostmapMsg::ConstSharedPtr> waitForCostmap(
      std::string &message) {
    const rclcpp::Time deadline =
        now() + rclcpp::Duration::from_seconds(costmap_timeout_sec_);
    while (rclcpp::ok() && (deadline - now()).seconds() > 0.0) {
      {
        std::lock_guard<std::mutex> lk(costmap_mutex_);
        if (latest_costmap_ &&
            (now() - latest_costmap_received_).seconds() <=
                costmap_timeout_sec_) {
          return latest_costmap_;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    message = "No recent local costmap";
    return std::nullopt;
  }

  std::optional<Candidate> chooseBestCandidate(const PointStampedMsg &midpoint,
                                               const CostmapMsg &costmap,
                                               std::string &message) {
    if (midpoint.header.frame_id.empty()) {
      message = "Midpoint frame_id is empty";
      return std::nullopt;
    }

    Pose2D target_in_base;
    target_in_base.x = midpoint.point.x;
    target_in_base.y = midpoint.point.y;
    target_in_base.yaw = 0.0;
    if (midpoint.header.frame_id != base_frame_) {
      try {
        target_in_base = transformPose(
            lookupTransform2D(base_frame_, midpoint.header.frame_id),
            target_in_base);
      } catch (const tf2::TransformException &ex) {
        message = std::string("Midpoint TF failed: ") + ex.what();
        return std::nullopt;
      }
    }

    const double target_x = target_in_base.x;
    const double target_y = target_in_base.y;
    if (!std::isfinite(target_x) || !std::isfinite(target_y)) {
      message = "Invalid y_decider midpoint";
      return std::nullopt;
    }

    const double target_dist = std::hypot(target_x, target_y);
    if (target_dist < min_target_distance_) {
      message = "Target midpoint too close for Nav2 goal decision";
      return std::nullopt;
    }

    if (costmap.header.frame_id.empty()) {
      message = "Costmap frame_id is empty";
      return std::nullopt;
    }

    Transform2D costmap_from_base;
    Transform2D output_from_base;
    try {
      costmap_from_base = lookupTransform2D(costmap.header.frame_id, base_frame_);
      output_from_base = lookupTransform2D(output_frame_, base_frame_);
    } catch (const tf2::TransformException &ex) {
      message = std::string("TF failed: ") + ex.what();
      return std::nullopt;
    }

    const double dir_x = target_x / target_dist;
    const double dir_y = target_y / target_dist;
    const double side_x = -dir_y;
    const double side_y = dir_x;

    std::optional<Candidate> best;
    std::string last_reject = "No candidate checked";
    for (const double standoff : standoff_candidates_) {
      for (const double lateral : lateral_candidates_) {
        Candidate cand;
        cand.standoff = standoff;
        cand.lateral = lateral;
        cand.base_pose.x = target_x - dir_x * standoff + side_x * lateral;
        cand.base_pose.y = target_y - dir_y * standoff + side_y * lateral;
        cand.base_pose.yaw =
            std::atan2(target_y - cand.base_pose.y,
                       target_x - cand.base_pose.x);

        const double along =
            cand.base_pose.x * dir_x + cand.base_pose.y * dir_y;
        if (along < min_goal_forward_distance_) {
          last_reject = "candidate behind or too close to current base";
          continue;
        }

        cand.costmap_pose = transformPose(costmap_from_base, cand.base_pose);
        std::string reject_reason;
        if (!checkFootprint(costmap, cand.costmap_pose, cand.max_cost,
                            reject_reason)) {
          last_reject = reject_reason;
          continue;
        }

        cand.output_pose = transformPose(output_from_base, cand.base_pose);
        cand.score = scoreCandidate(cand);
        if (!best || cand.score > best->score) {
          best = cand;
        }
      }
    }

    if (!best) {
      message = "No collision-free Nav2 goal candidate: " + last_reject;
      return std::nullopt;
    }
    return best;
  }

  Transform2D lookupTransform2D(const std::string &target_frame,
                                const std::string &source_frame) {
    if (target_frame == source_frame) {
      return {};
    }
    const auto tf = tf_buffer_.lookupTransform(
        target_frame, source_frame, tf2::TimePointZero,
        tf2::durationFromSec(tf_timeout_sec_));
    Transform2D out;
    out.x = tf.transform.translation.x;
    out.y = tf.transform.translation.y;
    out.yaw = yawFromQuaternion(tf.transform.rotation);
    return out;
  }

  double scoreCandidate(const Candidate &cand) const {
    const double standoff_penalty =
        std::abs(cand.standoff - preferred_standoff_distance_) /
        std::max(preferred_standoff_distance_, 0.01);
    const double lateral_penalty =
        std::abs(cand.lateral) / std::max(footprint_width_, 0.01);
    const double cost_penalty =
        static_cast<double>(cand.max_cost) /
        static_cast<double>(lethal_cost_threshold_);
    return 100.0 - 35.0 * cost_penalty - 30.0 * lateral_penalty -
           20.0 * standoff_penalty;
  }

  bool checkFootprint(const CostmapMsg &costmap, const Pose2D &pose,
                      int &max_cost, std::string &reason) const {
    if (costmap.info.resolution <= 0.0F || costmap.info.width == 0 ||
        costmap.info.height == 0 ||
        costmap.data.size() !=
            static_cast<std::size_t>(costmap.info.width) *
                static_cast<std::size_t>(costmap.info.height)) {
      reason = "invalid costmap metadata";
      return false;
    }

    const double half_x = footprint_length_ * 0.5 + footprint_padding_;
    const double half_y = footprint_width_ * 0.5 + footprint_padding_;
    const double cy = std::cos(pose.yaw);
    const double sy = std::sin(pose.yaw);

    const std::array<std::pair<double, double>, 4> corners = {
        std::make_pair(half_x, half_y),
        std::make_pair(half_x, -half_y),
        std::make_pair(-half_x, half_y),
        std::make_pair(-half_x, -half_y),
    };
    for (const auto &[lx, ly] : corners) {
      const double wx = pose.x + cy * lx - sy * ly;
      const double wy = pose.y + sy * lx + cy * ly;
      int mx = 0;
      int my = 0;
      if (!worldToMap(costmap, wx, wy, mx, my)) {
        reason = "candidate footprint leaves local costmap";
        return false;
      }
    }

    double center_mx = 0.0;
    double center_my = 0.0;
    if (!worldToMapFloat(costmap, pose.x, pose.y, center_mx, center_my)) {
      reason = "candidate center leaves local costmap";
      return false;
    }

    const int radius_cells = static_cast<int>(
        std::ceil(std::hypot(half_x, half_y) / costmap.info.resolution)) +
                             2;
    const int mx0 =
        std::max(0, static_cast<int>(std::floor(center_mx)) - radius_cells);
    const int mx1 = std::min(static_cast<int>(costmap.info.width) - 1,
                             static_cast<int>(std::ceil(center_mx)) +
                                 radius_cells);
    const int my0 =
        std::max(0, static_cast<int>(std::floor(center_my)) - radius_cells);
    const int my1 = std::min(static_cast<int>(costmap.info.height) - 1,
                             static_cast<int>(std::ceil(center_my)) +
                                 radius_cells);

    max_cost = 0;
    int checked = 0;
    for (int my = my0; my <= my1; ++my) {
      for (int mx = mx0; mx <= mx1; ++mx) {
        double wx = 0.0;
        double wy = 0.0;
        mapToWorld(costmap, mx, my, wx, wy);
        const double dx = wx - pose.x;
        const double dy = wy - pose.y;
        const double local_x = cy * dx + sy * dy;
        const double local_y = -sy * dx + cy * dy;
        if (std::abs(local_x) > half_x || std::abs(local_y) > half_y) {
          continue;
        }
        ++checked;
        const int value = static_cast<int>(
            costmap.data[static_cast<std::size_t>(my) * costmap.info.width +
                         static_cast<std::size_t>(mx)]);
        if (value < 0) {
          if (unknown_is_blocked_) {
            reason = "candidate footprint touches unknown costmap";
            return false;
          }
          continue;
        }
        max_cost = std::max(max_cost, value);
        if (value >= lethal_cost_threshold_) {
          reason = "candidate footprint touches lethal/inflated cost";
          return false;
        }
      }
    }

    if (checked == 0) {
      reason = "candidate footprint has no checked cells";
      return false;
    }
    return true;
  }

  bool worldToMapFloat(const CostmapMsg &costmap, double wx, double wy,
                       double &mx, double &my) const {
    const auto &origin = costmap.info.origin;
    const double yaw = yawFromQuaternion(origin.orientation);
    const double dx = wx - origin.position.x;
    const double dy = wy - origin.position.y;
    const double c = std::cos(-yaw);
    const double s = std::sin(-yaw);
    mx = (c * dx - s * dy) / costmap.info.resolution;
    my = (s * dx + c * dy) / costmap.info.resolution;
    return mx >= 0.0 && my >= 0.0 &&
           mx < static_cast<double>(costmap.info.width) &&
           my < static_cast<double>(costmap.info.height);
  }

  bool worldToMap(const CostmapMsg &costmap, double wx, double wy, int &mx,
                  int &my) const {
    double mxf = 0.0;
    double myf = 0.0;
    if (!worldToMapFloat(costmap, wx, wy, mxf, myf)) {
      return false;
    }
    mx = static_cast<int>(std::floor(mxf));
    my = static_cast<int>(std::floor(myf));
    return mx >= 0 && my >= 0 &&
           mx < static_cast<int>(costmap.info.width) &&
           my < static_cast<int>(costmap.info.height);
  }

  void mapToWorld(const CostmapMsg &costmap, int mx, int my, double &wx,
                  double &wy) const {
    const auto &origin = costmap.info.origin;
    const double yaw = yawFromQuaternion(origin.orientation);
    const double gx = (static_cast<double>(mx) + 0.5) * costmap.info.resolution;
    const double gy = (static_cast<double>(my) + 0.5) * costmap.info.resolution;
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    wx = origin.position.x + c * gx - s * gy;
    wy = origin.position.y + s * gx + c * gy;
  }

  std::string action_name_;
  std::string y_decider_enable_service_name_;
  std::string y_decider_midpoint_topic_;
  std::string costmap_topic_;
  std::string base_frame_;
  std::string output_frame_;
  double y_decider_enable_timeout_sec_ = 1.0;
  double midpoint_timeout_sec_ = 2.0;
  double costmap_timeout_sec_ = 1.0;
  double tf_timeout_sec_ = 0.2;
  double min_target_distance_ = 0.2;
  double min_goal_forward_distance_ = 0.03;
  double preferred_standoff_distance_ = 0.55;
  std::vector<double> standoff_candidates_;
  std::vector<double> lateral_candidates_;
  double footprint_length_ = 0.70;
  double footprint_width_ = 0.65;
  double footprint_padding_ = 0.05;
  int lethal_cost_threshold_ = 65;
  bool unknown_is_blocked_ = true;
  bool debug_log_ = true;

  rclcpp::Subscription<CostmapMsg>::SharedPtr costmap_sub_;
  rclcpp::Subscription<PointStampedMsg>::SharedPtr midpoint_sub_;
  rclcpp::Client<SetBool>::SharedPtr y_enable_client_;
  rclcpp_action::Server<YNavGoalDecision>::SharedPtr server_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex costmap_mutex_;
  CostmapMsg::ConstSharedPtr latest_costmap_;
  rclcpp::Time latest_costmap_received_{0, 0, RCL_ROS_TIME};

  std::mutex midpoint_mutex_;
  PointStampedMsg::ConstSharedPtr latest_midpoint_;
  rclcpp::Time latest_midpoint_received_{0, 0, RCL_ROS_TIME};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<YNavGoalDeciderNode>());
  rclcpp::shutdown();
  return 0;
}
