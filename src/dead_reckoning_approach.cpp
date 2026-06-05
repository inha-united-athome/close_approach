#include "close_approach/dead_reckoning_approach.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <functional>
#include <iomanip>
#include <sstream>
#include <thread>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace {

constexpr double kPi = 3.14159265358979323846;

double normalizeAngle(double angle) {
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle < -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

double approachValue(double current, double target, double max_delta) {
  if (current < target) {
    return std::min(current + max_delta, target);
  }
  return std::max(current - max_delta, target);
}

std::string currentTimeForFilename() {
  const auto now = std::chrono::system_clock::now();
  const auto now_time_t = std::chrono::system_clock::to_time_t(now);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

  std::tm local_time{};
  localtime_r(&now_time_t, &local_time);

  std::ostringstream stamp;
  stamp << std::put_time(&local_time, "%Y%m%d_%H%M%S") << "_"
        << std::setw(3) << std::setfill('0') << ms.count();
  return stamp.str();
}

} // namespace

DeadReckoningApproachNode::DeadReckoningApproachNode()
    : Node("dead_reckoning_approach_node"),
      tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
  this->declare_parameter<std::string>("action_name",
                                       "dead_reckoning_approach");
  this->declare_parameter<std::string>("odom_frame", "odom");
  this->declare_parameter<std::string>("robot_frame", "base_nav");
  this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
  this->declare_parameter<float>("max_goal_distance", 0.15F);
  this->declare_parameter<float>("min_goal_distance", 0.01F);
  this->declare_parameter<float>("goal_tolerance", 0.01F);
  this->declare_parameter<float>("max_v", 0.04F);
  this->declare_parameter<float>("min_v", 0.01F);
  this->declare_parameter<float>("max_w", 0.20F);
  this->declare_parameter<float>("accel_limit", 0.05F);
  this->declare_parameter<float>("decel_limit", 0.08F);
  this->declare_parameter<float>("kp_distance", 0.8F);
  this->declare_parameter<float>("kp_lateral", 1.0F);
  this->declare_parameter<float>("kp_yaw", 1.2F);
  this->declare_parameter<float>("slow_down_distance", 0.08F);
  this->declare_parameter<float>("control_rate_hz", 20.0F);
  this->declare_parameter<float>("timeout_margin_sec", 3.0F);
  this->declare_parameter<bool>("log_enabled", true);
  this->declare_parameter<std::string>(
      "log_dir", "/home/thor/inha_logs/module/close_approach/dead_reckoning");

  this->get_parameter("action_name", action_name_);
  this->get_parameter("odom_frame", odom_frame_);
  this->get_parameter("robot_frame", robot_frame_);
  this->get_parameter("cmd_vel_topic", cmd_vel_topic_);
  this->get_parameter("max_goal_distance", max_goal_distance_);
  this->get_parameter("min_goal_distance", min_goal_distance_);
  this->get_parameter("goal_tolerance", goal_tolerance_);
  this->get_parameter("max_v", max_v_);
  this->get_parameter("min_v", min_v_);
  this->get_parameter("max_w", max_w_);
  this->get_parameter("accel_limit", accel_limit_);
  this->get_parameter("decel_limit", decel_limit_);
  this->get_parameter("kp_distance", kp_distance_);
  this->get_parameter("kp_lateral", kp_lateral_);
  this->get_parameter("kp_yaw", kp_yaw_);
  this->get_parameter("slow_down_distance", slow_down_distance_);
  this->get_parameter("control_rate_hz", control_rate_hz_);
  this->get_parameter("timeout_margin_sec", timeout_margin_sec_);
  this->get_parameter("log_enabled", log_enabled_);
  std::string log_dir;
  this->get_parameter("log_dir", log_dir);
  log_dir_ = log_dir;

  if (log_enabled_) {
    std::filesystem::create_directories(log_dir_);
    RCLCPP_INFO(this->get_logger(), "Dead reckoning logs will be saved to %s",
                log_dir_.c_str());
  }

  cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());

  action_server_ = rclcpp_action::create_server<ApproachAction>(
      this, action_name_,
      std::bind(&DeadReckoningApproachNode::handle_goal, this,
                std::placeholders::_1, std::placeholders::_2),
      std::bind(&DeadReckoningApproachNode::handle_cancel, this,
                std::placeholders::_1),
      std::bind(&DeadReckoningApproachNode::handle_accepted, this,
                std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(),
              "DeadReckoningApproachNode ready: action=%s max_goal=%.3fm",
              action_name_.c_str(), max_goal_distance_);
}

rclcpp_action::GoalResponse DeadReckoningApproachNode::handle_goal(
    const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const ApproachAction::Goal> goal) {
  (void)uuid;
  if (active_) {
    RCLCPP_WARN(this->get_logger(),
                "Rejecting dead reckoning approach goal: already active");
    return rclcpp_action::GoalResponse::REJECT;
  }
  if (!goal || !std::isfinite(goal->goal_distance) ||
      std::abs(goal->goal_distance) < min_goal_distance_) {
    RCLCPP_WARN(this->get_logger(),
                "Rejecting dead reckoning approach goal: invalid "
                "goal_distance=%.3f",
                goal ? goal->goal_distance : 0.0F);
    return rclcpp_action::GoalResponse::REJECT;
  }

  const float clamped =
      std::clamp(goal->goal_distance, -max_goal_distance_, max_goal_distance_);
  RCLCPP_INFO(this->get_logger(),
              "Received dead reckoning approach goal: requested=%.3fm "
              "clamped=%.3fm",
              goal->goal_distance, clamped);
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse DeadReckoningApproachNode::handle_cancel(
    const std::shared_ptr<GoalHandleApproach> goal_handle) {
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "Received dead reckoning approach cancel");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void DeadReckoningApproachNode::handle_accepted(
    const std::shared_ptr<GoalHandleApproach> goal_handle) {
  std::thread(std::bind(&DeadReckoningApproachNode::execute, this,
                        goal_handle))
      .detach();
}

bool DeadReckoningApproachNode::getRobotPoseInOdom(double &x, double &y,
                                                   double &yaw) {
  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_.lookupTransform(odom_frame_, robot_frame_,
                                    tf2::TimePointZero,
                                    tf2::durationFromSec(0.1));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "TF %s->%s lookup failed: %s", odom_frame_.c_str(),
                         robot_frame_.c_str(), ex.what());
    return false;
  }

  x = tf.transform.translation.x;
  y = tf.transform.translation.y;
  yaw = tf2::getYaw(tf.transform.rotation);
  return true;
}

void DeadReckoningApproachNode::publishStop() {
  geometry_msgs::msg::Twist stop;
  stop.linear.x = 0.0;
  stop.angular.z = 0.0;
  cmd_vel_publisher_->publish(stop);
}

void DeadReckoningApproachNode::openActionLog(float requested_distance,
                                              float clamped_distance) {
  if (!log_enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(log_mutex_);
  closeActionLog();

  std::filesystem::create_directories(log_dir_);
  action_log_path_ =
      log_dir_ / ("dead_reckoning_" + currentTimeForFilename() + ".csv");
  action_log_file_.open(action_log_path_, std::ios::out | std::ios::trunc);
  if (!action_log_file_.is_open()) {
    RCLCPP_WARN(this->get_logger(), "Failed to open action log: %s",
                action_log_path_.c_str());
    return;
  }

  action_log_file_ << "# requested_distance_m," << requested_distance << '\n';
  action_log_file_ << "# clamped_distance_m," << clamped_distance << '\n';
  action_log_file_ << "t_sec,travelled_m,remaining_m,lateral_m,yaw_error_rad,"
                      "v_target,v_cmd,w_cmd\n";
  action_log_file_.flush();
  RCLCPP_INFO(this->get_logger(), "Dead reckoning action log: %s",
              action_log_path_.c_str());
}

void DeadReckoningApproachNode::closeActionLog() {
  if (action_log_file_.is_open()) {
    action_log_file_.flush();
    action_log_file_.close();
  }
}

void DeadReckoningApproachNode::writeActionLog(const std::string &line) {
  if (!log_enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(log_mutex_);
  if (action_log_file_.is_open()) {
    action_log_file_ << line << '\n';
  }
}

void DeadReckoningApproachNode::execute(
    const std::shared_ptr<GoalHandleApproach> goal_handle) {
  if (active_.exchange(true)) {
    auto result = std::make_shared<ApproachAction::Result>();
    result->success = false;
    result->success_message = "Dead reckoning approach already active";
    goal_handle->abort(result);
    return;
  }

  auto finish = [this]() {
    publishStop();
    closeActionLog();
    active_ = false;
  };

  auto result = std::make_shared<ApproachAction::Result>();
  auto feedback = std::make_shared<ApproachAction::Feedback>();
  const auto goal = goal_handle->get_goal();
  const float requested_distance = goal->goal_distance;
  const float target_distance =
      std::clamp(requested_distance, -max_goal_distance_, max_goal_distance_);
  const double target_abs_distance = std::abs(target_distance);
  const double direction = target_distance >= 0.0F ? 1.0 : -1.0;

  openActionLog(requested_distance, target_distance);

  double start_x = 0.0;
  double start_y = 0.0;
  double start_yaw = 0.0;
  if (!getRobotPoseInOdom(start_x, start_y, start_yaw)) {
    finish();
    result->success = false;
    result->success_message = "Failed to get initial odom pose";
    goal_handle->abort(result);
    return;
  }

  const double forward_x = std::cos(start_yaw);
  const double forward_y = std::sin(start_yaw);
  const double travel_x = direction * forward_x;
  const double travel_y = direction * forward_y;
  const rclcpp::Time start_time = this->now();
  rclcpp::Time previous_time = start_time;
  const double expected_duration =
      target_abs_distance / std::max(static_cast<double>(min_v_), 1e-3);
  const double timeout_sec = expected_duration + timeout_margin_sec_;
  double v_cmd = 0.0;

  RCLCPP_INFO(this->get_logger(),
              "Dead reckoning approach started: distance=%.3fm direction=%s "
              "start=(%.3f, %.3f, %.3f) timeout=%.2fs",
              target_abs_distance, direction > 0.0 ? "forward" : "backward",
              start_x, start_y, start_yaw, timeout_sec);

  rclcpp::Rate loop_rate(control_rate_hz_);
  while (rclcpp::ok()) {
    if (goal_handle->is_canceling()) {
      finish();
      result->success = false;
      result->success_message = "Dead reckoning approach canceled";
      goal_handle->canceled(result);
      RCLCPP_INFO(this->get_logger(), "Dead reckoning approach canceled");
      return;
    }

    const rclcpp::Time now = this->now();
    const double elapsed_sec = (now - start_time).seconds();
    const double dt = std::max((now - previous_time).seconds(), 0.0);
    previous_time = now;

    if (elapsed_sec > timeout_sec) {
      finish();
      result->success = false;
      result->success_message = "Dead reckoning approach timeout";
      goal_handle->abort(result);
      RCLCPP_WARN(this->get_logger(),
                  "Dead reckoning approach timeout after %.2fs", elapsed_sec);
      return;
    }

    double rx = 0.0;
    double ry = 0.0;
    double odom_yaw = 0.0;
    if (!getRobotPoseInOdom(rx, ry, odom_yaw)) {
      loop_rate.sleep();
      continue;
    }

    const double dx = rx - start_x;
    const double dy = ry - start_y;
    const double travelled = dx * travel_x + dy * travel_y;
    const double remaining = target_abs_distance - travelled;
    const double lateral = -dx * forward_y + dy * forward_x;
    const double yaw_error = normalizeAngle(start_yaw - odom_yaw);

    if (remaining <= goal_tolerance_) {
      finish();
      result->success = true;
      result->success_message = "Dead reckoning approach reached target";
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(),
                  "Dead reckoning approach done: travelled=%.3fm "
                  "remaining=%.3fm lateral=%.3fm yaw_error=%.3f",
                  travelled, remaining, lateral, yaw_error);
      return;
    }

    double target_speed = kp_distance_ * std::max(remaining, 0.0);
    if (slow_down_distance_ > 1e-6F) {
      target_speed = std::min(
          target_speed,
          static_cast<double>(max_v_) *
              std::clamp(remaining / static_cast<double>(slow_down_distance_),
                         0.0, 1.0));
    }
    if (remaining > goal_tolerance_) {
      target_speed = std::clamp(target_speed, static_cast<double>(min_v_),
                                static_cast<double>(max_v_));
    }

    const double v_target = direction * target_speed;
    const bool speeding_up = std::abs(v_target) > std::abs(v_cmd);
    const double limit = speeding_up ? accel_limit_ : decel_limit_;
    v_cmd = approachValue(v_cmd, v_target, static_cast<double>(limit) * dt);

    double w_cmd = kp_yaw_ * yaw_error +
                   direction * static_cast<double>(kp_lateral_) * lateral;
    w_cmd = std::clamp(w_cmd, -static_cast<double>(max_w_),
                       static_cast<double>(max_w_));

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = v_cmd;
    cmd.angular.z = w_cmd;
    cmd_vel_publisher_->publish(cmd);

    feedback->x_error = static_cast<float>(remaining);
    feedback->y_error = static_cast<float>(lateral);
    feedback->theta_error = static_cast<float>(yaw_error);
    goal_handle->publish_feedback(feedback);

    std::ostringstream line;
    line << std::fixed << std::setprecision(6) << elapsed_sec << ","
         << travelled << "," << remaining << "," << lateral << ","
         << yaw_error << "," << v_target << "," << v_cmd << "," << w_cmd;
    writeActionLog(line.str());

    RCLCPP_DEBUG(this->get_logger(),
                 "dead_reckoning: travelled=%.3f remaining=%.3f lateral=%.3f "
                 "yaw=%.3f v_target=%.3f v=%.3f w=%.3f",
                 travelled, remaining, lateral, yaw_error, v_target, v_cmd,
                 w_cmd);

    loop_rate.sleep();
  }

  finish();
  result->success = false;
  result->success_message = "Node shutdown";
  goal_handle->abort(result);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DeadReckoningApproachNode>());
  rclcpp::shutdown();
  return 0;
}
