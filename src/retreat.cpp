#include "close_approach/retreat.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

#include <tf2/utils.h>

/*
RetreatNode
- ApproachNode가 latched로 발행한 /approach/trail (odom 프레임 PoseStamped 시퀀스)을
  받아두었다가, Retreat 액션 호출 시 그것을 역순으로 따라 후진한다.
- 제어는 pure pursuit. 후진이므로 v_x = -retreat_speed_ 고정, ω 만 조향에 사용.
- Adaptive lookahead + terminal mode로 짧은 trail / 끝점 부근에서 안전하게 정지.
*/

RetreatNode::RetreatNode()
    : Node("retreat_node"),
      tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {

  this->declare_parameter<std::string>("odom_frame", "odom");
  this->declare_parameter<std::string>("robot_frame", "base_nav");
  this->declare_parameter<float>("retreat_speed", 0.07F);
  this->declare_parameter<float>("lookahead_max", 0.15F);
  this->declare_parameter<float>("lookahead_min", 0.05F);
  this->declare_parameter<float>("terminal_threshold", 0.05F);
  this->declare_parameter<float>("min_trail_length", 0.05F);
  this->declare_parameter<float>("max_retreat_distance", 0.30F);
  this->declare_parameter<float>("w_max", 0.8F);
  this->declare_parameter<float>("control_rate_hz", 10.0F);
  this->declare_parameter<float>("timeout_margin_sec", 3.0F);

  this->get_parameter("odom_frame", odom_frame_);
  this->get_parameter("robot_frame", robot_frame_);
  this->get_parameter("retreat_speed", retreat_speed_);
  this->get_parameter("lookahead_max", lookahead_max_);
  this->get_parameter("lookahead_min", lookahead_min_);
  this->get_parameter("terminal_threshold", terminal_threshold_);
  this->get_parameter("min_trail_length", min_trail_length_);
  this->get_parameter("max_retreat_distance", max_retreat_distance_);
  this->get_parameter("w_max", w_max_);
  this->get_parameter("control_rate_hz", control_rate_hz_);
  this->get_parameter("timeout_margin_sec", timeout_margin_sec_);

  auto qos_reliable = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
  cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel", qos_reliable);

  // ApproachNode가 transient_local로 latched 발행하므로 동일 QoS로 구독.
  rclcpp::QoS trail_qos(rclcpp::KeepLast(1));
  trail_qos.reliable();
  trail_qos.transient_local();
  trail_subscriber_ = this->create_subscription<nav_msgs::msg::Path>(
      "/approach/trail", trail_qos,
      std::bind(&RetreatNode::trailCallback, this, std::placeholders::_1));

  action_server_ = rclcpp_action::create_server<RetreatAction>(
      this, "retreat",
      std::bind(&RetreatNode::handle_goal, this, std::placeholders::_1,
                std::placeholders::_2),
      std::bind(&RetreatNode::handle_cancel, this, std::placeholders::_1),
      std::bind(&RetreatNode::handle_accepted, this, std::placeholders::_1));
}

void RetreatNode::trailCallback(const nav_msgs::msg::Path::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(trail_mutex_);
  latest_trail_ = *msg;
  has_trail_ = !msg->poses.empty();
  RCLCPP_INFO(this->get_logger(), "Received trail: %zu poses (frame=%s)",
              msg->poses.size(), msg->header.frame_id.c_str());
}

rclcpp_action::GoalResponse
RetreatNode::handle_goal(const rclcpp_action::GoalUUID &uuid,
                         std::shared_ptr<const RetreatAction::Goal> goal) {
  (void)uuid;
  (void)goal;
  if (active_) {
    RCLCPP_WARN(this->get_logger(),
                "Rejecting retreat goal: retreat is already active");
    return rclcpp_action::GoalResponse::REJECT;
  }
  RCLCPP_INFO(this->get_logger(), "Received retreat goal");
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse RetreatNode::handle_cancel(
    const std::shared_ptr<GoalHandleRetreat> goal_handle) {
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "Received retreat cancel");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void RetreatNode::handle_accepted(
    const std::shared_ptr<GoalHandleRetreat> goal_handle) {
  std::thread(std::bind(&RetreatNode::execute, this, goal_handle)).detach();
}

bool RetreatNode::getRobotPoseInOdom(double &x, double &y, double &yaw) {
  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_.lookupTransform(odom_frame_, robot_frame_,
                                    tf2::TimePointZero,
                                    tf2::durationFromSec(0.1));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(this->get_logger(), "TF %s->%s lookup failed: %s",
                odom_frame_.c_str(), robot_frame_.c_str(), ex.what());
    return false;
  }
  x = tf.transform.translation.x;
  y = tf.transform.translation.y;
  yaw = tf2::getYaw(tf.transform.rotation);
  return true;
}

void RetreatNode::publishStop() {
  geometry_msgs::msg::Twist stop;
  stop.linear.x = 0.0;
  stop.angular.z = 0.0;
  cmd_vel_publisher_->publish(stop);
}

void RetreatNode::execute(
    const std::shared_ptr<GoalHandleRetreat> goal_handle) {
  auto result = std::make_shared<RetreatAction::Result>();
  auto feedback = std::make_shared<RetreatAction::Feedback>();

  if (active_.exchange(true)) {
    result->success = false;
    result->success_message = "Retreat already active";
    goal_handle->abort(result);
    return;
  }

  auto finish = [this]() {
    publishStop();
    active_ = false;
  };

  // 1) trail 스냅샷 + 역순 변환
  std::vector<geometry_msgs::msg::PoseStamped> rev;
  {
    std::lock_guard<std::mutex> lock(trail_mutex_);
    if (!has_trail_ || latest_trail_.poses.empty()) {
      RCLCPP_WARN(this->get_logger(), "No trail available for retreat");
      result->success = false;
      result->success_message = "No trail available";
      finish();
      goal_handle->abort(result);
      return;
    }
    rev.assign(latest_trail_.poses.rbegin(), latest_trail_.poses.rend());
  }

  // 2) 누적 arc-length 계산 (rev[0]에서 rev[i]까지)
  std::vector<double> cum(rev.size(), 0.0);
  for (std::size_t i = 1; i < rev.size(); ++i) {
    const double dx = rev[i].pose.position.x - rev[i - 1].pose.position.x;
    const double dy = rev[i].pose.position.y - rev[i - 1].pose.position.y;
    cum[i] = cum[i - 1] + std::sqrt(dx * dx + dy * dy);
  }
  const double total_length = cum.back();

  if (total_length < min_trail_length_) {
    RCLCPP_INFO(this->get_logger(),
                "Trail too short (%.3fm < %.3fm). Nothing to retreat.",
                total_length, min_trail_length_);
    finish();
    result->success = true;
    result->success_message = "Trail too short, no retreat needed";
    goal_handle->succeed(result);
    return;
  }

  if (retreat_speed_ <= 0.0F) {
    RCLCPP_ERROR(this->get_logger(),
                 "Invalid retreat_speed %.3f. Retreat aborted.",
                 retreat_speed_);
    finish();
    result->success = false;
    result->success_message = "Invalid retreat speed";
    goal_handle->abort(result);
    return;
  }

  if (max_retreat_distance_ <= 0.0F) {
    RCLCPP_ERROR(this->get_logger(),
                 "Invalid max_retreat_distance %.3f. Retreat aborted.",
                 max_retreat_distance_);
    finish();
    result->success = false;
    result->success_message = "Invalid maximum retreat distance";
    goal_handle->abort(result);
    return;
  }

  // 3) 제어 루프
  rclcpp::Rate loop_rate(control_rate_hz_);
  std::size_t closest_idx = 0;
  const double v = -static_cast<double>(retreat_speed_);
  const double allowed_distance =
      std::min(total_length, static_cast<double>(max_retreat_distance_));
  const double timeout_sec =
      allowed_distance / static_cast<double>(retreat_speed_) +
      std::max(0.0, static_cast<double>(timeout_margin_sec_));
  const auto control_start = std::chrono::steady_clock::now();
  const auto &before_end = rev[rev.size() - 2].pose.position;
  const auto &end = rev.back().pose.position;
  const double end_segment_x = end.x - before_end.x;
  const double end_segment_y = end.y - before_end.y;
  double travelled_distance = 0.0;
  double previous_rx = 0.0;
  double previous_ry = 0.0;
  bool previous_pose_available = false;

  RCLCPP_INFO(this->get_logger(),
              "Retreat started: trail=%.3fm timeout=%.2fs",
              total_length, timeout_sec);

  while (rclcpp::ok()) {
    if (goal_handle->is_canceling()) {
      finish();
      result->success = false;
      result->success_message = "Retreat canceled";
      goal_handle->canceled(result);
      RCLCPP_INFO(this->get_logger(), "Retreat canceled");
      return;
    }

    const double elapsed_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      control_start)
            .count();
    if (elapsed_sec > timeout_sec) {
      finish();
      result->success = false;
      result->success_message = "Retreat timeout";
      goal_handle->abort(result);
      RCLCPP_WARN(this->get_logger(),
                  "Retreat timeout after %.2fs (limit=%.2fs)", elapsed_sec,
                  timeout_sec);
      return;
    }

    double rx, ry, ryaw;
    if (!getRobotPoseInOdom(rx, ry, ryaw)) {
      loop_rate.sleep();
      continue;
    }

    if (previous_pose_available) {
      const double step_x = rx - previous_rx;
      const double step_y = ry - previous_ry;
      travelled_distance += std::sqrt(step_x * step_x + step_y * step_y);
    }
    previous_rx = rx;
    previous_ry = ry;
    previous_pose_available = true;

    if (travelled_distance >= allowed_distance) {
      finish();
      result->success = true;
      result->success_message = "Retreat reached allowed distance";
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(),
                  "Retreat reached allowed distance: %.3fm >= %.3fm",
                  travelled_distance, allowed_distance);
      return;
    }

    // closest point: 역행 방지를 위해 현재 인덱스부터 앞으로만 탐색.
    double best_d2 = std::numeric_limits<double>::infinity();
    std::size_t best_i = closest_idx;
    const std::size_t search_end =
        std::min(closest_idx + 10, rev.size() - 1);
    for (std::size_t i = closest_idx; i <= search_end; ++i) {
      const double dx = rev[i].pose.position.x - rx;
      const double dy = rev[i].pose.position.y - ry;
      const double d2 = dx * dx + dy * dy;
      if (d2 < best_d2) {
        best_d2 = d2;
        best_i = i;
      }
    }
    closest_idx = best_i;

    const double remaining = total_length - cum[closest_idx];

    // 종료 판정: 끝점과 충분히 가까우면 success.
    const double dx_end = rev.back().pose.position.x - rx;
    const double dy_end = rev.back().pose.position.y - ry;
    const double dist_end = std::sqrt(dx_end * dx_end + dy_end * dy_end);
    if (dist_end < terminal_threshold_) {
      finish();
      result->success = true;
      result->success_message = "Retreat reached trail start";
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Retreat done (dist_end=%.3fm)",
                  dist_end);
      return;
    }

    // 끝점을 지나쳤다면 정확한 tolerance에 못 들어가도 후진 목적은 달성한 것으로 본다.
    const double past_end =
        (rx - end.x) * end_segment_x + (ry - end.y) * end_segment_y;
    if (closest_idx + 1 == rev.size() && past_end > 0.0) {
      finish();
      result->success = true;
      result->success_message = "Retreat passed trail start";
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(),
                  "Retreat passed trail start (dist_end=%.3fm)", dist_end);
      return;
    }

    // adaptive lookahead: 끝까지 남은 길이에 맞춰 축소.
    double Ld = std::min(static_cast<double>(lookahead_max_), remaining);
    Ld = std::max(static_cast<double>(lookahead_min_), Ld);

    // lookahead point 찾기: closest_idx부터 cum 차이가 Ld 이상이 되는 첫 점.
    geometry_msgs::msg::Point target;
    bool terminal_mode = (remaining < terminal_threshold_);
    if (terminal_mode) {
      target = rev.back().pose.position;
    } else {
      const double cum_target = cum[closest_idx] + Ld;
      std::size_t li = closest_idx;
      while (li + 1 < rev.size() && cum[li + 1] < cum_target) {
        ++li;
      }
      if (li + 1 >= rev.size()) {
        target = rev.back().pose.position;
      } else {
        // (cum[li], cum[li+1]) 구간에서 cum_target 위치로 선형 보간.
        const double seg = cum[li + 1] - cum[li];
        const double t = seg > 1e-6 ? (cum_target - cum[li]) / seg : 0.0;
        const auto &a = rev[li].pose.position;
        const auto &b = rev[li + 1].pose.position;
        target.x = a.x + t * (b.x - a.x);
        target.y = a.y + t * (b.y - a.y);
        target.z = 0.0;
      }
    }

    // base 프레임으로 변환.
    const double dx_w = target.x - rx;
    const double dy_w = target.y - ry;
    const double cy = std::cos(ryaw);
    const double sy = std::sin(ryaw);
    const double dx_b = cy * dx_w + sy * dy_w;
    const double dy_b = -sy * dx_w + cy * dy_w;
    const double L2 = dx_b * dx_b + dy_b * dy_b;

    double v_cmd = v;

    double w_cmd = 0.0;
    if (L2 > 1e-6) {
      // pure pursuit 곡률: κ = 2 * y / L^2, ω = v * κ.
      // v < 0 (후진)이어도 동일 식으로 타겟점이 뒤에 있을 때 수렴.
      w_cmd = 2.0 * v_cmd * dy_b / L2;
    }
    w_cmd = std::clamp(w_cmd, -static_cast<double>(w_max_),
                       static_cast<double>(w_max_));

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = v_cmd;
    cmd.angular.z = w_cmd;
    cmd_vel_publisher_->publish(cmd);

    feedback->distance_remaining = static_cast<float>(remaining);
    feedback->progress = static_cast<float>(
        std::clamp(1.0 - remaining / total_length, 0.0, 1.0));
    goal_handle->publish_feedback(feedback);

    RCLCPP_DEBUG(this->get_logger(),
                 "retreat: idx=%zu/%zu rem=%.3fm Ld=%.3fm v=%.3f w=%.3f",
                 closest_idx, rev.size(), remaining, Ld, v_cmd, w_cmd);

    loop_rate.sleep();
  }

  finish();
  result->success = false;
  result->success_message = "Node shutdown";
  goal_handle->abort(result);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RetreatNode>());
  rclcpp::shutdown();
  return 0;
}
