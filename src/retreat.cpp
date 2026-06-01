#include "close_approach/retreat.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>
#include <thread>
#include <vector>

#include <tf2/utils.h>

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

double pointToSegmentDistance(double px, double py,
                              const geometry_msgs::msg::Point &a,
                              const geometry_msgs::msg::Point &b) {
  const double segment_x = b.x - a.x;
  const double segment_y = b.y - a.y;
  const double segment_length_squared =
      segment_x * segment_x + segment_y * segment_y;
  if (segment_length_squared <= 1e-12) {
    return std::hypot(px - a.x, py - a.y);
  }

  const double projection =
      std::clamp(((px - a.x) * segment_x + (py - a.y) * segment_y) /
                     segment_length_squared,
                 0.0, 1.0);
  const double closest_x = a.x + projection * segment_x;
  const double closest_y = a.y + projection * segment_y;
  return std::hypot(px - closest_x, py - closest_y);
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
  this->declare_parameter<bool>("log_enabled", true);
  this->declare_parameter<std::string>(
      "log_dir", "/home/thor/inha_logs/module/close_approach/retreat");

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
  this->get_parameter("log_enabled", log_enabled_);
  std::string log_dir;
  this->get_parameter("log_dir", log_dir);
  log_dir_ = log_dir;

  if (log_enabled_) {
    std::error_code error;
    std::filesystem::create_directories(log_dir_, error);
    if (error) {
      RCLCPP_WARN(this->get_logger(),
                  "Failed to create retreat log directory %s: %s",
                  log_dir_.c_str(), error.message().c_str());
    } else {
      RCLCPP_INFO(this->get_logger(), "Retreat logs will be saved to %s",
                  log_dir_.c_str());
    }
  }

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

void RetreatNode::openActionLog() {
  if (!log_enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(log_mutex_);
  if (action_trace_log_file_.is_open()) {
    action_trace_log_file_.flush();
    action_trace_log_file_.close();
  }

  action_log_dir_ = log_dir_ / ("action_" + currentTimeForFilename());
  action_trace_log_path_ = action_log_dir_ / "trace.csv";
  action_summary_log_path_ = action_log_dir_ / "summary.csv";
  std::error_code error;
  std::filesystem::create_directories(action_log_dir_, error);
  if (error) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to create retreat action log directory %s: %s",
                action_log_dir_.c_str(), error.message().c_str());
    return;
  }

  action_trace_log_file_.open(action_trace_log_path_,
                              std::ios::out | std::ios::trunc);
  if (!action_trace_log_file_.is_open()) {
    RCLCPP_WARN(this->get_logger(), "Failed to open retreat trace log: %s",
                action_trace_log_path_.c_str());
    return;
  }

  action_trace_log_file_
      << "t_sec,robot_x_m,robot_y_m,robot_yaw_rad,closest_idx,"
         "trail_pose_count,trail_progress_m,remaining_m,travelled_m,"
         "allowed_distance_m,reached_allowed_distance,"
         "exceeded_allowed_distance,exceeded_max_retreat_distance,"
         "cross_track_error_m,yaw_error_rad,dist_end_m,lookahead_m,target_x_m,"
         "target_y_m,v_cmd_mps,w_cmd_radps,terminal_mode,event\n";
  action_trace_log_file_.flush();
  RCLCPP_INFO(this->get_logger(), "Retreat action logs will be saved to %s",
              action_log_dir_.c_str());
}

void RetreatNode::closeActionLog() {
  std::lock_guard<std::mutex> lock(log_mutex_);
  if (action_trace_log_file_.is_open()) {
    action_trace_log_file_.flush();
    action_trace_log_file_.close();
  }
}

void RetreatNode::writeActionTrace(const std::string &line) {
  if (!log_enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(log_mutex_);
  if (action_trace_log_file_.is_open()) {
    action_trace_log_file_ << line << '\n';
  }
}

void RetreatNode::writeTrailSnapshot(
    const std::vector<geometry_msgs::msg::PoseStamped> &trail,
    const std::vector<double> &cumulative_distance) {
  if (!log_enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(log_mutex_);
  const auto trail_log_path = action_log_dir_ / "trail.csv";
  std::ofstream trail_log(trail_log_path, std::ios::out | std::ios::trunc);
  if (!trail_log.is_open()) {
    RCLCPP_WARN(this->get_logger(), "Failed to open retreat trail log: %s",
                trail_log_path.c_str());
    return;
  }

  trail_log << "idx,x_m,y_m,yaw_rad,cumulative_m\n";
  trail_log << std::fixed << std::setprecision(6);
  for (std::size_t i = 0; i < trail.size(); ++i) {
    const auto &pose = trail[i].pose;
    trail_log << i << "," << pose.position.x << "," << pose.position.y << ","
              << tf2::getYaw(pose.orientation) << "," << cumulative_distance[i]
              << '\n';
  }
}

void RetreatNode::writeActionSummary(const std::string &status, bool success,
                                     double elapsed_sec,
                                     std::size_t trail_pose_count,
                                     double trail_length,
                                     double allowed_distance,
                                     double travelled_distance,
                                     double final_dist_end,
                                     const TrackingStats &tracking_stats) {
  if (!log_enabled_) {
    return;
  }

  const bool has_distance_limit =
      std::isfinite(allowed_distance) && allowed_distance > 0.0;
  const bool reached_allowed_distance =
      has_distance_limit && travelled_distance >= allowed_distance;
  const double allowed_distance_overshoot =
      has_distance_limit
          ? std::max(0.0, travelled_distance - allowed_distance)
          : 0.0;
  const bool exceeded_allowed_distance = allowed_distance_overshoot > 1e-6;
  const bool has_max_retreat_distance =
      std::isfinite(max_retreat_distance_) && max_retreat_distance_ > 0.0F;
  const double max_retreat_distance_overshoot =
      has_max_retreat_distance
          ? std::max(0.0, travelled_distance - max_retreat_distance_)
          : 0.0;
  const bool exceeded_max_retreat_distance =
      max_retreat_distance_overshoot > 1e-6;
  const double mean_cross_track_error =
      tracking_stats.samples > 0
          ? tracking_stats.cross_track_error_sum / tracking_stats.samples
          : std::numeric_limits<double>::quiet_NaN();
  const double mean_yaw_error =
      tracking_stats.samples > 0
          ? tracking_stats.yaw_error_sum / tracking_stats.samples
          : std::numeric_limits<double>::quiet_NaN();

  std::lock_guard<std::mutex> lock(log_mutex_);
  std::ofstream summary(action_summary_log_path_,
                        std::ios::out | std::ios::trunc);
  if (!summary.is_open()) {
    RCLCPP_WARN(this->get_logger(), "Failed to open retreat summary log: %s",
                action_summary_log_path_.c_str());
    return;
  }

  summary << std::fixed << std::setprecision(6);
  summary << "metric,value\n";
  summary << "status," << status << '\n';
  summary << "success," << success << '\n';
  summary << "elapsed_sec," << elapsed_sec << '\n';
  summary << "trail_pose_count," << trail_pose_count << '\n';
  summary << "trail_length_m," << trail_length << '\n';
  summary << "configured_max_retreat_distance_m," << max_retreat_distance_
          << '\n';
  summary << "allowed_distance_m," << allowed_distance << '\n';
  summary << "travelled_distance_m," << travelled_distance << '\n';
  summary << "reached_allowed_distance," << reached_allowed_distance << '\n';
  summary << "exceeded_allowed_distance," << exceeded_allowed_distance << '\n';
  summary << "allowed_distance_overshoot_m," << allowed_distance_overshoot
          << '\n';
  summary << "exceeded_max_retreat_distance," << exceeded_max_retreat_distance
          << '\n';
  summary << "max_retreat_distance_overshoot_m,"
          << max_retreat_distance_overshoot << '\n';
  summary << "final_dist_end_m," << final_dist_end << '\n';
  summary << "tracking_samples," << tracking_stats.samples << '\n';
  summary << "mean_cross_track_error_m," << mean_cross_track_error << '\n';
  summary << "max_cross_track_error_m,"
          << tracking_stats.max_cross_track_error << '\n';
  summary << "mean_abs_yaw_error_rad," << mean_yaw_error << '\n';
  summary << "max_abs_yaw_error_rad," << tracking_stats.max_yaw_error << '\n';
  summary.flush();

  RCLCPP_INFO(this->get_logger(),
              "Retreat summary: status=%s travelled=%.3fm allowed=%.3fm "
              "allowed_overshoot=%.3fm exceeded_max=%d "
              "max_cross_track=%.3fm log=%s",
              status.c_str(), travelled_distance, allowed_distance,
              allowed_distance_overshoot, exceeded_max_retreat_distance,
              tracking_stats.max_cross_track_error,
              action_summary_log_path_.c_str());
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

  const auto action_start = std::chrono::steady_clock::now();
  openActionLog();

  std::size_t trail_pose_count = 0;
  double total_length = std::numeric_limits<double>::quiet_NaN();
  double allowed_distance = std::numeric_limits<double>::quiet_NaN();
  double travelled_distance = 0.0;
  double final_dist_end = std::numeric_limits<double>::quiet_NaN();
  TrackingStats tracking_stats;

  auto finish = [this]() {
    publishStop();
    closeActionLog();
    active_ = false;
  };

  auto finalize = [this, &finish, &action_start, &trail_pose_count,
                   &total_length, &allowed_distance, &travelled_distance,
                   &final_dist_end, &tracking_stats](const std::string &status,
                                                    bool success) {
    const double elapsed_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      action_start)
            .count();
    writeActionSummary(status, success, elapsed_sec, trail_pose_count,
                       total_length, allowed_distance, travelled_distance,
                       final_dist_end, tracking_stats);
    finish();
  };

  // 1) trail 스냅샷 + 역순 변환
  std::vector<geometry_msgs::msg::PoseStamped> rev;
  {
    std::lock_guard<std::mutex> lock(trail_mutex_);
    if (!has_trail_ || latest_trail_.poses.empty()) {
      RCLCPP_WARN(this->get_logger(), "No trail available for retreat");
      result->success = false;
      result->success_message = "No trail available";
      finalize("no_trail", false);
      goal_handle->abort(result);
      return;
    }
    rev.assign(latest_trail_.poses.rbegin(), latest_trail_.poses.rend());
  }
  trail_pose_count = rev.size();

  // 2) 누적 arc-length 계산 (rev[0]에서 rev[i]까지)
  std::vector<double> cum(rev.size(), 0.0);
  for (std::size_t i = 1; i < rev.size(); ++i) {
    const double dx = rev[i].pose.position.x - rev[i - 1].pose.position.x;
    const double dy = rev[i].pose.position.y - rev[i - 1].pose.position.y;
    cum[i] = cum[i - 1] + std::sqrt(dx * dx + dy * dy);
  }
  writeTrailSnapshot(rev, cum);
  total_length = cum.back();
  allowed_distance =
      std::min(total_length, static_cast<double>(max_retreat_distance_));

  if (!std::isfinite(total_length)) {
    RCLCPP_ERROR(this->get_logger(), "Invalid trail length. Retreat aborted.");
    finalize("invalid_trail_length", false);
    result->success = false;
    result->success_message = "Invalid trail length";
    goal_handle->abort(result);
    return;
  }

  if (rev.size() < 2 || total_length <= 1e-6 ||
      total_length < min_trail_length_) {
    RCLCPP_INFO(this->get_logger(),
                "Trail too short (%.3fm < %.3fm). Nothing to retreat.",
                total_length, min_trail_length_);
    finalize("trail_too_short", true);
    result->success = true;
    result->success_message = "Trail too short, no retreat needed";
    goal_handle->succeed(result);
    return;
  }

  if (!std::isfinite(retreat_speed_) || retreat_speed_ <= 0.0F) {
    RCLCPP_ERROR(this->get_logger(),
                 "Invalid retreat_speed %.3f. Retreat aborted.",
                 retreat_speed_);
    finalize("invalid_retreat_speed", false);
    result->success = false;
    result->success_message = "Invalid retreat speed";
    goal_handle->abort(result);
    return;
  }

  if (!std::isfinite(max_retreat_distance_) || max_retreat_distance_ <= 0.0F) {
    RCLCPP_ERROR(this->get_logger(),
                 "Invalid max_retreat_distance %.3f. Retreat aborted.",
                 max_retreat_distance_);
    finalize("invalid_max_retreat_distance", false);
    result->success = false;
    result->success_message = "Invalid maximum retreat distance";
    goal_handle->abort(result);
    return;
  }

  if (!std::isfinite(control_rate_hz_) || control_rate_hz_ <= 0.0F) {
    RCLCPP_ERROR(this->get_logger(),
                 "Invalid control_rate_hz %.3f. Retreat aborted.",
                 control_rate_hz_);
    finalize("invalid_control_rate", false);
    result->success = false;
    result->success_message = "Invalid control rate";
    goal_handle->abort(result);
    return;
  }

  // 3) 제어 루프
  rclcpp::Rate loop_rate(control_rate_hz_);
  std::size_t closest_idx = 0;
  const double v = -static_cast<double>(retreat_speed_);
  const double timeout_sec =
      allowed_distance / static_cast<double>(retreat_speed_) +
      std::max(0.0, static_cast<double>(timeout_margin_sec_));
  const auto control_start = std::chrono::steady_clock::now();
  const auto &before_end = rev[rev.size() - 2].pose.position;
  const auto &end = rev.back().pose.position;
  const double end_segment_x = end.x - before_end.x;
  const double end_segment_y = end.y - before_end.y;
  double previous_rx = 0.0;
  double previous_ry = 0.0;
  bool previous_pose_available = false;

  auto write_trace_sample =
      [this, &rev, &allowed_distance](
          double elapsed_sec, double rx, double ry, double ryaw,
          std::size_t closest_idx, double trail_progress, double remaining,
          double travelled, double cross_track_error, double yaw_error,
          double dist_end, double lookahead, double target_x, double target_y,
          double v_cmd, double w_cmd, bool terminal_mode,
          const std::string &event) {
        std::ostringstream line;
        line << std::fixed << std::setprecision(6) << elapsed_sec << "," << rx
             << "," << ry << "," << ryaw << "," << closest_idx << ","
             << rev.size() << "," << trail_progress << "," << remaining << ","
             << travelled << "," << allowed_distance << ","
             << (travelled >= allowed_distance) << ","
             << (travelled > allowed_distance + 1e-6) << ","
             << (travelled >
                 static_cast<double>(max_retreat_distance_) + 1e-6)
             << ","
             << cross_track_error << "," << yaw_error << "," << dist_end << ","
             << lookahead << "," << target_x << "," << target_y << "," << v_cmd
             << "," << w_cmd << "," << terminal_mode << "," << event;
        writeActionTrace(line.str());
      };

  RCLCPP_INFO(this->get_logger(),
              "Retreat started: trail=%.3fm timeout=%.2fs",
              total_length, timeout_sec);

  while (rclcpp::ok()) {
    if (goal_handle->is_canceling()) {
      finalize("canceled", false);
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
      finalize("timeout", false);
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

    double cross_track_error = std::sqrt(best_d2);
    if (closest_idx > 0) {
      cross_track_error =
          std::min(cross_track_error,
                   pointToSegmentDistance(
                       rx, ry, rev[closest_idx - 1].pose.position,
                       rev[closest_idx].pose.position));
    }
    if (closest_idx + 1 < rev.size()) {
      cross_track_error =
          std::min(cross_track_error,
                   pointToSegmentDistance(
                       rx, ry, rev[closest_idx].pose.position,
                       rev[closest_idx + 1].pose.position));
    }

    const double remaining = total_length - cum[closest_idx];
    const double trail_progress = total_length - remaining;

    // 종료 판정: 끝점과 충분히 가까우면 success.
    const double dx_end = rev.back().pose.position.x - rx;
    const double dy_end = rev.back().pose.position.y - ry;
    const double dist_end = std::sqrt(dx_end * dx_end + dy_end * dy_end);
    final_dist_end = dist_end;

    const double trail_yaw =
        tf2::getYaw(rev[closest_idx].pose.orientation);
    const double yaw_error = normalizeAngle(ryaw - trail_yaw);
    tracking_stats.samples += 1;
    tracking_stats.cross_track_error_sum += cross_track_error;
    tracking_stats.max_cross_track_error =
        std::max(tracking_stats.max_cross_track_error, cross_track_error);
    tracking_stats.yaw_error_sum += std::abs(yaw_error);
    tracking_stats.max_yaw_error =
        std::max(tracking_stats.max_yaw_error, std::abs(yaw_error));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (travelled_distance >= allowed_distance) {
      write_trace_sample(elapsed_sec, rx, ry, ryaw, closest_idx,
                         trail_progress, remaining, travelled_distance,
                         cross_track_error, yaw_error, dist_end, nan, nan, nan,
                         0.0, 0.0, false, "distance_limit_reached");
      finalize("distance_limit_reached", true);
      result->success = true;
      result->success_message = "Retreat reached allowed distance";
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(),
                  "Retreat reached allowed distance: %.3fm >= %.3fm",
                  travelled_distance, allowed_distance);
      return;
    }

    if (dist_end < terminal_threshold_) {
      write_trace_sample(elapsed_sec, rx, ry, ryaw, closest_idx,
                         trail_progress, remaining, travelled_distance,
                         cross_track_error, yaw_error, dist_end, nan, nan, nan,
                         0.0, 0.0, true, "trail_start_reached");
      finalize("trail_start_reached", true);
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
      write_trace_sample(elapsed_sec, rx, ry, ryaw, closest_idx,
                         trail_progress, remaining, travelled_distance,
                         cross_track_error, yaw_error, dist_end, nan, nan, nan,
                         0.0, 0.0, true, "trail_start_passed");
      finalize("trail_start_passed", true);
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

    write_trace_sample(elapsed_sec, rx, ry, ryaw, closest_idx, trail_progress,
                       remaining, travelled_distance, cross_track_error,
                       yaw_error, dist_end, Ld, target.x, target.y, v_cmd, w_cmd,
                       terminal_mode, "control");

    RCLCPP_DEBUG(this->get_logger(),
                 "retreat: idx=%zu/%zu rem=%.3fm Ld=%.3fm v=%.3f w=%.3f",
                 closest_idx, rev.size(), remaining, Ld, v_cmd, w_cmd);

    loop_rate.sleep();
  }

  finalize("node_shutdown", false);
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
