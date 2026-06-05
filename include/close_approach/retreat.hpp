#pragma once

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "inha_interfaces/action/retreat.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class RetreatNode : public rclcpp::Node {
public:
  RetreatNode();

private:
  using RetreatAction = inha_interfaces::action::Retreat;
  using GoalHandleRetreat = rclcpp_action::ServerGoalHandle<RetreatAction>;

  rclcpp_action::Server<RetreatAction>::SharedPtr action_server_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr trail_subscriber_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex trail_mutex_;
  nav_msgs::msg::Path latest_trail_;
  bool has_trail_ = false;

  std::string odom_frame_;
  std::string robot_frame_;
  float retreat_speed_ = 0.07F;
  float lookahead_max_ = 0.15F;
  float lookahead_min_ = 0.05F;
  float terminal_threshold_ = 0.05F; // 끝점 도달 판정 거리(m)
  float min_trail_length_ = 0.05F;   // 이보다 짧으면 retreat 의미 없음
  float max_retreat_distance_ = 0.30F; // 안전상 허용할 최대 누적 이동 거리(m)
  float w_max_ = 0.8F;               // 각속도 클램프
  float control_rate_hz_ = 10.0F;
  float timeout_margin_sec_ = 3.0F;

  bool log_enabled_ = true;
  std::filesystem::path log_dir_;
  std::filesystem::path action_log_dir_;
  std::filesystem::path action_trace_log_path_;
  std::filesystem::path action_summary_log_path_;
  std::ofstream action_trace_log_file_;
  std::mutex log_mutex_;

  struct TrackingStats {
    std::size_t samples = 0;
    double cross_track_error_sum = 0.0;
    double max_cross_track_error = 0.0;
    double yaw_error_sum = 0.0;
    double max_yaw_error = 0.0;
  };

  std::atomic<bool> active_{false};

  void trailCallback(const nav_msgs::msg::Path::SharedPtr msg);

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const RetreatAction::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleRetreat> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandleRetreat> goal_handle);
  void execute(const std::shared_ptr<GoalHandleRetreat> goal_handle);

  bool getRobotPoseInOdom(double &x, double &y, double &yaw);
  void publishStop();
  void openActionLog();
  void closeActionLog();
  void writeActionTrace(const std::string &line);
  void writeTrailSnapshot(
      const std::vector<geometry_msgs::msg::PoseStamped> &trail,
      const std::vector<double> &cumulative_distance);
  void writeActionSummary(const std::string &status, bool success,
                          double elapsed_sec, std::size_t trail_pose_count,
                          double trail_length, double allowed_distance,
                          double travelled_distance, double final_dist_end,
                          const TrackingStats &tracking_stats);
};
