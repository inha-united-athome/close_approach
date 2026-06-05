#pragma once

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "inha_interfaces/action/approach.hpp"
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class DeadReckoningApproachNode : public rclcpp::Node {
public:
  DeadReckoningApproachNode();

private:
  using ApproachAction = inha_interfaces::action::Approach;
  using GoalHandleApproach = rclcpp_action::ServerGoalHandle<ApproachAction>;

  rclcpp_action::Server<ApproachAction>::SharedPtr action_server_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string action_name_;
  std::string odom_frame_;
  std::string robot_frame_;
  std::string cmd_vel_topic_;

  float max_goal_distance_ = 0.20F;
  float min_goal_distance_ = 0.01F;
  float goal_tolerance_ = 0.01F;
  float max_v_ = 0.04F;
  float min_v_ = 0.01F;
  float max_w_ = 0.20F;
  float accel_limit_ = 0.05F;
  float decel_limit_ = 0.08F;
  float kp_distance_ = 0.8F;
  float kp_lateral_ = 1.0F;
  float kp_yaw_ = 1.2F;
  float slow_down_distance_ = 0.08F;
  float control_rate_hz_ = 20.0F;
  float timeout_margin_sec_ = 3.0F;

  bool log_enabled_ = true;
  std::filesystem::path log_dir_;
  std::filesystem::path action_log_path_;
  std::ofstream action_log_file_;
  std::mutex log_mutex_;

  std::atomic<bool> active_{false};

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const ApproachAction::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleApproach> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandleApproach> goal_handle);
  void execute(const std::shared_ptr<GoalHandleApproach> goal_handle);

  bool getRobotPoseInOdom(double &x, double &y, double &yaw);
  void publishStop();

  void openActionLog(float requested_distance, float clamped_distance);
  void closeActionLog();
  void writeActionLog(const std::string &line);
};
