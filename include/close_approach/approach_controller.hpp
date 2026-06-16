#pragma once

#include "close_approach/msg/approach_error.hpp"
#include "close_approach/pid_controller.hpp"

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>

class ApproachController : public rclcpp::Node {
public:
  ApproachController();

private:
  using ApproachError = close_approach::msg::ApproachError;

  rclcpp::Subscription<ApproachError>::SharedPtr       error_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr active_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;

  std::shared_ptr<PIDController> pid_;

  // Params
  float kp_x_, ki_x_, kd_x_;
  float kp_theta_, ki_theta_, kd_theta_;
  float max_v_, max_w_;
  float decel_dist_max_, decel_dist_min_, decel_ratio_;
  bool debug_log_ = true;

  // Runtime state
  float initial_dist_    = 0.0f;
  bool  initial_dist_set_= false;
  rclcpp::Time prev_time_;
  bool         prev_time_valid_ = false;

  void errorCallback(const ApproachError::ConstSharedPtr &msg);
  void activeCallback(const std_msgs::msg::Bool::SharedPtr msg);
};
