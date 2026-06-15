#pragma once

#include "close_approach/msg/approach_error.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/bool.hpp>

class EdgeDetector : public rclcpp::Node {
public:
  EdgeDetector();

private:
  using ApproachError = close_approach::msg::ApproachError;

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr img_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr               active_sub_;
  rclcpp::Publisher<ApproachError>::SharedPtr                        error_pub_;

  std::string img_topic_;

  // Edge detection params (loaded from YAML, matches edge_yaw_viewer saved format)
  int canny_low_    = 50;
  int canny_high_   = 150;
  int hough_thresh_ = 40;
  int min_length_   = 60;
  int max_gap_      = 15;
  int roi_top_pct_  = 20;
  int roi_bot_pct_  = 80;
  int roi_left_pct_ = 20;
  int roi_right_pct_= 80;

  // Last valid theta (rad) — held when no horizontal edges found
  float last_theta_rad_    = 0.0f;
  bool  theta_initialized_ = false;

  void imgCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg);
  void activeCallback(const std_msgs::msg::Bool::SharedPtr msg);
};
