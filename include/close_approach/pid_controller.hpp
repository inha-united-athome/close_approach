#pragma once

#include "close_approach/error_estimator.hpp"
#include <Eigen/Dense>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

class PIDController {
public:
  PIDController() = default;

  // 횡방향(y) 에러 제어 게인 추가
  void setParameters(float kp_x, float kp_y, float kp_theta, float ki_x,
                     float ki_y, float ki_theta, float kd_x, float kd_y,
                     float kd_theta, float base_to_rotationcore);

  // dt와 R|t 행렬(기본값은 단위행렬)을 받아 최종 제어값 Twist 반환
  geometry_msgs::msg::Twist compute_control(const SE2Error &se2_error,
                                            float dt);

private:
  float kp_x{0.25f}, kp_y{2.0f}, kp_theta{1.2f};
  float ki_x{0.0f}, ki_y{0.0f}, ki_theta{0.0f};
  float kd_x{0.0f}, kd_y{0.0f}, kd_theta{0.0f};
  float base_to_rotationcore{0.2f};

  SE2Error se2_error_prev{0.0f, 0.0f, 0.0f};
  SE2Error se2_error_sum{0.0f, 0.0f, 0.0f};

  Eigen::Matrix4f base_to_cor_rt_ = Eigen::Matrix4f::Identity();
};
