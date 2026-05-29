#pragma once

#include "close_approach/error_estimator.hpp"
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

class PIDController {
public:
  PIDController() = default;

  // 횡방향(y) 에러 제어 게인 추가
  void setParameters(float kp_x, float kp_y, float kp_theta, float ki_x,
                     float ki_y, float ki_theta, float kd_x, float kd_y,
                     float kd_theta);

  // 속도/회전 한계 (감속 cap의 base가 되는 max_v)
  void setLimits(float max_v, float max_w);

  // 적분/미분 히스토리 초기화 (상태 전이 시 wind-up 방지)
  void reset();

  // v_scale ∈ [0,1]: 감속 ramp용. 1이면 max_v 그대로 사용.
  geometry_msgs::msg::Twist compute_control(const SE2Error &se2_error,
                                            float dt,
                                            float v_scale = 1.0f);

  // ALIGN_THETA 전용: v_x=0, w_z만 kp_theta * e_theta 로 제어 (P-only)
  geometry_msgs::msg::Twist compute_align_only(float theta_err) const;

private:
  float kp_x{0.25f}, kp_y{2.0f}, kp_theta{1.2f};
  float ki_x{0.0f}, ki_y{0.0f}, ki_theta{0.0f};
  float kd_x{0.0f}, kd_y{0.0f}, kd_theta{0.0f};
  float max_v_{0.05f};
  float max_w_{0.2f};

  SE2Error se2_error_prev{0.0f, 0.0f, 0.0f};
  SE2Error se2_error_sum{0.0f, 0.0f, 0.0f};
};
