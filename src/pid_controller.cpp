#include "close_approach/pid_controller.hpp"
#include <cmath>

void PIDController::setParameters(float kp_x, float kp_y, float kp_theta,
                                  float ki_x, float ki_y, float ki_theta,
                                  float kd_x, float kd_y, float kd_theta,
                                  float base_to_rotationcore) {
  this->kp_x = kp_x;
  this->kp_y = kp_y;
  this->kp_theta = kp_theta;
  this->ki_x = ki_x;
  this->ki_y = ki_y;
  this->ki_theta = ki_theta;
  this->kd_x = kd_x;
  this->kd_y = kd_y;
  this->kd_theta = kd_theta;
  this->base_to_rotationcore = base_to_rotationcore;
}

geometry_msgs::msg::Twist
PIDController::compute_control(const SE2Error &se2_error, float dt) {
  geometry_msgs::msg::Twist cmd_vel;
  if (dt <= 0.0f)
    return cmd_vel; // 방어 코드

  // 1. 센서 기준 에러를 회전 중심(CoR) 기준으로 변환
  float e_x = se2_error.x + base_to_rotationcore;
  float e_y = se2_error.y;

  float e_theta = se2_error.degree_theta;

  // 2. PID 연산
  se2_error_sum.x += e_x * dt;
  se2_error_sum.y += e_y * dt;
  se2_error_sum.degree_theta += e_theta * dt;

  float d_x = (e_x - se2_error_prev.x) / dt;
  float d_y = (e_y - se2_error_prev.y) / dt;
  float d_theta = (e_theta - se2_error_prev.degree_theta) / dt;

  se2_error_prev = {e_x, e_y, e_theta};

  float control_x = (kp_x * e_x) + (ki_x * se2_error_sum.x) + (kd_x * d_x);
  float control_y = (kp_y * e_y) + (ki_y * se2_error_sum.y) + (kd_y * d_y);
  float control_theta = (kp_theta * e_theta) +
                        (ki_theta * se2_error_sum.degree_theta) +
                        (kd_theta * d_theta);

  // 3. 기구학 모델(Lyapunov 기반) 방정식 적용
  // 종방향 속도 (단순 거리 오차 제어)
  float v_x = control_x;

  // 회전 속도 (각조향 복구 + 횡방향 복구)
  // 선속도 v_x가 존재할 때 횡방향 오차 제어가 활성화되어 선에 부드럽게 진입하게
  // 함
  float w_z = control_theta + control_y;

  // Limits
  float max_v = 0.05f; // 로봇의 최대 직진 속도 (m/s)
  float max_w = 0.2f;  // 로봇의 최대 회전 속도 (rad/s)
  // v_x 포화가 w_z 조향 능력을 깎지 않도록 각 축을 독립적으로 제한한다.
  v_x = std::clamp(v_x, -max_v, max_v);
  w_z = std::clamp(w_z, -max_w, max_w);
  cmd_vel.linear.x = v_x;
  cmd_vel.angular.z = w_z;

  return cmd_vel;
}
