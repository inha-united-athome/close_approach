#pragma once

#include <atomic>
#include <mutex>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

/*
VisualServoNode
- /detection/bbox_stream (vision_msgs/Detection2DArray) 구독.
- 두 타깃 클래스 각각에 대해 이미지 중심에 가장 가까운 bbox 하나만 선택.
- 두 선택 bbox 중심의 중점(midpoint)을 이미지 중심에 맞추는 횡방향 비주얼 서보잉.
  디퍼런셜 휠이므로 횡오차는 yaw(angular.z)로만 보정.
- 종방향은 "필요한 만큼만" 앞으로 전진: 정렬이 잘 될수록 빠르게, 횡오차가
  크면 먼저 회전(전진 감속). 후진은 하지 않는다(전진 전용).
- 검출이 끊기면(timeout) 명령을 0으로 램프해 정지.
*/
class VisualServoNode : public rclcpp::Node {
public:
  VisualServoNode();

private:
  using Detection2DArray = vision_msgs::msg::Detection2DArray;
  using Twist = geometry_msgs::msg::Twist;

  // 가장 최근 검출에서 산출한 횡오차(정규화) 캐시. 검출 콜백이 채우고,
  // 고정 주기 제어 타이머가 읽어 cmd_vel 을 만든다.
  struct TargetState {
    bool valid = false;          // 두 클래스(또는 require_both=false 시 1개) 확보
    float err_norm = 0.0f;       // (midpoint_x - center_x) / (width/2), [-1,1]
    rclcpp::Time stamp;          // 검출 헤더 stamp
  };

  void detectionCallback(const Detection2DArray::ConstSharedPtr &msg);
  void activeCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void controlTimerCallback();

  // 한 축을 target 으로 가속/감속 한계 내에서 이동.
  static float slew(float current, float target, float dt, float accel_limit,
                    float decel_limit);

  // Subscribers / Publishers / Timer
  rclcpp::Subscription<Detection2DArray>::SharedPtr detection_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr active_sub_;
  rclcpp::Publisher<Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  // Params
  std::string bbox_topic_;
  std::string active_topic_;
  std::string class_id_a_, class_id_b_;
  float image_width_  = 640.0f;
  float image_height_ = 480.0f;
  float center_x_ = 320.0f;  // image_width_/2 (또는 override)
  float center_y_ = 240.0f;  // image_height_/2
  bool  require_both_ = true;

  float kp_yaw_ = 0.8f, ki_yaw_ = 0.0f, kd_yaw_ = 0.2f;
  bool  invert_yaw_ = true;     // angular.z = -gain*err (이미지 +x 우측 → 우회전)
  float tol_norm_ = 0.03f;      // 횡오차 데드밴드(정규화)

  float forward_speed_ = 0.10f; // 기본 전진 속도(m/s)
  float align_gate_ = 0.30f;    // |err|>=gate 면 전진 0(회전 우선)
  float max_v_ = 0.20f, max_w_ = 0.5f;

  float lin_accel_limit_ = 0.5f, lin_decel_limit_ = 1.0f, ang_accel_limit_ = 2.0f;
  float detection_timeout_sec_ = 0.3f;
  float control_rate_hz_ = 20.0f;
  bool  start_enabled_ = true;
  bool  debug_log_ = true;

  // Runtime
  std::atomic<bool> enabled_{true};
  std::mutex        target_mutex_;
  TargetState       target_;

  float err_prev_ = 0.0f;
  float err_sum_ = 0.0f;        // 적분항(기본 ki=0 → PD). windup 클램프 적용.
  bool  err_prev_valid_ = false;
  float last_vx_ = 0.0f, last_wz_ = 0.0f;
};
