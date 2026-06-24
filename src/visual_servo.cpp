#include "close_approach/visual_servo.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

VisualServoNode::VisualServoNode() : Node("visual_servo") {
  // ── Params ─────────────────────────────────────────────────────────────
  this->declare_parameter<std::string>("bbox_topic", "/detection/bbox_stream");
  this->declare_parameter<std::string>("active_topic", "/visual_servo/active");
  this->declare_parameter<std::string>("class_id_a", "0");
  this->declare_parameter<std::string>("class_id_b", "1");
  this->declare_parameter<float>("image_width",  640.0f);
  this->declare_parameter<float>("image_height", 480.0f);
  this->declare_parameter<float>("image_center_x", -1.0f);  // <0 이면 width/2
  this->declare_parameter<float>("image_center_y", -1.0f);  // <0 이면 height/2
  this->declare_parameter<bool>("require_both", true);
  this->declare_parameter<float>("kp_yaw", 0.8f);
  this->declare_parameter<float>("ki_yaw", 0.0f);
  this->declare_parameter<float>("kd_yaw", 0.2f);
  this->declare_parameter<bool>("invert_yaw", true);
  this->declare_parameter<float>("tol_norm", 0.03f);
  this->declare_parameter<float>("forward_speed", 0.10f);
  this->declare_parameter<float>("align_gate", 0.30f);
  this->declare_parameter<float>("max_v", 0.20f);
  this->declare_parameter<float>("max_w", 0.5f);
  this->declare_parameter<float>("lin_accel_limit", 0.5f);
  this->declare_parameter<float>("lin_decel_limit", 1.0f);
  this->declare_parameter<float>("ang_accel_limit", 2.0f);
  this->declare_parameter<float>("detection_timeout_sec", 0.3f);
  this->declare_parameter<float>("control_rate_hz", 20.0f);
  this->declare_parameter<bool>("start_enabled", true);
  this->declare_parameter<bool>("debug_log", true);

  this->get_parameter("bbox_topic", bbox_topic_);
  this->get_parameter("active_topic", active_topic_);
  this->get_parameter("class_id_a", class_id_a_);
  this->get_parameter("class_id_b", class_id_b_);
  this->get_parameter("image_width",  image_width_);
  this->get_parameter("image_height", image_height_);
  float ov_cx = -1.0f, ov_cy = -1.0f;
  this->get_parameter("image_center_x", ov_cx);
  this->get_parameter("image_center_y", ov_cy);
  center_x_ = (ov_cx >= 0.0f) ? ov_cx : image_width_  * 0.5f;
  center_y_ = (ov_cy >= 0.0f) ? ov_cy : image_height_ * 0.5f;
  this->get_parameter("require_both", require_both_);
  this->get_parameter("kp_yaw", kp_yaw_);
  this->get_parameter("ki_yaw", ki_yaw_);
  this->get_parameter("kd_yaw", kd_yaw_);
  this->get_parameter("invert_yaw", invert_yaw_);
  this->get_parameter("tol_norm", tol_norm_);
  this->get_parameter("forward_speed", forward_speed_);
  this->get_parameter("align_gate", align_gate_);
  this->get_parameter("max_v", max_v_);
  this->get_parameter("max_w", max_w_);
  this->get_parameter("lin_accel_limit", lin_accel_limit_);
  this->get_parameter("lin_decel_limit", lin_decel_limit_);
  this->get_parameter("ang_accel_limit", ang_accel_limit_);
  this->get_parameter("detection_timeout_sec", detection_timeout_sec_);
  this->get_parameter("control_rate_hz", control_rate_hz_);
  this->get_parameter("start_enabled", start_enabled_);
  this->get_parameter("debug_log", debug_log_);

  enabled_.store(start_enabled_, std::memory_order_release);
  control_rate_hz_ = std::max(1.0f, control_rate_hz_);
  align_gate_ = std::max(1e-3f, align_gate_);

  // ── ROS I/O ────────────────────────────────────────────────────────────
  auto qos_be  = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
  auto qos_rel = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

  detection_sub_ = this->create_subscription<Detection2DArray>(
      bbox_topic_, qos_be,
      std::bind(&VisualServoNode::detectionCallback, this,
                std::placeholders::_1));
  active_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      active_topic_, qos_rel,
      std::bind(&VisualServoNode::activeCallback, this, std::placeholders::_1));
  cmd_vel_pub_ = this->create_publisher<Twist>("/cmd_vel", qos_rel);

  const auto period = std::chrono::duration<double>(1.0 / control_rate_hz_);
  control_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&VisualServoNode::controlTimerCallback, this));

  RCLCPP_INFO(this->get_logger(),
              "VisualServoNode ready. bbox=%s classes=[%s,%s] center=(%.1f,%.1f) "
              "fwd=%.2f start_enabled=%d",
              bbox_topic_.c_str(), class_id_a_.c_str(), class_id_b_.c_str(),
              center_x_, center_y_, forward_speed_, start_enabled_);
}

void VisualServoNode::activeCallback(const std_msgs::msg::Bool::SharedPtr msg) {
  const bool was = enabled_.exchange(msg->data, std::memory_order_acq_rel);
  if (was != msg->data) {
    RCLCPP_INFO(this->get_logger(), "VisualServo %s",
                msg->data ? "enabled" : "disabled");
  }
  if (!msg->data) {
    // 비활성화 시 즉시 정지 + 상태 리셋(다음 활성화로 히스토리 누수 방지).
    std::lock_guard<std::mutex> lk(target_mutex_);
    target_.valid = false;
    err_prev_valid_ = false;
    err_sum_ = 0.0f;
    last_vx_ = last_wz_ = 0.0f;
    cmd_vel_pub_->publish(Twist{});
  }
}

void VisualServoNode::detectionCallback(
    const Detection2DArray::ConstSharedPtr &msg) {
  if (!enabled_.load(std::memory_order_acquire)) {
    return;
  }

  // 클래스별로 이미지 중심에 가장 가까운 bbox 중심 하나만 선택.
  bool found_a = false, found_b = false;
  float best_da = std::numeric_limits<float>::max();
  float best_db = std::numeric_limits<float>::max();
  float ax = 0.0f, bx = 0.0f;  // 선택된 중심의 x (횡오차에만 x 사용)

  for (const auto &det : msg->detections) {
    if (det.results.empty()) {
      continue;
    }
    const std::string &cls = det.results.front().hypothesis.class_id;
    const bool is_a = (cls == class_id_a_);
    const bool is_b = (cls == class_id_b_);
    if (!is_a && !is_b) {
      continue;
    }

    const float cx = static_cast<float>(det.bbox.center.position.x);
    const float cy = static_cast<float>(det.bbox.center.position.y);
    const float dx = cx - center_x_;
    const float dy = cy - center_y_;
    const float dist = dx * dx + dy * dy;  // 이미지 중심까지 거리(제곱)

    if (is_a && dist < best_da) {
      best_da = dist;
      ax = cx;
      found_a = true;
    }
    if (is_b && dist < best_db) {
      best_db = dist;
      bx = cx;
      found_b = true;
    }
  }

  float mid_x = 0.0f;
  bool valid = false;
  if (found_a && found_b) {
    mid_x = 0.5f * (ax + bx);
    valid = true;
  } else if (!require_both_ && (found_a || found_b)) {
    mid_x = found_a ? ax : bx;
    valid = true;
  }

  std::lock_guard<std::mutex> lk(target_mutex_);
  target_.valid = valid;
  // staleness 기준은 노드 클럭 수신 시각으로 통일(메시지 stamp의 클럭 소스가
  // now()와 달라 빼기에서 죽는 문제 방지).
  target_.stamp = this->now();
  if (valid) {
    const float half_w = std::max(1.0f, center_x_);
    target_.err_norm = std::clamp((mid_x - center_x_) / half_w, -1.0f, 1.0f);
  }
}

float VisualServoNode::slew(float current, float target, float dt,
                            float accel_limit, float decel_limit) {
  if (dt <= 0.0f) return target;
  const bool speeding_up = std::abs(target) > std::abs(current);
  const float step = (speeding_up ? accel_limit : decel_limit) * dt;
  const float delta = target - current;
  if (std::abs(delta) <= step) return target;
  return current + std::copysign(step, delta);
}

void VisualServoNode::controlTimerCallback() {
  const float dt = 1.0f / control_rate_hz_;

  bool  valid = false;
  float err = 0.0f;
  {
    std::lock_guard<std::mutex> lk(target_mutex_);
    // target_.valid 일 때만 시간 차를 계산(기본 생성된 stamp는 클럭 소스가
    // 달라 빼면 throw 하므로 첫 검출 전에는 접근하지 않는다).
    if (enabled_.load(std::memory_order_acquire) && target_.valid) {
      const double age = (this->now() - target_.stamp).seconds();
      valid = age <= static_cast<double>(detection_timeout_sec_);
      err = target_.err_norm;
    }
  }

  float target_vx = 0.0f;
  float target_wz = 0.0f;

  if (valid) {
    // ── 횡오차 → yaw PD ──────────────────────────────────────────────────
    float ctrl_err = (std::abs(err) < tol_norm_) ? 0.0f : err;
    const float d_err =
        err_prev_valid_ ? (ctrl_err - err_prev_) / dt : 0.0f;
    err_prev_ = ctrl_err;
    err_prev_valid_ = true;

    // 적분(기본 ki=0). ki>0일 때만 누적·클램프해 windup 방지.
    if (ki_yaw_ > 0.0f) {
      err_sum_ = std::clamp(err_sum_ + ctrl_err * dt, -1.0f / ki_yaw_,
                            1.0f / ki_yaw_);
    } else {
      err_sum_ = 0.0f;
    }

    float wz = kp_yaw_ * ctrl_err + ki_yaw_ * err_sum_ + kd_yaw_ * d_err;
    if (invert_yaw_) wz = -wz;  // 이미지 +x(우측) → 우회전(angular.z<0)
    target_wz = std::clamp(wz, -max_w_, max_w_);

    // ── 종방향: 정렬될수록 전진(전진 전용). |err|>=gate 면 회전 우선. ──────
    const float align = std::clamp(1.0f - std::abs(err) / align_gate_, 0.0f, 1.0f);
    target_vx = std::clamp(forward_speed_ * align, 0.0f, max_v_);
  } else {
    err_prev_valid_ = false;  // 타깃 끊김: 미분/적분 히스토리 리셋
    err_sum_ = 0.0f;
  }

  // 이산적 검출/valid 토글이 cmd_vel 급변으로 이어지지 않도록 슬루 제한.
  last_vx_ = slew(last_vx_, target_vx, dt, lin_accel_limit_, lin_decel_limit_);
  last_wz_ = slew(last_wz_, target_wz, dt, ang_accel_limit_, ang_accel_limit_);

  Twist out;
  out.linear.x  = last_vx_;
  out.angular.z = last_wz_;
  cmd_vel_pub_->publish(out);

  if (debug_log_) {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 300,
        "valid=%d err=%.3f -> target(vx=%.3f wz=%.3f) cmd(vx=%.3f wz=%.3f)",
        valid, err, target_vx, target_wz, last_vx_, last_wz_);
  }
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VisualServoNode>());
  rclcpp::shutdown();
  return 0;
}
