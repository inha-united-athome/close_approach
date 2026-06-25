#include "close_approach/approach_controller.hpp"
#include "close_approach/error_estimator.hpp"

#include <cmath>

ApproachController::ApproachController() : Node("approach_controller") {
  // Params
  this->declare_parameter<float>("kp_x",     0.25F);
  this->declare_parameter<float>("ki_x",     0.0F);
  this->declare_parameter<float>("kd_x",     0.0F);
  this->declare_parameter<float>("kp_theta", 1.2F);
  this->declare_parameter<float>("ki_theta", 0.0F);
  this->declare_parameter<float>("kd_theta", 0.0F);
  this->declare_parameter<float>("max_v",           0.10F);
  this->declare_parameter<float>("max_w",           0.2F);
  this->declare_parameter<float>("decel_dist_max",  0.20F);
  this->declare_parameter<float>("decel_dist_min",  0.05F);
  this->declare_parameter<float>("decel_ratio",     0.5F);
  this->declare_parameter<float>("lin_accel_limit", 0.5F);
  this->declare_parameter<float>("lin_decel_limit", 1.0F);
  this->declare_parameter<float>("ang_accel_limit", 2.0F);
  this->declare_parameter<bool>("debug_log",        true);

  this->get_parameter("kp_x",           kp_x_);
  this->get_parameter("ki_x",           ki_x_);
  this->get_parameter("kd_x",           kd_x_);
  this->get_parameter("kp_theta",       kp_theta_);
  this->get_parameter("ki_theta",       ki_theta_);
  this->get_parameter("kd_theta",       kd_theta_);
  this->get_parameter("max_v",          max_v_);
  this->get_parameter("max_w",          max_w_);
  this->get_parameter("decel_dist_max", decel_dist_max_);
  this->get_parameter("decel_dist_min", decel_dist_min_);
  this->get_parameter("decel_ratio",    decel_ratio_);
  this->get_parameter("lin_accel_limit", lin_accel_limit_);
  this->get_parameter("lin_decel_limit", lin_decel_limit_);
  this->get_parameter("ang_accel_limit", ang_accel_limit_);
  this->get_parameter("debug_log",      debug_log_);

  pid_ = std::make_shared<PIDController>();
  pid_->setParameters(kp_x_, 0.0F, kp_theta_,
                      ki_x_, 0.0F, ki_theta_,
                      kd_x_, 0.0F, kd_theta_);
  pid_->setLimits(max_v_, max_w_);

  auto qos_rel = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

  error_sub_  = this->create_subscription<ApproachError>(
      "/approach/control_error", qos_rel,
      std::bind(&ApproachController::errorCallback, this, std::placeholders::_1));
  // Latched to match the manager: receive current active state even if this
  // subscription matches after the manager already published it.
  auto active_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  active_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "/approach/active", active_qos,
      std::bind(&ApproachController::activeCallback, this, std::placeholders::_1));
  cmd_vel_pub_= this->create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel", qos_rel);

  RCLCPP_INFO(this->get_logger(), "ApproachController ready");
}

void ApproachController::activeCallback(const std_msgs::msg::Bool::SharedPtr msg) {
  // Reset both longitudinal and yaw PID state on every action boundary.
  // This prevents integral/derivative history or deceleration state from
  // leaking across success, failure, cancellation, or a new goal.
  pid_->reset();
  initial_dist_set_ = false;
  initial_dist_     = 0.0f;
  prev_time_valid_  = false;
  last_vx_          = 0.0f;
  last_wz_          = 0.0f;

  if (msg->data) {
    RCLCPP_INFO(this->get_logger(),
                "Approach active: longitudinal/yaw controller reset");
  } else {
    // Approach stopped — reset completed above, then command an explicit stop.
    geometry_msgs::msg::Twist stop;
    cmd_vel_pub_->publish(stop);
    RCLCPP_INFO(this->get_logger(),
                "Approach inactive: controller reset and stop published");
  }
}

float ApproachController::slew(float current, float target, float dt,
                               float accel_limit, float decel_limit) {
  if (dt <= 0.0f) return target;
  const bool speeding_up = std::abs(target) > std::abs(current);
  const float step = (speeding_up ? accel_limit : decel_limit) * dt;
  const float delta = target - current;
  if (std::abs(delta) <= step) return target;
  return current + std::copysign(step, delta);
}

void ApproachController::errorCallback(const ApproachError::ConstSharedPtr &msg) {
  // dt is computed for every message (valid or not) so the slew limiter keeps
  // a consistent time base while ramping down through invalid frames.
  const rclcpp::Time now = msg->header.stamp;
  float dt = 0.05f;  // default 20 Hz
  if (prev_time_valid_) {
    const float measured = static_cast<float>((now - prev_time_).seconds());
    if (measured > 0.0f && measured < 1.0f) dt = measured;
  }
  prev_time_       = now;
  prev_time_valid_ = true;

  // Decide the target command. An invalid error means "stop", but we ramp the
  // command toward zero instead of snapping to it, so a brief detection dropout
  // no longer produces a hard stop/go stutter.
  float target_vx = 0.0f;
  float target_wz = 0.0f;
  float v_scale   = 0.0f;

  if (msg->valid) {
    // Capture initial distance for decel ramp
    if (!initial_dist_set_ && msg->initial_dist_m > 0.0f) {
      initial_dist_    = msg->initial_dist_m;
      initial_dist_set_= true;
    }

    // Decel ramp (using initial_dist and current x_error)
    v_scale = 1.0f;
    if (initial_dist_set_ && initial_dist_ > 0.0f) {
      const float eff_decel = std::clamp(
          initial_dist_ * decel_ratio_, decel_dist_min_, decel_dist_max_);
      v_scale = std::clamp(std::abs(msg->x_error) / eff_decel, 0.0f, 1.0f);
    }

    // Build SE2Error: x from PC, y=0, degree_theta from image (already in rad)
    SE2Error se2;
    se2.x            = msg->x_error;
    se2.y            = 0.0f;
    se2.degree_theta = msg->theta_error;  // PIDController expects radians despite the name

    const auto cmd = pid_->compute_control(se2, dt, v_scale);
    target_vx = static_cast<float>(cmd.linear.x);
    target_wz = static_cast<float>(cmd.angular.z);
  }

  // Slew-rate limit both axes so discrete x updates, yaw initialization, and
  // validity toggles turn into continuous motion.
  last_vx_ = slew(last_vx_, target_vx, dt, lin_accel_limit_, lin_decel_limit_);
  last_wz_ = slew(last_wz_, target_wz, dt, ang_accel_limit_, ang_accel_limit_);

  geometry_msgs::msg::Twist out;
  out.linear.x  = last_vx_;
  out.angular.z = last_wz_;
  cmd_vel_pub_->publish(out);

  if (debug_log_) {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 300,
        "control valid=%d x=%.4f theta=%.4f rad dt=%.3f v_scale=%.2f "
        "target(vx=%.4f wz=%.4f) -> cmd(vx=%.4f wz=%.4f)",
        msg->valid, msg->x_error, msg->theta_error, dt, v_scale,
        target_vx, target_wz, last_vx_, last_wz_);
  }
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ApproachController>());
  rclcpp::shutdown();
  return 0;
}
