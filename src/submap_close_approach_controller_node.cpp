#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

#include "close_approach/msg/close_approach_error.hpp"

namespace close_approach
{
namespace
{
float clampFloat(const float value, const float min_value, const float max_value)
{
  return std::max(min_value, std::min(max_value, value));
}
}  // namespace

class SubmapCloseApproachControllerNode : public rclcpp::Node
{
public:
  SubmapCloseApproachControllerNode()
  : Node("submap_close_approach_controller_node")
  {
    declareParameters();
    loadParameters();

    last_error_time_ = now();
    in_tolerance_since_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());

    error_sub_ = create_subscription<close_approach::msg::CloseApproachError>(
      error_topic_, 10,
      std::bind(&SubmapCloseApproachControllerNode::errorCallback, this, std::placeholders::_1));
    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

    watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&SubmapCloseApproachControllerNode::watchdogCallback, this));

    RCLCPP_INFO(
      get_logger(), "Submap close approach controller started. error=%s cmd=%s",
      error_topic_.c_str(), cmd_vel_topic_.c_str());
  }

private:
  void declareParameters()
  {
    declare_parameter<std::string>("error_topic", "/close_approach/error");
    declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    declare_parameter<bool>("enable_control", true);
    declare_parameter<bool>("use_lateral_control", true);
    declare_parameter<bool>("stop_on_invalid", true);

    declare_parameter<double>("kp_x", 0.25);
    declare_parameter<double>("kp_y", 1.0);
    declare_parameter<double>("kp_theta", 1.2);

    declare_parameter<double>("max_vx", 0.10);
    declare_parameter<double>("max_vy", 0.08);
    declare_parameter<double>("max_wz", 0.20);
    declare_parameter<double>("min_confidence", 0.0);

    declare_parameter<double>("tol_x", 0.05);
    declare_parameter<double>("tol_y", 0.04);
    declare_parameter<double>("tol_theta", 0.08);
    declare_parameter<double>("dwell_duration_sec", 1.0);
    declare_parameter<double>("watchdog_timeout_sec", 0.5);
  }

  void loadParameters()
  {
    error_topic_ = get_parameter("error_topic").as_string();
    cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
    enable_control_ = get_parameter("enable_control").as_bool();
    use_lateral_control_ = get_parameter("use_lateral_control").as_bool();
    stop_on_invalid_ = get_parameter("stop_on_invalid").as_bool();

    kp_x_ = get_parameter("kp_x").as_double();
    kp_y_ = get_parameter("kp_y").as_double();
    kp_theta_ = get_parameter("kp_theta").as_double();

    max_vx_ = get_parameter("max_vx").as_double();
    max_vy_ = get_parameter("max_vy").as_double();
    max_wz_ = get_parameter("max_wz").as_double();
    min_confidence_ = get_parameter("min_confidence").as_double();

    tol_x_ = get_parameter("tol_x").as_double();
    tol_y_ = get_parameter("tol_y").as_double();
    tol_theta_ = get_parameter("tol_theta").as_double();
    dwell_duration_sec_ = get_parameter("dwell_duration_sec").as_double();
    watchdog_timeout_sec_ = get_parameter("watchdog_timeout_sec").as_double();
  }

  void errorCallback(const close_approach::msg::CloseApproachError::SharedPtr msg)
  {
    last_error_time_ = now();
    has_error_ = true;

    if (!enable_control_) {
      return;
    }

    if (!msg->valid || msg->confidence < min_confidence_) {
      in_tolerance_since_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
      if (stop_on_invalid_) {
        publishStop();
      }
      return;
    }

    const bool in_tolerance =
      std::abs(msg->x_error) < tol_x_ &&
      std::abs(msg->y_error) < tol_y_ &&
      std::abs(msg->theta_error) < tol_theta_;

    if (in_tolerance) {
      if (in_tolerance_since_.nanoseconds() == 0) {
        in_tolerance_since_ = now();
      }
      if ((now() - in_tolerance_since_).seconds() >= dwell_duration_sec_) {
        publishStop();
        return;
      }
    } else {
      in_tolerance_since_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    }

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = clampFloat(
      static_cast<float>(kp_x_ * msg->x_error),
      static_cast<float>(-max_vx_), static_cast<float>(max_vx_));

    if (use_lateral_control_) {
      cmd.linear.y = clampFloat(
        static_cast<float>(kp_y_ * msg->y_error),
        static_cast<float>(-max_vy_), static_cast<float>(max_vy_));
    }

    cmd.angular.z = clampFloat(
      static_cast<float>(kp_theta_ * msg->theta_error),
      static_cast<float>(-max_wz_), static_cast<float>(max_wz_));

    cmd_pub_->publish(cmd);
  }

  void watchdogCallback()
  {
    if (!enable_control_ || !has_error_ || !stop_on_invalid_) {
      return;
    }
    if ((now() - last_error_time_).seconds() > watchdog_timeout_sec_) {
      publishStop();
    }
  }

  void publishStop()
  {
    geometry_msgs::msg::Twist cmd;
    cmd_pub_->publish(cmd);
  }

  rclcpp::Subscription<close_approach::msg::CloseApproachError>::SharedPtr error_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;

  std::string error_topic_;
  std::string cmd_vel_topic_;
  bool enable_control_ = true;
  bool use_lateral_control_ = true;
  bool stop_on_invalid_ = true;

  double kp_x_ = 0.25;
  double kp_y_ = 1.0;
  double kp_theta_ = 1.2;
  double max_vx_ = 0.10;
  double max_vy_ = 0.08;
  double max_wz_ = 0.20;
  double min_confidence_ = 0.0;
  double tol_x_ = 0.05;
  double tol_y_ = 0.04;
  double tol_theta_ = 0.08;
  double dwell_duration_sec_ = 1.0;
  double watchdog_timeout_sec_ = 0.5;

  bool has_error_ = false;
  rclcpp::Time last_error_time_;
  rclcpp::Time in_tolerance_since_;
};
}  // namespace close_approach

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<close_approach::SubmapCloseApproachControllerNode>());
  rclcpp::shutdown();
  return 0;
}
