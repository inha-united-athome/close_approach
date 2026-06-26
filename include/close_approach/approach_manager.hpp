#pragma once

#include "close_approach/msg/approach_error.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include "inha_interfaces/action/approach.hpp"
#include "inha_interfaces/srv/set_enable.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class ApproachManager : public rclcpp::Node {
public:
  ApproachManager();

private:
  using ApproachError = close_approach::msg::ApproachError;
  using Approach      = inha_interfaces::action::Approach;
  using GoalHandle    = rclcpp_action::ServerGoalHandle<Approach>;

  enum class State { IDLE, APPROACH, ALIGN_THETA, DWELL, DONE };

  // Action server
  rclcpp_action::Server<Approach>::SharedPtr action_server_;
  rclcpp_action::GoalResponse   handleGoal(const rclcpp_action::GoalUUID &,
                                           std::shared_ptr<const Approach::Goal>);
  rclcpp_action::CancelResponse handleCancel(std::shared_ptr<GoalHandle>);
  void handleAccepted(std::shared_ptr<GoalHandle>);
  void execute(std::shared_ptr<GoalHandle>);

  // Subscribers
  rclcpp::Subscription<ApproachError>::SharedPtr pc_error_sub_;
  rclcpp::Subscription<ApproachError>::SharedPtr edge_error_sub_;

  // Publishers
  rclcpp::Publisher<ApproachError>::SharedPtr     control_error_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr  active_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr  trail_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Client<inha_interfaces::srv::SetEnable>::SharedPtr edge_enable_client_;

  // Timers
  rclcpp::TimerBase::SharedPtr state_timer_;
  rclcpp::TimerBase::SharedPtr trail_timer_;

  // TF (for trail recording)
  tf2_ros::Buffer            tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Cached sensor inputs
  ApproachError::SharedPtr latest_pc_error_;
  std::mutex               pc_mutex_;
  std::uint64_t            pc_error_rx_count_ = 0;
  std::uint64_t            pc_error_valid_count_ = 0;
  float                    last_good_x_err_   = 0.0f;  // last valid longitudinal x
  float                    last_theta_rad_    = 0.0f;
  bool                     theta_initialized_ = false;
  rclcpp::Time             last_theta_time_;           // time of last fresh edge theta
  float                    last_pc_yaw_       = 0.0f;   // plane-normal yaw fallback
  bool                     pc_yaw_valid_      = false;
  rclcpp::Time             last_pc_yaw_time_;
  float                    last_known_theta_  = 0.0f;   // last actually-measured yaw
  bool                     yaw_ever_measured_ = false;

  // State machine
  State       state_               = State::IDLE;
  rclcpp::Time dwell_start_;
  rclcpp::Time align_start_;
  rclcpp::Time x_converged_since_;
  bool         x_convergence_pending_ = false;
  rclcpp::Time yaw_jitter_since_;
  bool         yaw_jitter_tracking_ = false;
  float        yaw_jitter_min_theta_ = 0.0f;
  float        yaw_jitter_max_theta_ = 0.0f;
  int          yaw_jitter_last_sign_ = 0;
  int          yaw_jitter_sign_flips_ = 0;
  bool         post_align_done_    = false;
  float        initial_dist_       = 0.0f;
  bool         initial_dist_set_   = false;
  float        standoff_distance_  = 0.3f;

  // Params
  float tol_x_, tol_theta_;
  float dwell_duration_sec_, align_timeout_sec_;
  float x_converged_hold_sec_;
  float trail_min_dist_, trail_min_yaw_;
  float pc_timeout_sec_;
  float x_hold_sec_;       // bridge brief invalid PC bursts with last good x
  float theta_timeout_sec_;  // no fresh yaw for this long -> stop rotating
  float yaw_jitter_success_sec_;
  float yaw_jitter_min_range_rad_;
  int yaw_jitter_min_sign_flips_;
  // Terminal acceptance: on give-up (PC/align timeout) succeed if the pose is
  // good enough, fail only if x was lost while far OR yaw is badly off.
  float yaw_fail_rad_;     // |yaw| >= this at give-up -> FAIL
  float x_fail_dist_;      // |x_error| >= this when lost -> FAIL (lost while far)
  rclcpp::Time last_valid_pc_time_;
  std::string  odom_frame_, target_frame_;
  std::string  edge_enable_service_name_;
  bool debug_log_ = true;
  std::string failure_message_;

  // Trail
  std::deque<geometry_msgs::msg::PoseStamped> trail_;
  std::mutex                                  trail_mutex_;

  // Action flags (atomic: timer-thread ↔ execute-thread)
  std::atomic<bool> action_succeeded_{false};
  std::atomic<bool> action_failed_{false};
  std::atomic<bool> approach_active_{false};

  void stateMachineCallback();
  void recordTrailPose();
  void publishTrail();
  void startApproach(float standoff);
  void stopApproach();
  void setEdgeDetectorEnabled(bool enabled);
  void resetYawJitterWatchdog();
  bool updateYawJitterWatchdog(const rclcpp::Time &now, bool x_ready,
                               bool theta_fresh, float theta_err);
  float goalRelativeXError(const ApproachError &msg) const;
  const char *stateStr() const;
};
