#include "close_approach/approach_manager.hpp"

#include <chrono>
#include <cmath>
#include <thread>

#include <tf2/utils.h>

ApproachManager::ApproachManager()
    : Node("approach_manager"),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_) {

  // Params
  this->declare_parameter<float>("tol_x",              0.03F);
  this->declare_parameter<float>("tol_theta",          0.08F);
  this->declare_parameter<float>("dwell_duration_sec", 2.0F);
  this->declare_parameter<float>("align_timeout_sec",  5.0F);
  this->declare_parameter<float>("pc_timeout_sec",     2.0F);
  this->declare_parameter<float>("x_hold_sec",         0.2F);
  this->declare_parameter<float>("x_converged_hold_sec", 0.5F);
  this->declare_parameter<float>("trail_min_dist",     0.03F);
  this->declare_parameter<float>("trail_min_yaw",      0.052F);
  this->declare_parameter<std::string>("target_frame", "base_nav");
  this->declare_parameter<std::string>("odom_frame",   "odom");
  this->declare_parameter<std::string>("edge_enable_service_name",
                                       "/approach/edge_detector/set_enable");
  this->declare_parameter<bool>("debug_log",           true);

  this->get_parameter("tol_x",              tol_x_);
  this->get_parameter("tol_theta",          tol_theta_);
  this->get_parameter("dwell_duration_sec", dwell_duration_sec_);
  this->get_parameter("align_timeout_sec",  align_timeout_sec_);
  this->get_parameter("pc_timeout_sec",     pc_timeout_sec_);
  this->get_parameter("x_hold_sec",         x_hold_sec_);
  this->get_parameter("x_converged_hold_sec", x_converged_hold_sec_);
  this->get_parameter("trail_min_dist",     trail_min_dist_);
  this->get_parameter("trail_min_yaw",      trail_min_yaw_);
  this->get_parameter("target_frame",       target_frame_);
  this->get_parameter("odom_frame",         odom_frame_);
  this->get_parameter("edge_enable_service_name", edge_enable_service_name_);
  this->get_parameter("debug_log",          debug_log_);

  auto qos_rel = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

  // Subscribers
  pc_error_sub_ = this->create_subscription<ApproachError>(
      "/approach/pc_error", qos_rel,
      [this](const ApproachError::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(pc_mutex_);
        latest_pc_error_ = msg;
        if (msg->valid) {
          last_valid_pc_time_ = this->now();
          last_good_x_err_    = msg->x_error;
        }
      });

  edge_error_sub_ = this->create_subscription<ApproachError>(
      "/approach/edge_error", qos_rel,
      [this](const ApproachError::SharedPtr msg) {
        if (msg->valid) {
          last_theta_rad_    = msg->theta_error;
          theta_initialized_ = true;
        }
        if (debug_log_) {
          RCLCPP_INFO_THROTTLE(
              this->get_logger(), *this->get_clock(), 300,
              "edge_error rx valid=%d theta=%.4f rad (%.2f deg) initialized=%d mean_y=%.1f",
              msg->valid, msg->theta_error,
              msg->theta_error * 180.0f / static_cast<float>(M_PI),
              theta_initialized_, msg->mean_y_px);
        }
      });

  // Publishers
  control_error_pub_ = this->create_publisher<ApproachError>(
      "/approach/control_error", qos_rel);

  active_pub_ = this->create_publisher<std_msgs::msg::Bool>(
      "/approach/active", qos_rel);

  rclcpp::QoS trail_qos(rclcpp::KeepLast(1));
  trail_qos.reliable();
  trail_qos.transient_local();
  trail_pub_ = this->create_publisher<nav_msgs::msg::Path>(
      "/approach/trail", trail_qos);

  state_pub_ = this->create_publisher<std_msgs::msg::String>(
      "/approach/state", qos_rel);
  edge_enable_client_ =
      this->create_client<inha_interfaces::srv::SetEnable>(
          edge_enable_service_name_);

  // Action server
  action_server_ = rclcpp_action::create_server<Approach>(
      this, "approach",
      std::bind(&ApproachManager::handleGoal,     this,
                std::placeholders::_1, std::placeholders::_2),
      std::bind(&ApproachManager::handleCancel,   this, std::placeholders::_1),
      std::bind(&ApproachManager::handleAccepted, this, std::placeholders::_1));

  // 20 Hz timers
  state_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&ApproachManager::stateMachineCallback, this));
  trail_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&ApproachManager::recordTrailPose, this));

  RCLCPP_INFO(this->get_logger(), "ApproachManager ready");
}

// ─── Action server ──────────────────────────────────────────────────────────

rclcpp_action::GoalResponse ApproachManager::handleGoal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const Approach::Goal> goal) {
  if (!goal || !std::isfinite(goal->goal_distance) || goal->goal_distance <= 0.0F) {
    RCLCPP_WARN(this->get_logger(), "Rejecting invalid goal");
    return rclcpp_action::GoalResponse::REJECT;
  }
  RCLCPP_INFO(this->get_logger(), "Goal accepted: standoff=%.3f", goal->goal_distance);
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse ApproachManager::handleCancel(
    std::shared_ptr<GoalHandle>) {
  RCLCPP_INFO(this->get_logger(), "Cancel received");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void ApproachManager::handleAccepted(std::shared_ptr<GoalHandle> gh) {
  std::thread(std::bind(&ApproachManager::execute, this, gh)).detach();
}

void ApproachManager::execute(std::shared_ptr<GoalHandle> gh) {
  const auto goal = gh->get_goal();
  startApproach(goal->goal_distance);

  auto feedback = std::make_shared<Approach::Feedback>();
  auto result   = std::make_shared<Approach::Result>();
  rclcpp::Rate rate(10);

  while (rclcpp::ok()) {
    if (gh->is_canceling()) {
      result->success = false;
      result->success_message = "Cancelled";
      stopApproach();
      gh->canceled(result);
      return;
    }
    if (action_failed_) {
      result->success = false;
      result->success_message = "Approach failed";
      stopApproach();
      gh->abort(result);
      return;
    }
    if (action_succeeded_) {
      result->success = true;
      result->success_message = "Approach succeeded";
      stopApproach();
      gh->succeed(result);
      return;
    }

    // Publish feedback
    {
      std::lock_guard<std::mutex> lk(pc_mutex_);
      if (latest_pc_error_) {
        feedback->x_error     = latest_pc_error_->x_error;
        feedback->y_error     = latest_pc_error_->y_error;
        feedback->theta_error = static_cast<float>(last_theta_rad_);
        gh->publish_feedback(feedback);
      }
    }
    rate.sleep();
  }

  result->success = false;
  result->success_message = "ROS shutdown";
  stopApproach();
  gh->abort(result);
}

// ─── State machine (20 Hz) ──────────────────────────────────────────────────

void ApproachManager::stateMachineCallback() {
  if (state_ == State::IDLE) return;

  const auto now = this->now();

  // Read latest PC error
  float x_err  = 0.0f;
  bool  x_valid = false;
  bool  x_held  = false;
  {
    std::lock_guard<std::mutex> lk(pc_mutex_);
    const double since_valid = (now - last_valid_pc_time_).seconds();
    if (latest_pc_error_ && latest_pc_error_->valid &&
        since_valid <= static_cast<double>(pc_timeout_sec_)) {
      x_err   = latest_pc_error_->x_error;
      x_valid = true;
      if (!initial_dist_set_) {
        initial_dist_    = latest_pc_error_->x_error;
        initial_dist_set_= true;
      }
    } else if (initial_dist_set_ &&
               since_valid <= static_cast<double>(x_hold_sec_)) {
      // Bridge a brief invalid burst (a dropped frame or the short spike-verify
      // window in pc_detector) with the last good longitudinal estimate, so the
      // controller does not see validity toggle and command a hard stop/go.
      x_err   = last_good_x_err_;
      x_valid = true;
      x_held  = true;
    }
  }

  // PC timeout (only warn in APPROACH)
  const bool pc_timed_out = !x_valid &&
      (now - last_valid_pc_time_).seconds() >
          static_cast<double>(pc_timeout_sec_);

  const float theta_err = last_theta_rad_;
  if (debug_log_) {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 300,
        "manager state=%s x_valid=%d held=%d x=%.4f theta=%.4f rad (%.2f deg) theta_init=%d",
        stateStr(), x_valid, x_held, x_err, theta_err,
        theta_err * 180.0f / static_cast<float>(M_PI), theta_initialized_);
  }

  // Build control_error to send to controller
  ApproachError ctrl;
  ctrl.header.stamp    = now;
  ctrl.x_error         = x_err;
  ctrl.y_error         = 0.0f;
  ctrl.theta_error     = theta_err;
  ctrl.initial_dist_m  = initial_dist_;

  // Publish state string
  {
    std_msgs::msg::String s;
    s.data = stateStr();
    state_pub_->publish(s);
  }

  switch (state_) {
    case State::IDLE: return;

    case State::APPROACH: {
      ctrl.valid = x_valid;
      control_error_pub_->publish(ctrl);

      if (pc_timed_out) {
        RCLCPP_ERROR(this->get_logger(), "PC timeout in APPROACH → action failed");
        action_failed_ = true;
        break;
      }
      if (!x_valid || std::abs(x_err) >= tol_x_) {
        x_convergence_pending_ = false;
        break;
      }

      if (!x_convergence_pending_) {
        x_convergence_pending_ = true;
        x_converged_since_ = now;
        RCLCPP_INFO(this->get_logger(),
                    "Longitudinal tolerance entered: x=%.3f, verifying %.2fs",
                    x_err, x_converged_hold_sec_);
        break;
      }
      if ((now - x_converged_since_).seconds() <
          static_cast<double>(x_converged_hold_sec_)) {
        break;
      }

      {
        if (theta_initialized_ && std::abs(theta_err) < tol_theta_) {
          RCLCPP_INFO(this->get_logger(), "APPROACH → DWELL (x=%.3f θ=%.3f)",
                      x_err, theta_err);
          state_       = State::DWELL;
          dwell_start_ = now;
        } else {
          RCLCPP_INFO(this->get_logger(), "APPROACH → ALIGN_THETA (x=%.3f θ=%.3f)",
                      x_err, theta_err);
          state_       = State::ALIGN_THETA;
          align_start_ = now;
        }
      }
      break;
    }

    case State::ALIGN_THETA: {
      ctrl.x_error = 0.0f;   // theta-only control
      ctrl.valid   = true;
      control_error_pub_->publish(ctrl);

      if (x_valid && std::abs(x_err) >= tol_x_) {
        x_convergence_pending_ = false;
        RCLCPP_WARN(this->get_logger(),
                    "ALIGN_THETA → APPROACH (x moved out: %.3f)", x_err);
        state_ = State::APPROACH;
      } else if (pc_timed_out) {
        RCLCPP_ERROR(this->get_logger(),
                     "PC timeout in ALIGN_THETA → action failed");
        action_failed_ = true;
      } else if (theta_initialized_ && std::abs(theta_err) < tol_theta_) {
        if (!x_valid) {
          // Wait for a fresh longitudinal estimate; never declare DWELL from
          // an aligned image alone.
          break;
        }
        RCLCPP_INFO(this->get_logger(), "ALIGN_THETA → DWELL");
        state_       = State::DWELL;
        dwell_start_ = now;
      } else if ((now - align_start_).seconds() > static_cast<double>(align_timeout_sec_)) {
        RCLCPP_ERROR(this->get_logger(), "ALIGN_THETA timeout → action failed");
        action_failed_ = true;
      }
      break;
    }

    case State::DWELL: {
      ctrl.valid = false;
      control_error_pub_->publish(ctrl);

      // DWELL is a stability check, not an unconditional success delay.
      // A noisy one-frame x/theta estimate must return to active correction.
      if (pc_timed_out) {
        RCLCPP_ERROR(this->get_logger(), "PC timeout in DWELL → action failed");
        action_failed_ = true;
        break;
      }
      if (!x_valid || !theta_initialized_) {
        dwell_start_ = now;
        break;
      }
      if (std::abs(x_err) >= tol_x_) {
        RCLCPP_WARN(this->get_logger(), "DWELL → APPROACH (x drift %.3f)",
                    x_err);
        state_ = State::APPROACH;
        x_convergence_pending_ = false;
        break;
      }
      if (std::abs(theta_err) >= tol_theta_) {
        RCLCPP_WARN(this->get_logger(), "DWELL → ALIGN_THETA (theta drift %.3f)",
                    theta_err);
        state_ = State::ALIGN_THETA;
        align_start_ = now;
        break;
      }
      if ((now - dwell_start_).seconds() > static_cast<double>(dwell_duration_sec_)) {
        state_            = State::DONE;
        action_succeeded_ = true;
        RCLCPP_INFO(this->get_logger(), "DWELL → DONE");
      }
      break;
    }

    case State::DONE:
      break;
  }
}

// ─── Trail ──────────────────────────────────────────────────────────────────

void ApproachManager::recordTrailPose() {
  if (!approach_active_) return;

  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_.lookupTransform(odom_frame_, target_frame_,
                                    tf2::TimePointZero,
                                    tf2::durationFromSec(0.05));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_DEBUG(this->get_logger(), "trail TF failed: %s", ex.what());
    return;
  }

  geometry_msgs::msg::PoseStamped pose;
  pose.header = tf.header;
  pose.pose.position.x  = tf.transform.translation.x;
  pose.pose.position.y  = tf.transform.translation.y;
  pose.pose.position.z  = tf.transform.translation.z;
  pose.pose.orientation = tf.transform.rotation;

  std::lock_guard<std::mutex> lk(trail_mutex_);
  if (trail_.empty()) { trail_.push_back(pose); return; }

  const auto &last = trail_.back().pose;
  const float dx = (float)(pose.pose.position.x - last.position.x);
  const float dy = (float)(pose.pose.position.y - last.position.y);
  const float dist = std::sqrt(dx*dx + dy*dy);
  double dyaw = tf2::getYaw(pose.pose.orientation) - tf2::getYaw(last.orientation);
  while (dyaw >  M_PI) dyaw -= 2.0*M_PI;
  while (dyaw < -M_PI) dyaw += 2.0*M_PI;

  if (dist >= trail_min_dist_ || std::abs(dyaw) >= (double)trail_min_yaw_) {
    trail_.push_back(pose);
  }
}

void ApproachManager::publishTrail() {
  nav_msgs::msg::Path path;
  {
    std::lock_guard<std::mutex> lk(trail_mutex_);
    path.poses.assign(trail_.begin(), trail_.end());
  }
  path.header.stamp    = this->now();
  path.header.frame_id = odom_frame_;
  trail_pub_->publish(path);
  RCLCPP_INFO(this->get_logger(), "Trail published: %zu poses", path.poses.size());
}

// ─── Start / Stop ────────────────────────────────────────────────────────────

void ApproachManager::startApproach(float standoff) {
  standoff_distance_   = standoff;
  state_               = State::APPROACH;
  post_align_done_     = false;
  initial_dist_set_    = false;
  initial_dist_        = 0.0f;
  action_succeeded_    = false;
  action_failed_       = false;
  theta_initialized_   = false;
  last_theta_rad_      = 0.0f;
  x_convergence_pending_ = false;
  last_valid_pc_time_  = this->now();

  {
    std::lock_guard<std::mutex> lk(pc_mutex_);
    latest_pc_error_.reset();
    last_good_x_err_ = 0.0f;
  }

  {
    std::lock_guard<std::mutex> lk(trail_mutex_);
    trail_.clear();
  }

  approach_active_ = true;
  setEdgeDetectorEnabled(true);

  std_msgs::msg::Bool active_msg;
  active_msg.data = true;
  active_pub_->publish(active_msg);

  RCLCPP_INFO(this->get_logger(), "Approach started (standoff=%.3f)", standoff);
}

void ApproachManager::stopApproach() {
  setEdgeDetectorEnabled(false);

  approach_active_ = false;
  state_           = State::IDLE;

  // Every terminal path (success/failure/cancel/shutdown) comes through here.
  // Never let a following action inherit sensor or convergence state.
  {
    std::lock_guard<std::mutex> lk(pc_mutex_);
    latest_pc_error_.reset();
    last_good_x_err_ = 0.0f;
    last_valid_pc_time_ = this->now();
  }
  theta_initialized_ = false;
  last_theta_rad_ = 0.0F;
  initial_dist_set_ = false;
  initial_dist_ = 0.0F;
  x_convergence_pending_ = false;
  post_align_done_ = false;

  std_msgs::msg::Bool active_msg;
  active_msg.data = false;
  active_pub_->publish(active_msg);

  publishTrail();
  RCLCPP_INFO(this->get_logger(), "Approach stopped");
}

void ApproachManager::setEdgeDetectorEnabled(bool enabled) {
  if (!edge_enable_client_) {
    return;
  }

  if (!edge_enable_client_->wait_for_service(std::chrono::milliseconds(300))) {
    RCLCPP_WARN(this->get_logger(),
                "Edge detector enable service unavailable: %s",
                edge_enable_service_name_.c_str());
    return;
  }

  auto req = std::make_shared<inha_interfaces::srv::SetEnable::Request>();
  req->enable = enabled;
  auto future = edge_enable_client_->async_send_request(req);
  (void)future;
  RCLCPP_INFO(this->get_logger(), "Requested edge detector %s",
              enabled ? "enable" : "disable");
}

const char *ApproachManager::stateStr() const {
  switch (state_) {
    case State::IDLE:        return "IDLE";
    case State::APPROACH:    return "APPROACH";
    case State::ALIGN_THETA: return "ALIGN_THETA";
    case State::DWELL:       return "DWELL";
    case State::DONE:        return "DONE";
    default:                 return "UNKNOWN";
  }
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ApproachManager>());
  rclcpp::shutdown();
  return 0;
}
