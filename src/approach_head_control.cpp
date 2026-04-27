#include "inha_interfaces/action/approach.hpp"
#include "inha_interfaces/action/set_head_pose.hpp"

#include <chrono>
#include <cmath>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/trigger.hpp>

using namespace std::chrono_literals;

using SetHeadPose = inha_interfaces::action::SetHeadPose;
using GoalHandleSetHeadPose = rclcpp_action::ClientGoalHandle<SetHeadPose>;

class FeedbackMonitorNode : public rclcpp::Node {
public:
  FeedbackMonitorNode() : Node("feedback_monitor_node") {
    feedback_sub_ =
        this->create_subscription<inha_interfaces::action::Approach_FeedbackMessage>(
            "/approach/_action/feedback", rclcpp::QoS(10),
            std::bind(&FeedbackMonitorNode::feedback_callback, this,
                      std::placeholders::_1));

    head_pose_client_ =
        rclcpp_action::create_client<SetHeadPose>(this, "/rby1/set_head_pose");

    reset_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/approach/feedback_monitor/reset",
        std::bind(&FeedbackMonitorNode::reset_callback, this,
                  std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Feedback monitor node started.");
  }

private:
  // =========================
  // Parameters / thresholds
  // =========================

  // LPF: filtered = alpha * raw + (1-alpha) * prev
  const float filter_alpha_ = 0.15f;

  // candidate zone이 이 시간 이상 유지되어야 current zone으로 확정
  const double debounce_sec_ = 0.5;

  // head pose command 전송 후 최소 이 시간 동안은 새 명령 안 보냄
  const double cooldown_sec_ = 1.5;

  // 너무 작은 각도 차이면 재전송 안 함
  const float min_head1_change_deg_ = 3.0f;
  const float min_head0_change_deg_ = 3.0f;

  // =========================
  // State
  // =========================

  // 확정된 zone
  int current_zone_ = -1;

  // 관측 중인 후보 zone
  int candidate_zone_ = -1;
  rclcpp::Time candidate_since_{0, 0, RCL_ROS_TIME};

  // 필터링된 error_x
  bool filter_initialized_ = false;
  float filtered_error_x_ = 0.0f;

  // 액션 상태
  bool head_goal_active_ = false;
  rclcpp::Time last_head_cmd_time_{0, 0, RCL_ROS_TIME};

  // 마지막으로 보낸 목표값
  bool last_head_pose_sent_valid_ = false;
  float last_head_0_sent_ = 0.0f;
  float last_head_1_sent_ = 0.0f;

  // =========================
  // Callbacks
  // =========================

  void feedback_callback(
      const inha_interfaces::action::Approach_FeedbackMessage::SharedPtr msg) {
    const float error_x = msg->feedback.x_error;
    const float error_y = msg->feedback.y_error;
    const float error_theta = msg->feedback.theta_error;

    // 1) low-pass filter
    if (!filter_initialized_) {
      filtered_error_x_ = error_x;
      filter_initialized_ = true;
    } else {
      filtered_error_x_ =
          filter_alpha_ * error_x + (1.0f - filter_alpha_) * filtered_error_x_;
    }

    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 500,
        "raw_x: %.3f, filtered_x: %.3f, y: %.3f, theta: %.3f, current_zone: %d, candidate_zone: %d",
        error_x, filtered_error_x_, error_y, error_theta, current_zone_,
        candidate_zone_);

    // 2) hysteresis zone determination
    const int observed_zone = determine_zone(filtered_error_x_);
    const auto now = this->now();

    // 3) debounce
    if (observed_zone != candidate_zone_) {
      candidate_zone_ = observed_zone;
      candidate_since_ = now;

      RCLCPP_INFO(this->get_logger(),
                  "Candidate zone changed to %d. Waiting for debounce...",
                  candidate_zone_);
      return;
    }

    if ((now - candidate_since_).seconds() < debounce_sec_) {
      return;
    }

    // 4) only act when confirmed zone changes
    if (candidate_zone_ != current_zone_) {
      const int prev_zone = current_zone_;
      current_zone_ = candidate_zone_;

      RCLCPP_INFO(this->get_logger(),
                  "Zone confirmed: %d -> %d",
                  prev_zone, current_zone_);

      handle_zone_change(current_zone_);
    }
  }

  void reset_callback(
      const std_srvs::srv::Trigger::Request::SharedPtr /*req*/,
      std_srvs::srv::Trigger::Response::SharedPtr res) {
    current_zone_ = -1;
    candidate_zone_ = -1;
    filter_initialized_ = false;
    filtered_error_x_ = 0.0f;
    head_goal_active_ = false;
    last_head_pose_sent_valid_ = false;

    RCLCPP_INFO(this->get_logger(), "Feedback monitor state reset.");

    res->success = true;
    res->message = "Reset done";
  }

  // =========================
  // Zone logic
  // =========================

int determine_zone(float x) {
  if (current_zone_ == 0) {
    if (x > 0.35f)
      return 0;   // zone 0 유지
    else if (x > 0.08f)
      return 1;   // zone 1로 내려감
    else
      return -1;
  }

  if (current_zone_ == 1) {
    if (x > 0.45f)
      return 0;   // zone 0로 올라감
    else if (x > 0.08f)
      return 1;   // zone 1 유지
    else
      return -1;
  }

  // 초기 상태
  if (x > 0.4f)
    return 0;
  else if (x > 0.1f)
    return 1;
  else
    return -1;
}

  void handle_zone_change(int zone) {
    const auto now = this->now();

    if (zone == -1) {
      RCLCPP_INFO(this->get_logger(),
                  "Zone is -1. No head pose command will be sent.");
      return;
    }

    if (head_goal_active_) {
      RCLCPP_INFO(this->get_logger(),
                  "Head goal is still active. Skip new command.");
      return;
    }

    if ((now - last_head_cmd_time_).seconds() < cooldown_sec_) {
      RCLCPP_INFO(this->get_logger(),
                  "Cooldown active (%.2f sec remaining). Skip new command.",
                  cooldown_sec_ - (now - last_head_cmd_time_).seconds());
      return;
    }

    float target_head_0 = 0.0f;
    float target_head_1 = 0.0f;
    float duration = 1.0f;

    if (zone == 0) {
      target_head_0 = 0.0f;
      target_head_1 = 20.0f;
    } else if (zone == 1) {
      target_head_0 = 0.0f;
      target_head_1 = 35.0f;
    } else {
      return;
    }

    if (last_head_pose_sent_valid_) {
      const float d0 = std::fabs(target_head_0 - last_head_0_sent_);
      const float d1 = std::fabs(target_head_1 - last_head_1_sent_);

      if (d0 < min_head0_change_deg_ && d1 < min_head1_change_deg_) {
        RCLCPP_INFO(this->get_logger(),
                    "Target head pose change too small. Skip sending. "
                    "(d0=%.2f, d1=%.2f)",
                    d0, d1);
        return;
      }
    }

    send_head_pose(target_head_0, target_head_1, duration);
  }

  // =========================
  // Action sending
  // =========================

  void send_head_pose(float head_0, float head_1, float duration) {
    if (!head_pose_client_->wait_for_action_server(1s)) {
      RCLCPP_ERROR(this->get_logger(),
                   "SetHeadPose action server not available.");
      return;
    }

    auto goal = SetHeadPose::Goal();
    goal.head_0 = head_0;
    goal.head_1 = head_1;
    goal.duration = duration;

    auto options = rclcpp_action::Client<SetHeadPose>::SendGoalOptions();

    options.goal_response_callback =
        [this, head_0, head_1](GoalHandleSetHeadPose::SharedPtr goal_handle) {
          if (!goal_handle) {
            head_goal_active_ = false;
            RCLCPP_WARN(this->get_logger(), "SetHeadPose goal rejected.");
            return;
          }

          head_goal_active_ = true;
          last_head_cmd_time_ = this->now();
          last_head_pose_sent_valid_ = true;
          last_head_0_sent_ = head_0;
          last_head_1_sent_ = head_1;

          RCLCPP_INFO(this->get_logger(),
                      "SetHeadPose goal accepted. "
                      "head_0=%.2f, head_1=%.2f",
                      head_0, head_1);
        };

    options.result_callback =
        [this](const GoalHandleSetHeadPose::WrappedResult &result) {
          head_goal_active_ = false;

          switch (result.code) {
          case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(this->get_logger(), "SetHeadPose succeeded.");
            break;
          case rclcpp_action::ResultCode::ABORTED:
            RCLCPP_WARN(this->get_logger(), "SetHeadPose aborted.");
            break;
          case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_WARN(this->get_logger(), "SetHeadPose canceled.");
            break;
          default:
            RCLCPP_WARN(this->get_logger(),
                        "SetHeadPose finished with unknown result code.");
            break;
          }
        };

    RCLCPP_INFO(this->get_logger(),
                "Sending SetHeadPose goal: head_0=%.2f, head_1=%.2f, duration=%.2f",
                head_0, head_1, duration);

    head_pose_client_->async_send_goal(goal, options);
  }

  // =========================
  // ROS interfaces
  // =========================

  rclcpp::Subscription<inha_interfaces::action::Approach_FeedbackMessage>::SharedPtr
      feedback_sub_;
  rclcpp_action::Client<SetHeadPose>::SharedPtr head_pose_client_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FeedbackMonitorNode>());
  rclcpp::shutdown();
  return 0;
}