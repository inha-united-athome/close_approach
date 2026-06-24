#include "close_approach/align_decider.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

AlignDeciderNode::AlignDeciderNode()
    : Node("align_decider"),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_) {
  this->declare_parameter<std::string>("input_frame", "map");
  this->declare_parameter<std::string>("target_frame", "base_nav");
  this->declare_parameter<float>("turn_angle_deg", 45.0f);
  this->declare_parameter<bool>("debug_log", true);
  this->get_parameter("input_frame", input_frame_);
  this->get_parameter("target_frame", target_frame_);
  this->get_parameter("turn_angle_deg", turn_angle_deg_);
  this->get_parameter("debug_log", debug_log_);
  // 0/90 근처는 분모(sin)·기하가 무의미해지므로 안전 범위로 클램프.
  turn_angle_deg_ = std::clamp(turn_angle_deg_, 5.0f, 85.0f);

  server_ = rclcpp_action::create_server<AlignDecider>(
      this, "align_decision",
      std::bind(&AlignDeciderNode::handleGoal, this, std::placeholders::_1,
                std::placeholders::_2),
      std::bind(&AlignDeciderNode::handleCancel, this, std::placeholders::_1),
      std::bind(&AlignDeciderNode::handleAccepted, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(),
              "AlignDeciderNode ready (in=%s -> calc=%s, turn_angle=%.1f deg)",
              input_frame_.c_str(), target_frame_.c_str(), turn_angle_deg_);
}

rclcpp_action::GoalResponse AlignDeciderNode::handleGoal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const AlignDecider::Goal> goal) {
  if (!goal || goal->coords.empty()) {
    RCLCPP_WARN(this->get_logger(), "Rejecting goal: empty coords");
    return rclcpp_action::GoalResponse::REJECT;
  }
  for (const auto &p : goal->coords) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      RCLCPP_WARN(this->get_logger(), "Rejecting goal: non-finite coordinate");
      return rclcpp_action::GoalResponse::REJECT;
    }
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
AlignDeciderNode::handleCancel(std::shared_ptr<GoalHandle>) {
  return rclcpp_action::CancelResponse::ACCEPT;
}

void AlignDeciderNode::handleAccepted(std::shared_ptr<GoalHandle> gh) {
  // 순수 계산이라 즉시 처리. 실행 스레드 분리(서버 콜백 블로킹 방지).
  std::thread(std::bind(&AlignDeciderNode::execute, this, gh)).detach();
}

void AlignDeciderNode::execute(std::shared_ptr<GoalHandle> gh) {
  const auto goal = gh->get_goal();
  auto result = std::make_shared<AlignDecider::Result>();

  // goal coords 는 input_frame(map) 기준 → 계산 기준 target_frame(base_nav)로 변환.
  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_.lookupTransform(target_frame_, input_frame_,
                                    tf2::TimePointZero,
                                    tf2::durationFromSec(0.3));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(this->get_logger(), "TF %s <- %s failed: %s",
                 target_frame_.c_str(), input_frame_.c_str(), ex.what());
    result->success = false;
    gh->abort(result);
    return;
  }

  // 중점 P (base_nav) = 변환된 좌표 배열 전체의 평균. 횡오차는 py, 전방은 px.
  float px = 0.0f, py = 0.0f, pz = 0.0f;
  const float n = static_cast<float>(goal->coords.size());
  for (const auto &p : goal->coords) {
    geometry_msgs::msg::PointStamped in, out;
    in.header.frame_id = input_frame_;
    in.point = p;
    tf2::doTransform(in, out, tf);
    px += static_cast<float>(out.point.x);
    py += static_cast<float>(out.point.y);
    pz += static_cast<float>(out.point.z);
  }
  px /= n;
  py /= n;
  pz /= n;

  const float angle_rad = turn_angle_deg_ * static_cast<float>(M_PI) / 180.0f;
  const bool  go_left = (py >= 0.0f);

  result->midpoint.x = px;
  result->midpoint.y = py;
  result->midpoint.z = pz;
  result->direction = go_left ? "left" : "right";
  result->turn_angle_deg = go_left ? turn_angle_deg_ : -turn_angle_deg_;
  // 45도 대각선 주행으로 횡 |py| 만큼 이동: s*sin(angle)=|py| → s=|py|/sin(angle).
  result->remain_distance = std::abs(py) / std::sin(angle_rad);
  result->success = true;

  if (debug_log_) {
    RCLCPP_INFO(this->get_logger(),
                "midpoint=(%.3f, %.3f) -> turn %s %.1f deg, drive %.3f m",
                px, py, result->direction.c_str(),
                std::abs(result->turn_angle_deg), result->remain_distance);
  }

  gh->succeed(result);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AlignDeciderNode>());
  rclcpp::shutdown();
  return 0;
}
