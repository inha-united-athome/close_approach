#pragma once

#include <memory>
#include <string>


#include "inha_interfaces/action/align_decider.hpp"


#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

/*
AlignDeciderNode
- 액션 Goal 로 두 물체 좌표(base_nav 기준)를 받아 중점 P=(px,py)를 구한다.
- 로봇 전방 직선(x축)과 중점이 횡으로 py 만큼 어긋나 있을 때, 이를 45도
  dog-leg(측면 이동)로 맞추기 위한 (방향, 대각선 주행거리)를 계산해 Result 로
  돌려준다. 모션은 없다(순수 판단 노드).
    direction       = py>=0 ? "left" : "right"
    remain_distance = |py| / sin(turn_angle)   // 45도면 |py|*sqrt(2)
*/
class AlignDeciderNode : public rclcpp::Node {
public:
  AlignDeciderNode();

private:
  using AlignDecider = inha_interfaces::action::AlignDecider;
  using GoalHandle = rclcpp_action::ServerGoalHandle<AlignDecider>;


  rclcpp_action::Server<AlignDecider>::SharedPtr server_;

  rclcpp_action::GoalResponse
  handleGoal(const rclcpp_action::GoalUUID &uuid,
             std::shared_ptr<const AlignDecider::Goal> goal);
  rclcpp_action::CancelResponse handleCancel(std::shared_ptr<GoalHandle> gh);
  void handleAccepted(std::shared_ptr<GoalHandle> gh);
  void execute(std::shared_ptr<GoalHandle> gh);

  // map(입력) → base_nav(계산) TF
  tf2_ros::Buffer            tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string input_frame_  = "map";       // goal coords 프레임
  std::string target_frame_ = "base_nav";  // 계산 기준 프레임
  float turn_angle_deg_ = 45.0f;
  bool  debug_log_ = true;
};
