#pragma once

#include <Eigen/Dense>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <memory>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/bool.hpp>
#include <string>
#include <thread>
#include <inha_interfaces/action/align_to_person.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

enum class State { IDLE, BASE_CONTROL };

struct Bbox {
  float center_x_;
  float center_y_;
  float width_;
  float height_;
};

class AlignToPersonNode : public rclcpp::Node {
public:
  explicit AlignToPersonNode();

private:
  using CameraInfoMsg = sensor_msgs::msg::CameraInfo;
  using TwistMsg = geometry_msgs::msg::Twist;
  using Detection2DArrayMsg = vision_msgs::msg::Detection2DArray;
  using Detection2DMsg = vision_msgs::msg::Detection2D;
  using AlignToPersonAction = inha_interfaces::action::AlignToPerson;
  using GoalHandleAlignToPerson =
      rclcpp_action::ServerGoalHandle<AlignToPersonAction>;
  using Feedback = AlignToPersonAction::Feedback;
  using Result = AlignToPersonAction::Result;

  rclcpp_action::Server<AlignToPersonAction>::SharedPtr
      align_to_person_server_;

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const AlignToPersonAction::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleAlignToPerson> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandleAlignToPerson> goal_handle);
  void execute(const std::shared_ptr<GoalHandleAlignToPerson> goal_handle);

  rclcpp::Subscription<Detection2DMsg>::SharedPtr detection_subscriber_;
  rclcpp::Subscription<CameraInfoMsg>::SharedPtr camera_info_subscriber_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;

  void detection_callback(const Detection2DMsg::SharedPtr msg);
  void cameraInfoCallback(const CameraInfoMsg::SharedPtr msg);
  void process_base(const Bbox bbox);

  void stopAlgorithm();
  void startAlgorithm();

  rclcpp::QoS qos_best_effort_;
  rclcpp::QoS qos_reliable_;

  std::string detection_vision_msg_name_;
  std::string info_topic_name_;

  State current_state_ = State::IDLE;
  Bbox bbox_;
  std::string failure_message_;
  std::atomic<bool> algorithm_start_flag = false;
  std::atomic<bool> base_control_success = false;

  // camera info variables
  bool received_camera_info_ = false;
  float fx_ = 0.0F;
  float fy_ = 0.0F;
  float cx_ = 0.0F;
  float cy_ = 0.0F;

  // pid control variables b : base, t : torso
  float b_kp_theta_ = 0.001F;
  float b_ki_theta_ = 0.0F;
  float b_kd_theta_ = 0.0F;

  // 임계값
  float tol_x_ = 20.0F;

  // 이미지 좌표계 기준 x
  float reference_x = 320.0F;
  float error_x = 0.0F;
  float error_x_prev = 0.0F;
  float error_x_sum = 0.0F;
};
