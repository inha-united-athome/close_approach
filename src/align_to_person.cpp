#include "close_approach/align_to_person.hpp"

AlignToPersonNode::AlignToPersonNode()
    : Node("align_to_person_node"),
      qos_best_effort_(rclcpp::QoS(rclcpp::KeepLast(10)).best_effort()),
      qos_reliable_(rclcpp::QoS(rclcpp::KeepLast(10)).reliable()) {
  this->align_to_person_server_ =
      rclcpp_action::create_server<AlignToPersonAction>(
          this, "align_to_person",
          std::bind(&AlignToPersonNode::handle_goal, this, std::placeholders::_1,
                    std::placeholders::_2),
          std::bind(&AlignToPersonNode::handle_cancel, this, std::placeholders::_1),
          std::bind(&AlignToPersonNode::handle_accepted, this,
                    std::placeholders::_1));

  camera_info_subscriber_ = this->create_subscription<CameraInfoMsg>(
      "/camera/camera_head/color/camera_info", qos_best_effort_,
      std::bind(&AlignToPersonNode::cameraInfoCallback, this,
                std::placeholders::_1));

  detection_subscriber_ = this->create_subscription<Detection2DMsg>(
      "/face/bbox/nearest", qos_reliable_,
      std::bind(&AlignToPersonNode::detection_callback, this,
                std::placeholders::_1));

  cmd_vel_publisher_ =
      this->create_publisher<TwistMsg>("/cmd_vel", qos_reliable_);

  RCLCPP_INFO(this->get_logger(), "AlignToPersonNode has been initialized");
}

rclcpp_action::GoalResponse AlignToPersonNode::handle_goal(
    const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const AlignToPersonAction::Goal> goal) {
  (void)uuid;
  (void)goal;
  RCLCPP_INFO(this->get_logger(), "Received goal request");
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse AlignToPersonNode::handle_cancel(
    const std::shared_ptr<GoalHandleAlignToPerson> goal_handle) {
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "Received cancel request");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void AlignToPersonNode::handle_accepted(
    const std::shared_ptr<GoalHandleAlignToPerson> goal_handle) {
  RCLCPP_INFO(this->get_logger(), "Goal accepted");
  std::thread(std::bind(&AlignToPersonNode::execute, this, goal_handle))
      .detach();
}

void AlignToPersonNode::execute(
    const std::shared_ptr<GoalHandleAlignToPerson> goal_handle) {
  auto feedback = std::make_shared<Feedback>();
  auto result = std::make_shared<Result>();
  if (!algorithm_start_flag) {
    algorithm_start_flag = true;
    startAlgorithm();
    RCLCPP_INFO(this->get_logger(), "Algorithm started");
  }

  rclcpp::Rate loop_rate(10);

  while (rclcpp::ok()) {
    if (goal_handle->is_canceling()) {
      RCLCPP_INFO(this->get_logger(), "Approach Person goal canceled");
      goal_handle->abort(result);
      stopAlgorithm();
      return;
    }

    if (base_control_success) {
      RCLCPP_INFO(this->get_logger(), "Base control succeeded");
      current_state_ = State::IDLE;
      stopAlgorithm();
      result->success = true;
      result->success_message = "Align Success!!";
      goal_handle->succeed(result);
      return;
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Executing...");

    feedback->error_x = error_x;
    goal_handle->publish_feedback(feedback);
    loop_rate.sleep();
  }
}

void AlignToPersonNode::startAlgorithm() {
  current_state_ = State::BASE_CONTROL;
  base_control_success = false;
  algorithm_start_flag = true;
  error_x = 0.0F;
  error_x_prev = 0.0F;
  error_x_sum = 0.0F;
}

void AlignToPersonNode::stopAlgorithm() {
  current_state_ = State::IDLE;
  base_control_success = false;
  algorithm_start_flag = false;
  error_x = 0.0F;
  error_x_prev = 0.0F;
  error_x_sum = 0.0F;
}

void AlignToPersonNode::cameraInfoCallback(
    const CameraInfoMsg::SharedPtr msg) {
  if (algorithm_start_flag == false) {
    return;
  }
  if (received_camera_info_) {
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Received CameraInfo message");
  fx_ = msg->k[0];
  fy_ = msg->k[4];
  cx_ = msg->k[2];
  cy_ = msg->k[5];
  reference_x = msg->width / 2.0F;
  received_camera_info_ = true;
}

void AlignToPersonNode::detection_callback(
    const Detection2DMsg::SharedPtr msg) {
  if (algorithm_start_flag == false) {
    return;
  }
  if (!received_camera_info_) {
    return;
  }
  Bbox bbox_;

  bbox_.center_x_ = msg->bbox.center.position.x;
  bbox_.center_y_ = msg->bbox.center.position.y;
  bbox_.width_ = msg->bbox.size_x;
  bbox_.height_ = msg->bbox.size_y;

  process_base(bbox_);
}

void AlignToPersonNode::process_base(const Bbox bbox) {
  geometry_msgs::msg::Twist twist;

  error_x = bbox.center_x_ - reference_x;
  // error가 음수일 경우, 로봇은 왼쪽으로 회전해야함
  // error가 양수일 경우, 로봇은 오른쪽으로 회전해야함
  if (std::abs(error_x) < tol_x_) {
    base_control_success = true;
    return;
  }
  error_x_sum += error_x;

  float angular_z = b_kp_theta_ * error_x + b_ki_theta_ * error_x_sum +
                    b_kd_theta_ * (error_x - error_x_prev);

  error_x_prev = error_x;

  twist.linear.x = 0.0F;
  twist.angular.z = -angular_z;
  cmd_vel_publisher_->publish(twist);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AlignToPersonNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
