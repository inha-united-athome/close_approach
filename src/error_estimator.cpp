#include "close_approach/error_estimator.hpp"
#include <rclcpp/rclcpp.hpp>

SE2Error ErrorEstimator::estimate_error(const TargetEdge &target_edge) {
  /*
  진짜 목표점 = target_center - 0.4 * normal_axis
  */
  float x_error = target_edge.target_center.dot(target_edge.normal_axis) - 0.35;
  // PCA의 엣지 벡터 부호가 좌우로 깜빡이는 현상(Flipping) 방지
  // 가장 안정적인 수직축(normal_axis)을 기준으로 무조건 왼쪽 90도 회전시킨 고정
  // 가짜 축을 하나 만듭니다.
  Eigen::Vector2f consistent_lat_axis(-target_edge.normal_axis.y(),
                                      target_edge.normal_axis.x());
  float y_error = target_edge.target_center.dot(consistent_lat_axis);
  float theta_error =
      std::atan2(target_edge.normal_axis.y(), target_edge.normal_axis.x());

  rclcpp::Clock clock;
  RCLCPP_INFO_THROTTLE(rclcpp::get_logger("error_estimator"), clock, 500,
                       "x_error: %f", x_error);
  RCLCPP_INFO_THROTTLE(rclcpp::get_logger("error_estimator"), clock, 500,
                       "y_error: %f", y_error);
  RCLCPP_INFO_THROTTLE(rclcpp::get_logger("error_estimator"), clock, 500,
                       "theta_error: %f", theta_error);

  return {x_error, y_error, theta_error};
}