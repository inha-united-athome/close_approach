#include "close_approach/edge_detector.hpp"

#include <cmath>
#include <opencv2/opencv.hpp>

EdgeDetector::EdgeDetector() : Node("edge_detector") {
  this->declare_parameter<std::string>("img_topic",
      "/camera/camera_head/color/image_raw/compressed");
  this->declare_parameter<int>("canny_low",     50);
  this->declare_parameter<int>("canny_high",    150);
  this->declare_parameter<int>("hough_thresh",  40);
  this->declare_parameter<int>("min_length",    60);
  this->declare_parameter<int>("max_gap",       15);
  this->declare_parameter<int>("roi_top_pct",   20);
  this->declare_parameter<int>("roi_bot_pct",   80);
  this->declare_parameter<int>("roi_left_pct",  20);
  this->declare_parameter<int>("roi_right_pct", 80);
  this->declare_parameter<bool>("debug_log",    true);
  this->declare_parameter<bool>("debug_image",  true);

  this->get_parameter("img_topic",    img_topic_);
  this->get_parameter("canny_low",    canny_low_);
  this->get_parameter("canny_high",   canny_high_);
  this->get_parameter("hough_thresh", hough_thresh_);
  this->get_parameter("min_length",   min_length_);
  this->get_parameter("max_gap",      max_gap_);
  this->get_parameter("roi_top_pct",  roi_top_pct_);
  this->get_parameter("roi_bot_pct",  roi_bot_pct_);
  this->get_parameter("roi_left_pct", roi_left_pct_);
  this->get_parameter("roi_right_pct",roi_right_pct_);
  this->get_parameter("debug_log",    debug_log_);
  this->get_parameter("debug_image",  debug_image_);

  auto qos_be  = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
  auto qos_rel = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

  img_sub_  = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      img_topic_, qos_be,
      std::bind(&EdgeDetector::imgCallback, this, std::placeholders::_1));
  active_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "/approach/active", qos_rel,
      std::bind(&EdgeDetector::activeCallback, this, std::placeholders::_1));
  error_pub_ = this->create_publisher<ApproachError>("/approach/edge_error", qos_rel);
  debug_img_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
      "/approach/edge_debug/compressed", qos_be);

  RCLCPP_INFO(this->get_logger(), "EdgeDetector ready. img_topic=%s", img_topic_.c_str());
}

void EdgeDetector::activeCallback(const std_msgs::msg::Bool::SharedPtr msg) {
  if (msg->data) {
    last_theta_rad_    = 0.0f;
    theta_initialized_ = false;
  }
}

void EdgeDetector::imgCallback(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {

  cv::Mat frame = cv::imdecode(msg->data, cv::IMREAD_COLOR);
  if (frame.empty()) return;

  const int W = frame.cols;
  const int H = frame.rows;

  // ROI in pixels
  const int ry1 = static_cast<int>(std::clamp(roi_top_pct_  / 100.0f, 0.0f, 0.98f) * H);
  const int ry2 = static_cast<int>(std::clamp(roi_bot_pct_  / 100.0f, (roi_top_pct_+1)/100.0f, 1.0f) * H);
  const int rx1 = static_cast<int>(std::clamp(roi_left_pct_ / 100.0f, 0.0f, 0.98f) * W);
  const int rx2 = static_cast<int>(std::clamp(roi_right_pct_/ 100.0f, (roi_left_pct_+1)/100.0f, 1.0f) * W);

  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, {5, 5}, 0);

  cv::Mat roi_mask = cv::Mat::zeros(gray.size(), CV_8U);
  roi_mask(cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1)) = 255;

  cv::Mat edges;
  cv::Canny(gray, edges, canny_low_, canny_high_);
  cv::bitwise_and(edges, roi_mask, edges);

  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(edges, lines, 1, CV_PI / 180,
                  hough_thresh_, min_length_, max_gap_);

  cv::Mat debug_frame;
  if (debug_image_) {
    debug_frame = frame.clone();
    cv::rectangle(debug_frame, cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1),
                  cv::Scalar(255, 180, 0), 2);
  }

  // Length-weighted angle & y accumulation for horizontal lines (|angle| < 50°)
  float w_sum = 0.0f, wa_sum = 0.0f, wy_sum = 0.0f;
  int accepted_lines = 0;
  for (const auto &l : lines) {
    const float dx    = static_cast<float>(l[2] - l[0]);
    const float dy    = static_cast<float>(l[3] - l[1]);
    float angle = std::atan2(dy, dx) * 180.0f / CV_PI;
    if (angle >  90.0f) angle -= 180.0f;
    if (angle < -90.0f) angle += 180.0f;
    const bool accepted = std::abs(angle) < 50.0f;

    if (debug_image_) {
      const cv::Scalar color =
          accepted ? cv::Scalar(0, 255, 0) : cv::Scalar(120, 120, 120);
      cv::line(debug_frame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
               color, accepted ? 3 : 1, cv::LINE_AA);
    }
    if (!accepted) continue;

    const float len = std::hypot(dx, dy);
    const float my  = (l[1] + l[3]) * 0.5f;
    wa_sum += angle * len;
    wy_sum += my    * len;
    w_sum  += len;
    ++accepted_lines;
  }

  ApproachError err;
  err.header.stamp = msg->header.stamp;

  bool held_theta = false;
  if (w_sum > 0.0f) {
    const float yaw_deg = wa_sum / w_sum;
    last_theta_rad_    = yaw_deg * static_cast<float>(CV_PI) / 180.0f;
    theta_initialized_ = true;
    err.mean_y_px      = wy_sum / w_sum;
  } else {
    err.mean_y_px = 0.0f;
    held_theta = theta_initialized_;
  }

  err.valid       = theta_initialized_;
  err.theta_error = last_theta_rad_;
  err.x_error     = 0.0f;
  err.y_error     = 0.0f;
  err.initial_dist_m = 0.0f;

  error_pub_->publish(err);

  if (debug_log_) {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 300,
        "edge_yaw theta=%.4f rad (%.2f deg) valid=%d held=%d lines=%zu accepted=%d mean_y=%.1f",
        err.theta_error, err.theta_error * 180.0f / static_cast<float>(CV_PI),
        err.valid, held_theta, lines.size(), accepted_lines, err.mean_y_px);
  }

  if (debug_image_ && debug_img_pub_->get_subscription_count() > 0) {
    const std::string label = cv::format(
        "yaw %.2f deg | %.4f rad | valid %d | held %d | lines %d/%zu",
        err.theta_error * 180.0f / static_cast<float>(CV_PI), err.theta_error,
        err.valid, held_theta, accepted_lines, lines.size());
    cv::putText(debug_frame, label, cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2,
                cv::LINE_AA);

    std::vector<uchar> encoded;
    if (cv::imencode(".jpg", debug_frame, encoded)) {
      sensor_msgs::msg::CompressedImage out;
      out.header = msg->header;
      out.format = "jpeg";
      out.data.assign(encoded.begin(), encoded.end());
      debug_img_pub_->publish(out);
    }
  }
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EdgeDetector>());
  rclcpp::shutdown();
  return 0;
}
