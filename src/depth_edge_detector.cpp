#include "close_approach/msg/approach_error.hpp"
#include "trt_depth_estimator.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <inha_interfaces/srv/set_enable.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/bool.hpp>

namespace {

float normalizedAngleDeg(const cv::Vec4i &line) {
  float angle = std::atan2(static_cast<float>(line[3] - line[1]),
                           static_cast<float>(line[2] - line[0])) *
                180.0f / static_cast<float>(CV_PI);
  if (angle > 90.0f) angle -= 180.0f;
  if (angle < -90.0f) angle += 180.0f;
  return angle;
}

cv::Mat normalizeProximity(const cv::Mat &relative_depth) {
  double min_value = 0.0;
  double max_value = 0.0;
  cv::minMaxLoc(relative_depth, &min_value, &max_value);
  cv::Mat proximity;
  if (max_value - min_value < 1e-6) {
    proximity = cv::Mat::zeros(relative_depth.size(), CV_32F);
  } else {
    relative_depth.convertTo(proximity, CV_32F,
                             1.0 / (max_value - min_value),
                             -min_value / (max_value - min_value));
  }
  cv::max(proximity, 0.0, proximity);
  cv::min(proximity, 1.0, proximity);
  return proximity;
}

float lineMeanProximity(const cv::Mat &proximity, const cv::Vec4i &line) {
  cv::LineIterator it(proximity, {line[0], line[1]}, {line[2], line[3]}, 8);
  float sum = 0.0f;
  for (int i = 0; i < it.count; ++i, ++it) {
    sum += proximity.at<float>(it.pos());
  }
  return it.count > 0 ? sum / static_cast<float>(it.count) : 0.0f;
}

}  // namespace

class DepthEdgeDetector final : public rclcpp::Node {
public:
  using ApproachError = close_approach::msg::ApproachError;

  DepthEdgeDetector() : Node("depth_edge_detector") {
    declare_parameter<std::string>("img_topic",
        "/camera/camera_head/color/image_raw/compressed");
    declare_parameter<std::string>("depth_engine", "");
    declare_parameter<int>("depth_input_width", 518);
    declare_parameter<int>("depth_input_height", 518);
    declare_parameter<double>("depth_threshold_start", 0.8);
    declare_parameter<double>("depth_threshold_min", 0.5);
    declare_parameter<double>("depth_threshold_step", 0.05);
    declare_parameter<double>("depth_weight_gamma", 2.0);
    declare_parameter<int>("min_candidate_lines", 1);
    declare_parameter<bool>("depth_fallback_full", true);
    declare_parameter<int>("canny_low", 50);
    declare_parameter<int>("canny_high", 150);
    declare_parameter<int>("hough_thresh", 40);
    declare_parameter<int>("min_length", 60);
    declare_parameter<int>("max_gap", 15);
    declare_parameter<int>("roi_top_pct", 50);
    declare_parameter<int>("roi_bot_pct", 100);
    declare_parameter<int>("roi_left_pct", 20);
    declare_parameter<int>("roi_right_pct", 80);
    declare_parameter<double>("max_abs_yaw_deg", 30.0);
    declare_parameter<std::string>("enable_service_name",
                                     "/approach/edge_detector/set_enable");
    declare_parameter<bool>("start_enabled", false);
    declare_parameter<bool>("debug_log", true);
    declare_parameter<bool>("debug_image", true);

    get_parameter("img_topic", img_topic_);
    get_parameter("depth_engine", depth_engine_);
    get_parameter("depth_input_width", depth_input_width_);
    get_parameter("depth_input_height", depth_input_height_);
    get_parameter("depth_threshold_start", depth_threshold_start_);
    get_parameter("depth_threshold_min", depth_threshold_min_);
    get_parameter("depth_threshold_step", depth_threshold_step_);
    get_parameter("depth_weight_gamma", depth_weight_gamma_);
    get_parameter("min_candidate_lines", min_candidate_lines_);
    get_parameter("depth_fallback_full", depth_fallback_full_);
    get_parameter("canny_low", canny_low_);
    get_parameter("canny_high", canny_high_);
    get_parameter("hough_thresh", hough_thresh_);
    get_parameter("min_length", min_length_);
    get_parameter("max_gap", max_gap_);
    get_parameter("roi_top_pct", roi_top_pct_);
    get_parameter("roi_bot_pct", roi_bot_pct_);
    get_parameter("roi_left_pct", roi_left_pct_);
    get_parameter("roi_right_pct", roi_right_pct_);
    get_parameter("max_abs_yaw_deg", max_abs_yaw_deg_);
    get_parameter("enable_service_name", enable_service_name_);
    get_parameter("start_enabled", start_enabled_);
    get_parameter("debug_log", debug_log_);
    get_parameter("debug_image", debug_image_);

    depth_threshold_start_ = std::clamp(depth_threshold_start_, 0.0, 1.0);
    depth_threshold_min_ = std::clamp(depth_threshold_min_, 0.0,
                                      depth_threshold_start_);
    depth_threshold_step_ = std::clamp(depth_threshold_step_, 0.01, 1.0);
    depth_weight_gamma_ = std::max(0.0, depth_weight_gamma_);
    max_abs_yaw_deg_ = std::clamp(max_abs_yaw_deg_, 0.0, 90.0);
    min_candidate_lines_ = std::max(1, min_candidate_lines_);
    enabled_.store(start_enabled_, std::memory_order_release);

    depth_estimator_ = std::make_unique<TrtDepthEstimator>(
        depth_engine_, depth_input_width_, depth_input_height_);
    if (!depth_estimator_->ready()) {
      RCLCPP_ERROR(get_logger(), "Depth engine unavailable: %s",
                   depth_estimator_->status().c_str());
    }

    const auto qos_be = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    const auto qos_rel = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    img_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        img_topic_, qos_be,
        std::bind(&DepthEdgeDetector::imgCallback, this, std::placeholders::_1));
    active_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/approach/active", qos_rel,
        std::bind(&DepthEdgeDetector::activeCallback, this,
                  std::placeholders::_1));
    error_pub_ = create_publisher<ApproachError>("/approach/edge_error", qos_rel);
    debug_img_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        "/approach/edge_debug/compressed", qos_be);
    enable_srv_ = create_service<inha_interfaces::srv::SetEnable>(
        enable_service_name_,
        [this](const std::shared_ptr<inha_interfaces::srv::SetEnable::Request> req,
               std::shared_ptr<inha_interfaces::srv::SetEnable::Response> resp) {
          const bool engine_ready = depth_estimator_ && depth_estimator_->ready();
          enabled_.store(req->enable && engine_ready, std::memory_order_release);
          resetTheta();
          resp->success = !req->enable || engine_ready;
          resp->message = !req->enable
              ? "depth edge detector disabled"
              : (engine_ready ? "depth edge detector enabled"
                              : "depth edge detector engine unavailable");
          RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
        });

    RCLCPP_INFO(get_logger(),
                "DepthEdgeDetector ready=%d engine=%s threshold=%.2f..%.2f step=%.2f",
                depth_estimator_->ready(), depth_engine_.c_str(),
                depth_threshold_start_, depth_threshold_min_,
                depth_threshold_step_);
  }

private:
  struct Detection {
    cv::Mat filtered_edges;
    std::vector<cv::Vec4i> lines;
    double threshold = 0.0;
    int accepted = 0;
  };

  Detection detectAdaptive(const cv::Mat &canny, const cv::Mat &proximity) const {
    Detection result;
    double current = depth_threshold_start_;
    while (true) {
      cv::Mat near_mask;
      cv::compare(proximity, current, near_mask, cv::CMP_GE);
      cv::bitwise_and(canny, near_mask, result.filtered_edges);
      result.lines.clear();
      cv::HoughLinesP(result.filtered_edges, result.lines, 1, CV_PI / 180.0,
                      hough_thresh_, min_length_, max_gap_);
      result.accepted = 0;
      for (const auto &line : result.lines) {
        if (std::abs(normalizedAngleDeg(line)) < max_abs_yaw_deg_) {
          ++result.accepted;
        }
      }
      result.threshold = current;
      if (result.accepted >= min_candidate_lines_ ||
          current <= depth_threshold_min_ + 1e-9) {
        break;
      }
      current = std::max(depth_threshold_min_,
                         current - depth_threshold_step_);
    }

    // Last resort: even at the minimum proximity threshold nothing usable was
    // found. Drop the hard depth gate and run on the full edge map. The caller
    // still weights each line by proximity^gamma, so far/background lines stay
    // low-weight — we just stop refusing to produce any estimate.
    if (depth_fallback_full_ && result.accepted < min_candidate_lines_) {
      result.filtered_edges = canny;
      result.lines.clear();
      cv::HoughLinesP(result.filtered_edges, result.lines, 1, CV_PI / 180.0,
                      hough_thresh_, min_length_, max_gap_);
      result.accepted = 0;
      for (const auto &line : result.lines) {
        if (std::abs(normalizedAngleDeg(line)) < max_abs_yaw_deg_) {
          ++result.accepted;
        }
      }
      result.threshold = 0.0;  // 0 = full image (no depth gate); shown in debug
    }
    return result;
  }

  void imgCallback(
      const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {
    if (!approach_active_.load(std::memory_order_acquire) ||
        !enabled_.load(std::memory_order_acquire) ||
        !depth_estimator_ || !depth_estimator_->ready()) {
      return;
    }

    const cv::Mat frame = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (frame.empty()) return;

    const int width = frame.cols;
    const int height = frame.rows;
    const int x1 = static_cast<int>(
        std::clamp(roi_left_pct_ / 100.0, 0.0, 0.98) * width);
    const int x2 = static_cast<int>(
        std::clamp(roi_right_pct_ / 100.0,
                   (roi_left_pct_ + 1) / 100.0, 1.0) * width);
    const int y1 = static_cast<int>(
        std::clamp(roi_top_pct_ / 100.0, 0.0, 0.98) * height);
    const int y2 = static_cast<int>(
        std::clamp(roi_bot_pct_ / 100.0,
                   (roi_top_pct_ + 1) / 100.0, 1.0) * height);
    const cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
    const cv::Mat roi = frame(roi_rect);

    cv::Mat relative_depth;
    double inference_ms = 0.0;
    if (!depth_estimator_->infer(roi, relative_depth, inference_ms)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "Depth inference failed: %s",
                            depth_estimator_->status().c_str());
      return;
    }
    const cv::Mat proximity = normalizeProximity(relative_depth);

    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, {5, 5}, 0.0);
    cv::Mat canny;
    cv::Canny(gray, canny, canny_low_, canny_high_);
    Detection detection = detectAdaptive(canny, proximity);

    float weight_sum = 0.0f;
    float weighted_angle = 0.0f;
    float weighted_y = 0.0f;
    cv::Mat debug_frame;
    if (debug_image_) {
      debug_frame = frame.clone();
      cv::rectangle(debug_frame, roi_rect, {255, 180, 0}, 2);
    }

    for (const auto &line : detection.lines) {
      const float angle = normalizedAngleDeg(line);
      const bool accepted = std::abs(angle) < max_abs_yaw_deg_;
      const float dx = static_cast<float>(line[2] - line[0]);
      const float dy = static_cast<float>(line[3] - line[1]);
      const float length = std::hypot(dx, dy);
      const float mean_proximity = lineMeanProximity(proximity, line);
      const float depth_weight = std::pow(
          std::clamp(mean_proximity, 0.0f, 1.0f),
          static_cast<float>(depth_weight_gamma_));
      const float weight = length * depth_weight;

      if (debug_image_) {
        const cv::Point p1(line[0] + x1, line[1] + y1);
        const cv::Point p2(line[2] + x1, line[3] + y1);
        const cv::Scalar color = accepted ? cv::Scalar(0, 220, 255)
                                          : cv::Scalar(100, 100, 100);
        cv::line(debug_frame, p1, p2, color, accepted ? 2 : 1, cv::LINE_AA);
      }
      if (!accepted || weight <= 0.0f) continue;

      weighted_angle += angle * weight;
      weighted_y += (0.5f * (line[1] + line[3]) + y1) * weight;
      weight_sum += weight;
    }

    ApproachError error;
    error.header.stamp = msg->header.stamp;
    bool held_theta = false;
    float mean_y = 0.0f;
    if (weight_sum > 0.0f) {
      const float yaw_deg = weighted_angle / weight_sum;
      mean_y = weighted_y / weight_sum;
      last_theta_rad_ = yaw_deg * static_cast<float>(CV_PI) / 180.0f;
      theta_initialized_ = true;
    } else {
      held_theta = theta_initialized_;
    }
    // valid는 "이번 프레임에 실제로 측정했는가"만 의미한다. held(직전 값 유지)는
    // valid=false → 매니저가 stale로 판단해 회전을 멈출 수 있게 한다.
    error.valid = (weight_sum > 0.0f);
    error.theta_error = last_theta_rad_;
    error.mean_y_px = mean_y;
    error.x_error = 0.0f;
    error.y_error = 0.0f;
    error.initial_dist_m = 0.0f;
    error_pub_->publish(error);

    if (debug_log_) {
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 300,
          "depth_edge yaw=%.2fdeg valid=%d held=%d near_thr=%.2f lines=%d/%zu infer=%.1fms",
          error.theta_error * 180.0f / static_cast<float>(CV_PI), error.valid,
          held_theta, detection.threshold, detection.accepted,
          detection.lines.size(), inference_ms);
    }

    if (debug_image_ && debug_img_pub_->get_subscription_count() > 0) {
      drawDebug(debug_frame, proximity, detection, roi_rect, error, mean_y,
                inference_ms);
      std::vector<uchar> encoded;
      if (cv::imencode(".jpg", debug_frame, encoded)) {
        sensor_msgs::msg::CompressedImage output;
        output.header = msg->header;
        output.format = "jpeg";
        output.data.assign(encoded.begin(), encoded.end());
        debug_img_pub_->publish(output);
      }
    }
  }

  void drawDebug(cv::Mat &frame, const cv::Mat &proximity,
                 const Detection &detection, const cv::Rect &roi,
                 const ApproachError &error, float mean_y,
                 double inference_ms) const {
    if (error.valid && detection.accepted > 0) {
      const float angle = error.theta_error;
      const int cx = roi.x + roi.width / 2;
      const int cy = static_cast<int>(mean_y);
      const int arm = static_cast<int>(roi.width * 0.45);
      cv::Point p1(cx - static_cast<int>(std::cos(angle) * arm),
                   cy - static_cast<int>(std::sin(angle) * arm));
      cv::Point p2(cx + static_cast<int>(std::cos(angle) * arm),
                   cy + static_cast<int>(std::sin(angle) * arm));
      if (cv::clipLine(roi, p1, p2)) {
        cv::line(frame, p1, p2, {255, 0, 255}, 4, cv::LINE_AA);
      }
    }

    cv::Mat depth_u8;
    proximity.convertTo(depth_u8, CV_8U, 255.0);
    cv::Mat depth_color;
    cv::applyColorMap(depth_u8, depth_color, cv::COLORMAP_INFERNO);
    const int preview_w = std::min(240, frame.cols / 3);
    const int preview_h = std::max(1,
        static_cast<int>(depth_color.rows * preview_w /
                         static_cast<double>(depth_color.cols)));
    cv::resize(depth_color, depth_color, {preview_w, preview_h});
    depth_color.copyTo(frame(cv::Rect(frame.cols - preview_w, 0,
                                      preview_w, preview_h)));

    const std::string label = cv::format(
        "yaw %+.2f deg | near >= %.2f | lines %d/%zu | TRT %.1f ms",
        error.theta_error * 180.0f / static_cast<float>(CV_PI),
        detection.threshold, detection.accepted, detection.lines.size(),
        inference_ms);
    cv::putText(frame, label, {15, 35}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
                {0, 255, 255}, 2, cv::LINE_AA);
  }

  void activeCallback(const std_msgs::msg::Bool::SharedPtr msg) {
    const bool was_active = approach_active_.exchange(
        msg->data, std::memory_order_acq_rel);
    if (was_active != msg->data) {
      resetTheta();
      RCLCPP_INFO(get_logger(), "Approach %s: depth inference %s",
                  msg->data ? "active" : "inactive",
                  msg->data ? "armed" : "stopped");
    }
  }

  void resetTheta() {
    last_theta_rad_ = 0.0f;
    theta_initialized_ = false;
  }

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr img_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr active_sub_;
  rclcpp::Publisher<ApproachError>::SharedPtr error_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr debug_img_pub_;
  rclcpp::Service<inha_interfaces::srv::SetEnable>::SharedPtr enable_srv_;

  std::string img_topic_;
  std::string depth_engine_;
  std::string enable_service_name_;
  int depth_input_width_ = 518;
  int depth_input_height_ = 518;
  double depth_threshold_start_ = 0.8;
  double depth_threshold_min_ = 0.5;
  double depth_threshold_step_ = 0.05;
  double depth_weight_gamma_ = 2.0;
  int min_candidate_lines_ = 1;
  bool depth_fallback_full_ = true;
  int canny_low_ = 50;
  int canny_high_ = 150;
  int hough_thresh_ = 40;
  int min_length_ = 60;
  int max_gap_ = 15;
  int roi_top_pct_ = 50;
  int roi_bot_pct_ = 100;
  int roi_left_pct_ = 20;
  int roi_right_pct_ = 80;
  double max_abs_yaw_deg_ = 30.0;
  bool start_enabled_ = false;
  bool debug_log_ = true;
  bool debug_image_ = true;
  std::atomic<bool> approach_active_{false};
  std::atomic<bool> enabled_{false};
  std::unique_ptr<TrtDepthEstimator> depth_estimator_;
  float last_theta_rad_ = 0.0f;
  bool theta_initialized_ = false;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DepthEdgeDetector>());
  rclcpp::shutdown();
  return 0;
}
