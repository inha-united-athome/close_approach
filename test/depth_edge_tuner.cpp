// Live tuning helper for depth_edge_detector parameters.
//
// Always processes incoming images (no active/enable gating) and exposes every
// detection parameter as a live ROS parameter, so they can be tuned like
// OpenCV trackbars without rebuilding:
//
//   ros2 param set /depth_edge_tuner canny_low 40
//   ros2 param set /depth_edge_tuner depth_threshold_start 0.7
//   ros2 param set /depth_edge_tuner depth_weight_gamma 1.5
//   ...watch /depth_edge_tuner/debug/compressed in rqt_image_view...
//   ros2 param dump /depth_edge_tuner    # then copy into depth_edge_detector.yaml
//
// rqt_reconfigure gives sliders for the scalar params. If no TensorRT depth
// engine is available, proximity is treated as all-near (1.0), so canny/Hough/
// ROI can still be tuned (depth gating just becomes a no-op).

#include "trt_depth_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

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
  double min_value = 0.0, max_value = 0.0;
  cv::minMaxLoc(relative_depth, &min_value, &max_value);
  cv::Mat proximity;
  if (max_value - min_value < 1e-6) {
    proximity = cv::Mat::zeros(relative_depth.size(), CV_32F);
  } else {
    relative_depth.convertTo(proximity, CV_32F, 1.0 / (max_value - min_value),
                             -min_value / (max_value - min_value));
  }
  cv::max(proximity, 0.0, proximity);
  cv::min(proximity, 1.0, proximity);
  return proximity;
}

float lineMeanProximity(const cv::Mat &proximity, const cv::Vec4i &line) {
  cv::LineIterator it(proximity, {line[0], line[1]}, {line[2], line[3]}, 8);
  float sum = 0.0f;
  for (int i = 0; i < it.count; ++i, ++it) sum += proximity.at<float>(it.pos());
  return it.count > 0 ? sum / static_cast<float>(it.count) : 0.0f;
}

}  // namespace

class DepthEdgeTuner : public rclcpp::Node {
public:
  DepthEdgeTuner() : Node("depth_edge_tuner") {
    img_topic_ = declare_parameter<std::string>(
        "img_topic", "/camera/camera_head/color/image_raw/compressed");
    depth_engine_ = declare_parameter<std::string>("depth_engine", "");
    depth_input_width_  = declare_parameter<int>("depth_input_width", 518);
    depth_input_height_ = declare_parameter<int>("depth_input_height", 518);
    depth_threshold_start_ = declare_parameter<double>("depth_threshold_start", 0.8);
    depth_threshold_min_   = declare_parameter<double>("depth_threshold_min", 0.5);
    depth_threshold_step_  = declare_parameter<double>("depth_threshold_step", 0.05);
    depth_weight_gamma_    = declare_parameter<double>("depth_weight_gamma", 2.0);
    min_candidate_lines_   = declare_parameter<int>("min_candidate_lines", 1);
    depth_fallback_full_   = declare_parameter<bool>("depth_fallback_full", true);
    canny_low_   = declare_parameter<int>("canny_low", 50);
    canny_high_  = declare_parameter<int>("canny_high", 150);
    hough_thresh_ = declare_parameter<int>("hough_thresh", 40);
    min_length_  = declare_parameter<int>("min_length", 60);
    max_gap_     = declare_parameter<int>("max_gap", 15);
    roi_top_pct_   = declare_parameter<int>("roi_top_pct", 50);
    roi_bot_pct_   = declare_parameter<int>("roi_bot_pct", 100);
    roi_left_pct_  = declare_parameter<int>("roi_left_pct", 20);
    roi_right_pct_ = declare_parameter<int>("roi_right_pct", 80);
    max_abs_yaw_deg_ = declare_parameter<double>("max_abs_yaw_deg", 30.0);
    applyClamps();

    depth_estimator_ = std::make_unique<TrtDepthEstimator>(
        depth_engine_, depth_input_width_, depth_input_height_);
    if (!depth_estimator_->ready()) {
      RCLCPP_WARN(get_logger(),
                  "Depth engine unavailable (%s): proximity=1.0, depth gating "
                  "disabled. canny/Hough/ROI tuning still works.",
                  depth_estimator_->status().c_str());
    }

    const auto qos_be = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    img_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        img_topic_, qos_be,
        std::bind(&DepthEdgeTuner::imgCb, this, std::placeholders::_1));
    debug_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        "/depth_edge_tuner/debug/compressed", qos_be);
    param_cb_ = add_on_set_parameters_callback(
        std::bind(&DepthEdgeTuner::onSetParam, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "depth_edge_tuner ready. img=%s engine_ready=%d",
                img_topic_.c_str(), depth_estimator_->ready());
  }

private:
  struct Detection {
    std::vector<cv::Vec4i> lines;
    double threshold = 0.0;
    int accepted = 0;
  };

  void applyClamps() {
    depth_threshold_start_ = std::clamp(depth_threshold_start_, 0.0, 1.0);
    depth_threshold_min_ = std::clamp(depth_threshold_min_, 0.0, depth_threshold_start_);
    depth_threshold_step_ = std::clamp(depth_threshold_step_, 0.01, 1.0);
    depth_weight_gamma_ = std::max(0.0, depth_weight_gamma_);
    min_candidate_lines_ = std::max(1, min_candidate_lines_);
    max_abs_yaw_deg_ = std::clamp(max_abs_yaw_deg_, 0.0, 90.0);
  }

  rcl_interfaces::msg::SetParametersResult onSetParam(
      const std::vector<rclcpp::Parameter> &params) {
    for (const auto &p : params) {
      const std::string &n = p.get_name();
      if (n == "depth_threshold_start") depth_threshold_start_ = p.as_double();
      else if (n == "depth_threshold_min") depth_threshold_min_ = p.as_double();
      else if (n == "depth_threshold_step") depth_threshold_step_ = p.as_double();
      else if (n == "depth_weight_gamma") depth_weight_gamma_ = p.as_double();
      else if (n == "min_candidate_lines") min_candidate_lines_ = static_cast<int>(p.as_int());
      else if (n == "depth_fallback_full") depth_fallback_full_ = p.as_bool();
      else if (n == "canny_low") canny_low_ = static_cast<int>(p.as_int());
      else if (n == "canny_high") canny_high_ = static_cast<int>(p.as_int());
      else if (n == "hough_thresh") hough_thresh_ = static_cast<int>(p.as_int());
      else if (n == "min_length") min_length_ = static_cast<int>(p.as_int());
      else if (n == "max_gap") max_gap_ = static_cast<int>(p.as_int());
      else if (n == "roi_top_pct") roi_top_pct_ = static_cast<int>(p.as_int());
      else if (n == "roi_bot_pct") roi_bot_pct_ = static_cast<int>(p.as_int());
      else if (n == "roi_left_pct") roi_left_pct_ = static_cast<int>(p.as_int());
      else if (n == "roi_right_pct") roi_right_pct_ = static_cast<int>(p.as_int());
      else if (n == "max_abs_yaw_deg") max_abs_yaw_deg_ = p.as_double();
    }
    applyClamps();
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    return result;
  }

  Detection detectAdaptive(const cv::Mat &canny, const cv::Mat &proximity) const {
    Detection result;
    double current = depth_threshold_start_;
    cv::Mat filtered;
    while (true) {
      cv::Mat near_mask;
      cv::compare(proximity, current, near_mask, cv::CMP_GE);
      cv::bitwise_and(canny, near_mask, filtered);
      result.lines.clear();
      cv::HoughLinesP(filtered, result.lines, 1, CV_PI / 180.0, hough_thresh_,
                      min_length_, max_gap_);
      result.accepted = 0;
      for (const auto &l : result.lines)
        if (std::abs(normalizedAngleDeg(l)) < max_abs_yaw_deg_) ++result.accepted;
      result.threshold = current;
      if (result.accepted >= min_candidate_lines_ ||
          current <= depth_threshold_min_ + 1e-9)
        break;
      current = std::max(depth_threshold_min_, current - depth_threshold_step_);
    }
    if (depth_fallback_full_ && result.accepted < min_candidate_lines_) {
      result.lines.clear();
      cv::HoughLinesP(canny, result.lines, 1, CV_PI / 180.0, hough_thresh_,
                      min_length_, max_gap_);
      result.accepted = 0;
      for (const auto &l : result.lines)
        if (std::abs(normalizedAngleDeg(l)) < max_abs_yaw_deg_) ++result.accepted;
      result.threshold = 0.0;
    }
    return result;
  }

  void imgCb(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {
    const cv::Mat frame = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (frame.empty()) return;
    const int W = frame.cols, H = frame.rows;
    const int x1 = static_cast<int>(std::clamp(roi_left_pct_ / 100.0, 0.0, 0.98) * W);
    const int x2 = static_cast<int>(std::clamp(roi_right_pct_ / 100.0,
                                               (roi_left_pct_ + 1) / 100.0, 1.0) * W);
    const int y1 = static_cast<int>(std::clamp(roi_top_pct_ / 100.0, 0.0, 0.98) * H);
    const int y2 = static_cast<int>(std::clamp(roi_bot_pct_ / 100.0,
                                               (roi_top_pct_ + 1) / 100.0, 1.0) * H);
    const cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
    const cv::Mat roi = frame(roi_rect);

    cv::Mat proximity;
    double inference_ms = 0.0;
    if (depth_estimator_ && depth_estimator_->ready()) {
      cv::Mat relative_depth;
      if (depth_estimator_->infer(roi, relative_depth, inference_ms)) {
        proximity = normalizeProximity(relative_depth);
      }
    }
    if (proximity.empty()) {
      // No engine / failed inference → treat everything as near (gate off).
      proximity = cv::Mat::ones(roi.size(), CV_32F);
    }

    cv::Mat gray, canny;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, {5, 5}, 0.0);
    cv::Canny(gray, canny, canny_low_, canny_high_);
    const Detection det = detectAdaptive(canny, proximity);

    cv::Mat dbg = frame.clone();
    cv::rectangle(dbg, roi_rect, {255, 180, 0}, 2);
    float w_sum = 0.0f, wa_sum = 0.0f, wy_sum = 0.0f;
    for (const auto &l : det.lines) {
      const float angle = normalizedAngleDeg(l);
      const bool ok = std::abs(angle) < max_abs_yaw_deg_;
      const cv::Point p1(l[0] + x1, l[1] + y1), p2(l[2] + x1, l[3] + y1);
      cv::line(dbg, p1, p2, ok ? cv::Scalar(0, 220, 255) : cv::Scalar(100, 100, 100),
               ok ? 2 : 1, cv::LINE_AA);
      if (!ok) continue;
      const float len = std::hypot(static_cast<float>(l[2] - l[0]),
                                   static_cast<float>(l[3] - l[1]));
      const float w = len * std::pow(std::clamp(lineMeanProximity(proximity, l), 0.0f, 1.0f),
                                     static_cast<float>(depth_weight_gamma_));
      if (w <= 0.0f) continue;
      wa_sum += angle * w;
      wy_sum += (0.5f * (l[1] + l[3]) + y1) * w;
      w_sum += w;
    }

    float yaw_deg = 0.0f, mean_y = 0.0f;
    const bool valid = w_sum > 0.0f;
    if (valid) {
      yaw_deg = wa_sum / w_sum;
      mean_y = wy_sum / w_sum;
      const float ang = yaw_deg * static_cast<float>(CV_PI) / 180.0f;
      const int cx = roi_rect.x + roi_rect.width / 2;
      const int arm = static_cast<int>(roi_rect.width * 0.45);
      cv::Point a(cx - static_cast<int>(std::cos(ang) * arm),
                  static_cast<int>(mean_y) - static_cast<int>(std::sin(ang) * arm));
      cv::Point b(cx + static_cast<int>(std::cos(ang) * arm),
                  static_cast<int>(mean_y) + static_cast<int>(std::sin(ang) * arm));
      cv::line(dbg, a, b, {255, 0, 255}, 4, cv::LINE_AA);
    }

    // Depth preview (top-right).
    cv::Mat depth_u8, depth_color;
    proximity.convertTo(depth_u8, CV_8U, 255.0);
    cv::applyColorMap(depth_u8, depth_color, cv::COLORMAP_INFERNO);
    const int pw = std::min(240, dbg.cols / 3);
    const int ph = std::max(1, depth_color.rows * pw / std::max(1, depth_color.cols));
    cv::resize(depth_color, depth_color, {pw, ph});
    depth_color.copyTo(dbg(cv::Rect(dbg.cols - pw, 0, pw, ph)));

    const std::string label = cv::format(
        "yaw %+.2f deg valid %d | near>=%.2f | lines %d/%zu | canny %d/%d hough %d "
        "| TRT %.1fms",
        yaw_deg, valid, det.threshold, det.accepted, det.lines.size(), canny_low_,
        canny_high_, hough_thresh_, inference_ms);
    cv::putText(dbg, label, {15, 35}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                {0, 255, 255}, 2, cv::LINE_AA);

    std::vector<uchar> enc;
    if (cv::imencode(".jpg", dbg, enc)) {
      sensor_msgs::msg::CompressedImage out;
      out.header = msg->header;
      out.format = "jpeg";
      out.data.assign(enc.begin(), enc.end());
      debug_pub_->publish(out);
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                         "yaw=%.2fdeg valid=%d near>=%.2f lines=%d/%zu",
                         yaw_deg, valid, det.threshold, det.accepted,
                         det.lines.size());
  }

  std::string img_topic_, depth_engine_;
  int depth_input_width_ = 518, depth_input_height_ = 518;
  double depth_threshold_start_ = 0.8, depth_threshold_min_ = 0.5;
  double depth_threshold_step_ = 0.05, depth_weight_gamma_ = 2.0;
  int min_candidate_lines_ = 1;
  bool depth_fallback_full_ = true;
  int canny_low_ = 50, canny_high_ = 150, hough_thresh_ = 40;
  int min_length_ = 60, max_gap_ = 15;
  int roi_top_pct_ = 50, roi_bot_pct_ = 100, roi_left_pct_ = 20, roi_right_pct_ = 80;
  double max_abs_yaw_deg_ = 30.0;

  std::unique_ptr<TrtDepthEstimator> depth_estimator_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr img_sub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr debug_pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DepthEdgeTuner>());
  rclcpp::shutdown();
  return 0;
}
