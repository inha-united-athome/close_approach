#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <opencv2/opencv.hpp>
#include "trt_depth_estimator.hpp"
#include "trt_semantic_segmenter.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>
#include <cmath>

// 수평 엣지 기반 Yaw 추정 테스트 노드
//
// 대상: 테이블·선반·식기세척기·냉장고 등
// 원리: 수평 에지 각도 → yaw error
//
// 트랙바: Canny / Hough / ROI (Top, Bot, Left, Right) 전부 실시간 조정
//
// 키: q / ESC → 종료

static const char *kWinLines = "Lines";
static const char *kWinEdges = "Edges";
static const char *kWinSemantic = "Semantic";
static const char *kWinDepthCompare = "Depth Compare";
static const char *kWinNearWeightedEdge = "Canny x Near Depth";
static const char *kWinDepthOnlyEdge = "Depth Only Edge";

class EdgeYawViewer : public rclcpp::Node {
public:
  EdgeYawViewer() : Node("edge_yaw_viewer") {
    this->declare_parameter<std::string>("topic",
        "/camera/camera_head/color/image_raw/compressed");
    topic_ = this->get_parameter("topic").as_string();
    this->declare_parameter<bool>("semantic_enabled", true);
    this->declare_parameter<std::string>("semantic_engine", "yolo26n-sem.engine");
    this->declare_parameter<int>("semantic_input_width", 1024);
    this->declare_parameter<int>("semantic_input_height", 1024);
    semantic_enabled_ = this->get_parameter("semantic_enabled").as_bool();
    semantic_engine_ = this->get_parameter("semantic_engine").as_string();
    const int semantic_w = this->get_parameter("semantic_input_width").as_int();
    const int semantic_h = this->get_parameter("semantic_input_height").as_int();
    this->declare_parameter<bool>("depth_enabled", true);
    this->declare_parameter<std::string>(
        "depth_engine", "depth_anything_v2_vits.engine");
    this->declare_parameter<int>("depth_input_width", 518);
    this->declare_parameter<int>("depth_input_height", 518);
    depth_enabled_ = this->get_parameter("depth_enabled").as_bool();
    depth_engine_ = this->get_parameter("depth_engine").as_string();
    const int depth_w = this->get_parameter("depth_input_width").as_int();
    const int depth_h = this->get_parameter("depth_input_height").as_int();

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        topic_, qos,
        [this](const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {
          const std::vector<uint8_t> &buf = msg->data;
          cv::Mat decoded = cv::imdecode(buf, cv::IMREAD_COLOR);
          if (decoded.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "imdecode 실패 (빈 프레임)");
            return;
          }
          std::lock_guard<std::mutex> lk(mtx_);
          frame_ = std::move(decoded);
          has_frame_ = true;
          ++frame_sequence_;
        });

    cv::namedWindow(kWinLines, cv::WINDOW_NORMAL);
    cv::namedWindow(kWinEdges, cv::WINDOW_NORMAL);
    if (semantic_enabled_) {
      cv::namedWindow(kWinSemantic, cv::WINDOW_NORMAL);
      segmenter_ = std::make_unique<TrtSemanticSegmenter>(
          semantic_engine_, semantic_w, semantic_h);
      if (segmenter_->ready()) {
        RCLCPP_INFO(this->get_logger(), "YOLO26-sem TensorRT 준비 완료: %s",
                    segmenter_->status().c_str());
      } else {
        RCLCPP_ERROR(this->get_logger(), "YOLO26-sem 비활성: %s",
                     segmenter_->status().c_str());
      }
    }
    if (depth_enabled_) {
      cv::namedWindow(kWinDepthCompare, cv::WINDOW_NORMAL);
      cv::namedWindow(kWinNearWeightedEdge, cv::WINDOW_NORMAL);
      cv::namedWindow(kWinDepthOnlyEdge, cv::WINDOW_NORMAL);
      depth_estimator_ = std::make_unique<TrtDepthEstimator>(
          depth_engine_, depth_w, depth_h);
      if (depth_estimator_->ready()) {
        RCLCPP_INFO(this->get_logger(), "Depth Anything V2 TensorRT 준비 완료: %s",
                    depth_estimator_->status().c_str());
      } else {
        RCLCPP_ERROR(this->get_logger(), "Depth Anything V2 비활성: %s",
                     depth_estimator_->status().c_str());
      }
    }
    initTrackbars();

    RCLCPP_INFO(this->get_logger(), "Edge Yaw Viewer 시작. topic=%s", topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "s: YAML 저장  q: 종료");
  }

  void saveYaml() {
    namespace fs = std::filesystem;
    const fs::path save_dir = fs::current_path() / "config";
    fs::create_directories(save_dir);
    const fs::path yaml_path = save_dir / "edge_detector.yaml";

    std::ofstream f(yaml_path);
    f << "edge_detector:\n";
    f << "  ros__parameters:\n";
    f << "    topic: \"" << topic_ << "\"\n";
    f << "    canny_low: "     << tb_.canny_low     << "\n";
    f << "    canny_high: "    << tb_.canny_high    << "\n";
    f << "    hough_thresh: "  << tb_.hough_thresh  << "\n";
    f << "    min_length: "    << tb_.min_length    << "\n";
    f << "    max_gap: "       << tb_.max_gap       << "\n";
    f << "    roi_top_pct: "   << tb_.roi_top_pct   << "\n";
    f << "    roi_bot_pct: "   << tb_.roi_bot_pct   << "\n";
    f << "    roi_left_pct: "  << tb_.roi_left_pct  << "\n";
    f << "    roi_right_pct: " << tb_.roi_right_pct << "\n";
    f << "    semantic_enabled: " << (semantic_enabled_ ? "true" : "false") << "\n";
    f << "    semantic_engine: \"" << semantic_engine_ << "\"\n";
    f << "    semantic_alpha_pct: " << tb_.semantic_alpha_pct << "\n";
    f << "    depth_enabled: " << (depth_enabled_ ? "true" : "false") << "\n";
    f << "    depth_engine: \"" << depth_engine_ << "\"\n";
    f << "    depth_edge_thresh_pct: " << tb_.depth_edge_thresh_pct << "\n";
    f << "    depth_near_floor_pct: " << tb_.depth_near_floor_pct << "\n";
    f << "    depth_near_gamma_x10: " << tb_.depth_near_gamma_x10 << "\n";
    f << "    depth_near_cutoff_pct: " << tb_.depth_near_cutoff_pct << "\n";
    f.close();

    RCLCPP_INFO(this->get_logger(), "저장 완료: %s", yaml_path.c_str());
    RCLCPP_INFO(this->get_logger(),
        "저장 완료 → edge_detector_node 에서 바로 사용 가능:\n"
        "  ros2 launch close_approach approach.launch.py");
  }

  void update() {
    cv::Mat frame;
    std::uint64_t frame_sequence = 0;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!has_frame_) return;
      frame = frame_.clone();
      frame_sequence = frame_sequence_;
    }

    readTrackbars(frame.cols, frame.rows);

    // TensorRT semantic segmentation is run only once per received frame and
    // only on the user-selected ROI. At this stage it is visualization-only;
    // Canny/Hough below are intentionally unchanged.
    if (semantic_enabled_ && segmenter_ && segmenter_->ready() &&
        frame_sequence != last_semantic_sequence_) {
      const cv::Rect roi(rx1_, ry1_, rx2_ - rx1_, ry2_ - ry1_);
      cv::Mat next_map;
      double inference_ms = 0.0;
      if (segmenter_->infer(frame(roi), next_map, inference_ms)) {
        semantic_class_map_ = std::move(next_map);
        semantic_inference_ms_ = inference_ms;
      } else {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                              "YOLO26-sem 추론 실패: %s",
                              segmenter_->status().c_str());
      }
      last_semantic_sequence_ = frame_sequence;
    }

    if (depth_enabled_ && depth_estimator_ && depth_estimator_->ready() &&
        frame_sequence != last_depth_sequence_) {
      const cv::Rect roi(rx1_, ry1_, rx2_ - rx1_, ry2_ - ry1_);
      cv::Mat next_depth;
      double inference_ms = 0.0;
      if (depth_estimator_->infer(frame(roi), next_depth, inference_ms)) {
        relative_depth_ = std::move(next_depth);
        depth_inference_ms_ = inference_ms;
      } else {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                              "Depth Anything V2 추론 실패: %s",
                              depth_estimator_->status().c_str());
      }
      last_depth_sequence_ = frame_sequence;
    }

    // ── 전처리 ──────────────────────────────────────────
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, {5, 5}, 0);

    // ROI 마스크 (사각형)
    cv::Mat roi_mask = cv::Mat::zeros(gray.size(), CV_8U);
    roi_mask(cv::Rect(rx1_, ry1_, rx2_ - rx1_, ry2_ - ry1_)) = 255;

    // Canny
    cv::Mat edges;
    cv::Canny(gray, edges, tb_.canny_low, tb_.canny_high);
    cv::bitwise_and(edges, roi_mask, edges);

    // HoughLinesP
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180,
                    tb_.hough_thresh, tb_.min_length, tb_.max_gap);

    // ── 수평 계열 라인 추출 ──────────────────────────────
    // |angle| < 45° → 수평 계열
    // 길이 가중 평균 각도 → yaw error
    // 길이 가중 평균 y중점 → 기울기 시각화 중심점
    float yaw_deg   = 0.0f;
    float mean_y    = 0.0f;
    bool  yaw_valid = false;
    size_t target_support_idx = lines.size();
    std::vector<size_t> horiz_idx;
    float w_sum = 0.0f, wa_sum = 0.0f, wy_sum = 0.0f;

    for (size_t i = 0; i < lines.size(); i++) {
      const auto &l = lines[i];
      const float dx = static_cast<float>(l[2] - l[0]);
      const float dy = static_cast<float>(l[3] - l[1]);
      float angle = std::atan2(dy, dx) * 180.0f / CV_PI;

      // 0° = 수평, ±90° = 수직. -90~+90 으로 정규화
      if (angle >  90.0f) angle -= 180.0f;
      if (angle < -90.0f) angle += 180.0f;

      // 수평 기준 절댓값 30° 이상인 선은 yaw 후보에서 제외
      if (std::abs(angle) >= 30.0f) continue;

      {
        const float len  = std::hypot(dx, dy);
        const float my   = (l[1] + l[3]) * 0.5f;
        horiz_idx.push_back(i);
        wa_sum += angle * len;
        wy_sum += my    * len;
        w_sum  += len;
      }
    }

    if (!horiz_idx.empty()) {
      yaw_deg   = wa_sum / w_sum;
      mean_y    = wy_sum / w_sum;
      yaw_valid = true;

      // Pick the real Hough segment that best represents the weighted
      // consensus. This is visualization/support; yaw still uses all valid
      // segments for stability.
      float best_score = std::numeric_limits<float>::max();
      const float roi_h = std::max(1, ry2_ - ry1_);
      const float roi_w = std::max(1, rx2_ - rx1_);
      for (size_t i : horiz_idx) {
        const auto &l = lines[i];
        const float dx = static_cast<float>(l[2] - l[0]);
        const float dy = static_cast<float>(l[3] - l[1]);
        float angle = std::atan2(dy, dx) * 180.0f / CV_PI;
        if (angle > 90.0f) angle -= 180.0f;
        if (angle < -90.0f) angle += 180.0f;
        const float mid_y = (l[1] + l[3]) * 0.5f;
        const float length = std::hypot(dx, dy);
        const float score = std::abs(angle - yaw_deg) / 30.0f +
                            std::abs(mid_y - mean_y) / roi_h -
                            0.20f * std::min(length / roi_w, 1.0f);
        if (score < best_score) {
          best_score = score;
          target_support_idx = i;
        }
      }
    }

    cv::Mat semantic_frame = frame.clone();
    drawSemantic(semantic_frame);
    drawLines(semantic_frame, lines, horiz_idx, target_support_idx,
              yaw_deg, mean_y, yaw_valid);
    drawEdges(edges);
    drawDepthComparison(frame, edges);
  }

private:
  struct LineAnalysis {
    std::vector<cv::Vec4i> lines;
    std::vector<size_t> horizontal_indices;
    size_t target_support_idx = 0;
    float yaw_deg = 0.0f;
    float mean_y = 0.0f;
    bool valid = false;
  };

  struct Trackbars {
    int canny_low     = 50;
    int canny_high    = 150;
    int hough_thresh  = 40;
    int min_length    = 60;
    int max_gap       = 15;
    int roi_top_pct   = 20;
    int roi_bot_pct   = 80;
    int roi_left_pct  = 20;
    int roi_right_pct = 80;
    int semantic_alpha_pct = 50;
    int depth_edge_thresh_pct = 15;
    int depth_near_floor_pct = 20;
    int depth_near_gamma_x10 = 20;
    int depth_near_cutoff_pct = 35;
  } tb_;

  // 픽셀 ROI (readTrackbars 에서 갱신)
  int rx1_ = 0, rx2_ = 640;
  int ry1_ = 0, ry2_ = 480;

  void initTrackbars() {
    cv::createTrackbar("Canny Low",    kWinLines, nullptr, 500, nullptr);
    cv::createTrackbar("Canny High",   kWinLines, nullptr, 500, nullptr);
    cv::createTrackbar("Hough Thresh", kWinLines, nullptr, 200, nullptr);
    cv::createTrackbar("Min Length",   kWinLines, nullptr, 400, nullptr);
    cv::createTrackbar("Max Gap",      kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("ROI Top %",    kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("ROI Bot %",    kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("ROI Left %",   kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("ROI Right %",  kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("Sem Alpha %",  kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("Depth Thr %",  kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("Near Floor %", kWinLines, nullptr, 100, nullptr);
    cv::createTrackbar("Near Gamma x10", kWinLines, nullptr, 50, nullptr);
    cv::createTrackbar("Near Cutoff %", kWinLines, nullptr, 100, nullptr);

    cv::setTrackbarPos("Canny Low",    kWinLines, tb_.canny_low);
    cv::setTrackbarPos("Canny High",   kWinLines, tb_.canny_high);
    cv::setTrackbarPos("Hough Thresh", kWinLines, tb_.hough_thresh);
    cv::setTrackbarPos("Min Length",   kWinLines, tb_.min_length);
    cv::setTrackbarPos("Max Gap",      kWinLines, tb_.max_gap);
    cv::setTrackbarPos("ROI Top %",    kWinLines, tb_.roi_top_pct);
    cv::setTrackbarPos("ROI Bot %",    kWinLines, tb_.roi_bot_pct);
    cv::setTrackbarPos("ROI Left %",   kWinLines, tb_.roi_left_pct);
    cv::setTrackbarPos("ROI Right %",  kWinLines, tb_.roi_right_pct);
    cv::setTrackbarPos("Sem Alpha %",  kWinLines, tb_.semantic_alpha_pct);
    cv::setTrackbarPos("Depth Thr %",  kWinLines, tb_.depth_edge_thresh_pct);
    cv::setTrackbarPos("Near Floor %", kWinLines, tb_.depth_near_floor_pct);
    cv::setTrackbarPos("Near Gamma x10", kWinLines, tb_.depth_near_gamma_x10);
    cv::setTrackbarPos("Near Cutoff %", kWinLines, tb_.depth_near_cutoff_pct);
  }

  void readTrackbars(int img_w, int img_h) {
    tb_.canny_low     = cv::getTrackbarPos("Canny Low",    kWinLines);
    tb_.canny_high    = cv::getTrackbarPos("Canny High",   kWinLines);
    tb_.hough_thresh  = cv::getTrackbarPos("Hough Thresh", kWinLines);
    tb_.min_length    = cv::getTrackbarPos("Min Length",   kWinLines);
    tb_.max_gap       = cv::getTrackbarPos("Max Gap",      kWinLines);
    tb_.roi_top_pct   = cv::getTrackbarPos("ROI Top %",    kWinLines);
    tb_.roi_bot_pct   = cv::getTrackbarPos("ROI Bot %",    kWinLines);
    tb_.roi_left_pct  = cv::getTrackbarPos("ROI Left %",   kWinLines);
    tb_.roi_right_pct = cv::getTrackbarPos("ROI Right %",  kWinLines);
    tb_.semantic_alpha_pct = cv::getTrackbarPos("Sem Alpha %", kWinLines);
    tb_.depth_edge_thresh_pct = cv::getTrackbarPos("Depth Thr %", kWinLines);
    tb_.depth_near_floor_pct = cv::getTrackbarPos("Near Floor %", kWinLines);
    tb_.depth_near_gamma_x10 = cv::getTrackbarPos("Near Gamma x10", kWinLines);
    tb_.depth_near_cutoff_pct = cv::getTrackbarPos("Near Cutoff %", kWinLines);

    const float top   = std::clamp(tb_.roi_top_pct  / 100.0f, 0.0f, 0.98f);
    const float bot   = std::clamp(tb_.roi_bot_pct  / 100.0f, top + 0.01f, 1.0f);
    const float left  = std::clamp(tb_.roi_left_pct / 100.0f, 0.0f, 0.98f);
    const float right = std::clamp(tb_.roi_right_pct/ 100.0f, left + 0.01f, 1.0f);

    ry1_    = static_cast<int>(top   * img_h);
    ry2_    = static_cast<int>(bot   * img_h);
    rx1_    = static_cast<int>(left  * img_w);
    rx2_    = static_cast<int>(right * img_w);
  }

  static const std::array<const char *, 19> &cityscapesNames() {
    static const std::array<const char *, 19> names = {
      "road", "sidewalk", "building", "wall", "fence", "pole",
      "traffic light", "traffic sign", "vegetation", "terrain", "sky",
      "person", "rider", "car", "truck", "bus", "train", "motorcycle",
      "bicycle"
    };
    return names;
  }

  static const std::array<cv::Vec3b, 19> &cityscapesPalette() {
    // Official Cityscapes colors converted from RGB to OpenCV BGR.
    static const std::array<cv::Vec3b, 19> colors = {
      cv::Vec3b{128, 64, 128}, cv::Vec3b{232, 35, 244},
      cv::Vec3b{70, 70, 70}, cv::Vec3b{156, 102, 102},
      cv::Vec3b{153, 153, 190}, cv::Vec3b{153, 153, 153},
      cv::Vec3b{30, 170, 250}, cv::Vec3b{0, 220, 220},
      cv::Vec3b{35, 142, 107}, cv::Vec3b{152, 251, 152},
      cv::Vec3b{180, 130, 70}, cv::Vec3b{60, 20, 220},
      cv::Vec3b{0, 0, 255}, cv::Vec3b{142, 0, 0},
      cv::Vec3b{70, 0, 0}, cv::Vec3b{100, 60, 0},
      cv::Vec3b{100, 80, 0}, cv::Vec3b{230, 0, 0},
      cv::Vec3b{32, 11, 119}
    };
    return colors;
  }

  cv::Mat colorizeSemantic(const cv::Mat &class_map) const {
    cv::Mat color(class_map.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    const auto &palette = cityscapesPalette();
    for (int y = 0; y < class_map.rows; ++y) {
      const std::uint8_t *src = class_map.ptr<std::uint8_t>(y);
      cv::Vec3b *dst = color.ptr<cv::Vec3b>(y);
      for (int x = 0; x < class_map.cols; ++x) {
        if (src[x] < palette.size()) dst[x] = palette[src[x]];
      }
    }
    return color;
  }

  void drawSemantic(cv::Mat &frame) const {
    if (!semantic_enabled_) return;

    if (semantic_class_map_.empty() ||
        semantic_class_map_.cols != rx2_ - rx1_ ||
        semantic_class_map_.rows != ry2_ - ry1_) {
      cv::Mat waiting = frame.clone();
      const std::string text = segmenter_ && segmenter_->ready()
          ? "Semantic: waiting for frame"
          : "Semantic unavailable (see ROS log)";
      cv::putText(waiting, text, {12, 32}, cv::FONT_HERSHEY_SIMPLEX,
                  0.7, {0, 0, 255}, 2, cv::LINE_AA);
      cv::imshow(kWinSemantic, waiting);
      return;
    }

    const cv::Mat color = colorizeSemantic(semantic_class_map_);
    const cv::Rect roi(rx1_, ry1_, rx2_ - rx1_, ry2_ - ry1_);
    const double alpha = tb_.semantic_alpha_pct / 100.0;
    cv::addWeighted(color, alpha, frame(roi), 1.0 - alpha, 0.0, frame(roi));

    std::array<int, 19> counts{};
    for (int y = 0; y < semantic_class_map_.rows; ++y) {
      const std::uint8_t *row = semantic_class_map_.ptr<std::uint8_t>(y);
      for (int x = 0; x < semantic_class_map_.cols; ++x) {
        if (row[x] < counts.size()) ++counts[row[x]];
      }
    }
    const int dominant = static_cast<int>(
        std::distance(counts.begin(), std::max_element(counts.begin(), counts.end())));
    const double ratio = 100.0 * counts[dominant] /
                         std::max<std::size_t>(1, semantic_class_map_.total());

    cv::Mat semantic_view(frame.size(), CV_8UC3, cv::Scalar(25, 25, 25));
    frame(roi).copyTo(semantic_view(roi));
    cv::rectangle(semantic_view, roi, {255, 255, 255}, 1);
    char info[160];
    std::snprintf(info, sizeof(info), "YOLO26-sem | %s %.1f%% | %.1f ms",
                  cityscapesNames()[dominant], ratio, semantic_inference_ms_);
    cv::putText(semantic_view, info, {10, 30}, cv::FONT_HERSHEY_SIMPLEX,
                0.65, {255, 255, 255}, 2, cv::LINE_AA);
    cv::imshow(kWinSemantic, semantic_view);
  }

  static cv::Mat normalizeRelativeDepth(const cv::Mat &depth) {
    double min_value = 0.0, max_value = 0.0;
    cv::minMaxLoc(depth, &min_value, &max_value);
    cv::Mat normalized;
    if (max_value - min_value < 1e-6) {
      normalized = cv::Mat::zeros(depth.size(), CV_32F);
    } else {
      depth.convertTo(normalized, CV_32F, 1.0 / (max_value - min_value),
                      -min_value / (max_value - min_value));
    }
    return normalized;
  }

  static cv::Mat makeDepthPanel(const cv::Mat &image,
                                const std::string &label,
                                cv::Size panel_size) {
    cv::Mat bgr;
    if (image.channels() == 1) cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    else bgr = image;
    cv::Mat panel;
    cv::resize(bgr, panel, panel_size, 0.0, 0.0, cv::INTER_AREA);
    cv::rectangle(panel, {0, 0}, {panel.cols, 32}, {0, 0, 0}, -1);
    cv::putText(panel, label, {8, 23}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                {255, 255, 255}, 1, cv::LINE_AA);
    return panel;
  }

  LineAnalysis analyzeLineMap(const cv::Mat &edge_map,
                              const cv::Mat &pixel_weight = cv::Mat()) const {
    LineAnalysis result;
    cv::HoughLinesP(edge_map, result.lines, 1, CV_PI / 180,
                    tb_.hough_thresh, tb_.min_length, tb_.max_gap);
    result.target_support_idx = result.lines.size();

    float total_weight = 0.0f;
    float weighted_angle = 0.0f;
    float weighted_y = 0.0f;
    for (size_t i = 0; i < result.lines.size(); ++i) {
      const auto &line = result.lines[i];
      const float dx = static_cast<float>(line[2] - line[0]);
      const float dy = static_cast<float>(line[3] - line[1]);
      float angle = std::atan2(dy, dx) * 180.0f / CV_PI;
      if (angle > 90.0f) angle -= 180.0f;
      if (angle < -90.0f) angle += 180.0f;
      if (std::abs(angle) >= 30.0f) continue;

      float support = 1.0f;
      if (!pixel_weight.empty()) {
        cv::LineIterator iterator(pixel_weight, {line[0], line[1]},
                                  {line[2], line[3]}, 8);
        float sum = 0.0f;
        for (int p = 0; p < iterator.count; ++p, ++iterator) {
          sum += pixel_weight.at<float>(iterator.pos());
        }
        support = iterator.count > 0 ? sum / iterator.count : 0.0f;
      }

      const float length = std::hypot(dx, dy);
      const float weight = length * std::max(support, 0.01f);
      const float mid_y = (line[1] + line[3]) * 0.5f;
      result.horizontal_indices.push_back(i);
      weighted_angle += angle * weight;
      weighted_y += mid_y * weight;
      total_weight += weight;
    }

    if (result.horizontal_indices.empty() || total_weight <= 0.0f) return result;
    result.yaw_deg = weighted_angle / total_weight;
    result.mean_y = weighted_y / total_weight;
    result.valid = true;

    float best_score = std::numeric_limits<float>::max();
    const float roi_h = std::max(1, ry2_ - ry1_);
    const float roi_w = std::max(1, rx2_ - rx1_);
    for (size_t i : result.horizontal_indices) {
      const auto &line = result.lines[i];
      const float dx = static_cast<float>(line[2] - line[0]);
      const float dy = static_cast<float>(line[3] - line[1]);
      float angle = std::atan2(dy, dx) * 180.0f / CV_PI;
      if (angle > 90.0f) angle -= 180.0f;
      if (angle < -90.0f) angle += 180.0f;
      const float mid_y = (line[1] + line[3]) * 0.5f;
      const float length = std::hypot(dx, dy);
      const float score = std::abs(angle - result.yaw_deg) / 30.0f +
                          std::abs(mid_y - result.mean_y) / roi_h -
                          0.20f * std::min(length / roi_w, 1.0f);
      if (score < best_score) {
        best_score = score;
        result.target_support_idx = i;
      }
    }
    return result;
  }

  void drawLineAnalysisWindow(const char *window_name, const cv::Mat &edge_map,
                              const LineAnalysis &analysis,
                              const std::string &title) const {
    cv::Mat view;
    cv::cvtColor(edge_map, view, cv::COLOR_GRAY2BGR);
    std::vector<bool> horizontal(analysis.lines.size(), false);
    for (size_t i : analysis.horizontal_indices) horizontal[i] = true;

    for (size_t i = 0; i < analysis.lines.size(); ++i) {
      const auto &line = analysis.lines[i];
      if (!horizontal[i]) {
        cv::line(view, {line[0], line[1]}, {line[2], line[3]},
                 {70, 70, 70}, 1, cv::LINE_AA);
      }
    }
    for (size_t i : analysis.horizontal_indices) {
      if (i == analysis.target_support_idx) continue;
      const auto &line = analysis.lines[i];
      cv::line(view, {line[0], line[1]}, {line[2], line[3]},
               {0, 220, 255}, 2, cv::LINE_AA);
    }
    if (analysis.target_support_idx < analysis.lines.size()) {
      const auto &line = analysis.lines[analysis.target_support_idx];
      cv::line(view, {line[0], line[1]}, {line[2], line[3]},
               {255, 255, 0}, 5, cv::LINE_AA);
    }
    if (analysis.valid) {
      const int cx = (rx1_ + rx2_) / 2;
      const int cy = static_cast<int>(analysis.mean_y);
      const float radians = analysis.yaw_deg * CV_PI / 180.0f;
      const int arm = static_cast<int>((rx2_ - rx1_) * 0.45f);
      cv::Point p1(cx - static_cast<int>(std::cos(radians) * arm),
                   cy - static_cast<int>(std::sin(radians) * arm));
      cv::Point p2(cx + static_cast<int>(std::cos(radians) * arm),
                   cy + static_cast<int>(std::sin(radians) * arm));
      if (cv::clipLine(cv::Rect(rx1_, ry1_, rx2_ - rx1_, ry2_ - ry1_), p1, p2)) {
        cv::line(view, p1, p2, {255, 0, 255}, 4, cv::LINE_AA);
      }
    }
    cv::rectangle(view, {rx1_, ry1_}, {rx2_, ry2_}, {100, 100, 255}, 2);
    cv::rectangle(view, {0, 0}, {view.cols, 36}, {0, 0, 0}, -1);
    const std::string info = analysis.valid
        ? cv::format("%s | ALIGN %+.1f deg | candidates %zu",
                     title.c_str(), analysis.yaw_deg,
                     analysis.horizontal_indices.size())
        : title + " | ALIGN N/A";
    cv::putText(view, info, {10, 26}, cv::FONT_HERSHEY_SIMPLEX, 0.62,
                {255, 255, 255}, 2, cv::LINE_AA);
    cv::imshow(window_name, view);
  }

  void drawDepthComparison(const cv::Mat &frame, const cv::Mat &rgb_edges) const {
    if (!depth_enabled_) return;
    const cv::Rect roi(rx1_, ry1_, rx2_ - rx1_, ry2_ - ry1_);
    if (relative_depth_.empty() || relative_depth_.size() != roi.size()) {
      cv::Mat waiting = frame.clone();
      const std::string text = depth_estimator_ && depth_estimator_->ready()
          ? "Depth: waiting for frame"
          : "Depth unavailable (see ROS log)";
      cv::putText(waiting, text, {12, 32}, cv::FONT_HERSHEY_SIMPLEX,
                  0.7, {0, 0, 255}, 2, cv::LINE_AA);
      cv::imshow(kWinDepthCompare, waiting);
      cv::imshow(kWinNearWeightedEdge, waiting);
      cv::imshow(kWinDepthOnlyEdge, waiting);
      return;
    }

    const cv::Mat depth_norm = normalizeRelativeDepth(relative_depth_);
    cv::Mat depth_u8, depth_color;
    depth_norm.convertTo(depth_u8, CV_8U, 255.0);
    cv::applyColorMap(depth_u8, depth_color, cv::COLORMAP_INFERNO);

    cv::Mat smooth, grad_x, grad_y, magnitude;
    cv::GaussianBlur(depth_norm, smooth, {5, 5}, 0.0);
    cv::Sobel(smooth, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(smooth, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, magnitude);
    double grad_max = 0.0;
    cv::minMaxLoc(magnitude, nullptr, &grad_max);
    cv::Mat depth_gradient;
    magnitude.convertTo(depth_gradient, CV_8U,
                        grad_max > 1e-6 ? 255.0 / grad_max : 0.0);

    cv::Mat depth_binary;
    const int depth_threshold = cv::saturate_cast<std::uint8_t>(
        tb_.depth_edge_thresh_pct * 255.0 / 100.0);
    cv::threshold(depth_gradient, depth_binary, depth_threshold, 255,
                  cv::THRESH_BINARY);

    const cv::Mat rgb_roi_edges = rgb_edges(roi);
    cv::Mat near_weighted(rgb_roi_edges.size(), CV_8U, cv::Scalar(0));
    const float near_floor = tb_.depth_near_floor_pct / 100.0f;
    const float near_gamma = std::max(0.1f, tb_.depth_near_gamma_x10 / 10.0f);
    const float near_cutoff = tb_.depth_near_cutoff_pct / 100.0f;
    cv::Mat nonlinear_proximity;
    cv::pow(depth_norm, near_gamma, nonlinear_proximity);
    for (int y = 0; y < near_weighted.rows; ++y) {
      const std::uint8_t *edge = rgb_roi_edges.ptr<std::uint8_t>(y);
      const float *near_depth = nonlinear_proximity.ptr<float>(y);
      const float *raw_proximity = depth_norm.ptr<float>(y);
      std::uint8_t *dst = near_weighted.ptr<std::uint8_t>(y);
      for (int x = 0; x < near_weighted.cols; ++x) {
        if (!edge[x]) continue;
        if (raw_proximity[x] < near_cutoff) continue;
        // DAv2 relative output is inverse depth: larger means relatively near.
        // Keep a configurable floor so far Canny edges remain faintly visible.
        const float weight = near_floor + (1.0f - near_floor) * near_depth[x];
        dst[x] = cv::saturate_cast<std::uint8_t>(255.0f * weight);
      }
    }

    cv::Mat near_full = cv::Mat::zeros(frame.size(), CV_8U);
    near_weighted.copyTo(near_full(roi));
    cv::Mat proximity_weight_roi;
    nonlinear_proximity.convertTo(proximity_weight_roi, CV_32F,
                                  1.0f - near_floor, near_floor);
    cv::Mat proximity_weight_full = cv::Mat::zeros(frame.size(), CV_32F);
    proximity_weight_roi.copyTo(proximity_weight_full(roi));
    // HoughLinesP is binary/non-zero based, so it must consume the genuinely
    // proximity-filtered edge map rather than the original RGB Canny map.
    const LineAnalysis near_analysis = analyzeLineMap(
        near_full, proximity_weight_full);
    drawLineAnalysisWindow(kWinNearWeightedEdge, near_full, near_analysis,
                           cv::format("Canny x near | cutoff %d%% floor %d%% gamma %.1f",
                                      tb_.depth_near_cutoff_pct,
                                      tb_.depth_near_floor_pct, near_gamma));

    cv::Mat depth_only_full = cv::Mat::zeros(frame.size(), CV_8U);
    depth_binary.copyTo(depth_only_full(roi));
    const LineAnalysis depth_analysis = analyzeLineMap(depth_only_full);
    drawLineAnalysisWindow(kWinDepthOnlyEdge, depth_only_full, depth_analysis,
                           "Depth-only Sobel edge");

    const cv::Size panel_size(300, 220);
    std::vector<cv::Mat> panels = {
      makeDepthPanel(frame(roi), "RGB ROI", panel_size),
      makeDepthPanel(depth_color, "DAv2 relative inverse depth", panel_size),
      makeDepthPanel(rgb_roi_edges, "RGB Canny", panel_size),
      makeDepthPanel(near_weighted,
                     cv::format("Canny x near (cut %d%%, gamma %.1f)",
                                tb_.depth_near_cutoff_pct, near_gamma), panel_size),
      makeDepthPanel(depth_binary,
                     cv::format("Depth-only edge (thr %d%%)",
                                tb_.depth_edge_thresh_pct), panel_size)
    };
    cv::Mat comparison;
    cv::hconcat(panels, comparison);
    cv::putText(comparison,
                cv::format("Depth Anything V2 Small TensorRT: %.1f ms | relative depth, not meters",
                           depth_inference_ms_),
                {10, comparison.rows - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                {255, 255, 255}, 1, cv::LINE_AA);
    cv::imshow(kWinDepthCompare, comparison);
  }

  void drawLines(const cv::Mat &frame,
                 const std::vector<cv::Vec4i> &lines,
                 const std::vector<size_t> &horiz_idx,
                 size_t target_support_idx,
                 float yaw_deg, float mean_y, bool yaw_valid) {
    cv::Mat disp = frame.clone();

    // ROI 박스
    cv::rectangle(disp, {rx1_, ry1_}, {rx2_, ry2_}, {100, 100, 255}, 2);

    // 비수평 라인 (회색)
    std::vector<bool> is_horiz(lines.size(), false);
    for (size_t i : horiz_idx) is_horiz[i] = true;
    for (size_t i = 0; i < lines.size(); i++) {
      if (!is_horiz[i]) {
        const auto &l = lines[i];
        cv::line(disp, {l[0], l[1]}, {l[2], l[3]}, {70, 70, 70}, 1, cv::LINE_AA);
      }
    }

    // 수평 계열 라인 (노란색)
    for (size_t i : horiz_idx) {
      if (i == target_support_idx) continue;
      const auto &l = lines[i];
      cv::line(disp, {l[0], l[1]}, {l[2], l[3]}, {0, 220, 255}, 2, cv::LINE_AA);
    }

    // 합의 각도/높이를 가장 잘 대표하는 실제 Hough segment (청록색)
    if (target_support_idx < lines.size()) {
      const auto &l = lines[target_support_idx];
      cv::line(disp, {l[0], l[1]}, {l[2], l[3]}, {255, 255, 0}, 5, cv::LINE_AA);
      const cv::Point label_at(std::min(l[0], l[2]), std::min(l[1], l[3]) - 8);
      cv::putText(disp, "TARGET SUPPORT", label_at,
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 0}, 2, cv::LINE_AA);
    }

    if (yaw_valid) {
      // 감지된 수평선 평균 y (초록 실선)
      cv::line(disp, {rx1_, static_cast<int>(mean_y)},
                     {rx2_, static_cast<int>(mean_y)},
               {0, 200, 80}, 1, cv::LINE_AA);

      // 최종 정렬 기준: 모든 유효 segment의 길이 가중 합의선 (자홍색)
      const int cx = (rx1_ + rx2_) / 2;
      const int cy = static_cast<int>(mean_y);
      const float rad = yaw_deg * CV_PI / 180.0f;
      const int arm = static_cast<int>((rx2_ - rx1_) * 0.45f);
      const int ex  = static_cast<int>(std::cos(rad) * arm);
      const int ey  = static_cast<int>(std::sin(rad) * arm);
      cv::Point p1(cx - ex, cy - ey), p2(cx + ex, cy + ey);
      if (cv::clipLine(cv::Rect(rx1_, ry1_, rx2_ - rx1_, ry2_ - ry1_), p1, p2)) {
        cv::line(disp, p1, p2, {255, 0, 255}, 4, cv::LINE_AA);
        cv::putText(disp, "ALIGN CONSENSUS", {p1.x, std::max(18, p1.y - 8)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 0, 255}, 2,
                    cv::LINE_AA);
      }

      // ── 텍스트 ───────────────────────────────────────
      const cv::Scalar yaw_col = std::abs(yaw_deg) < 3.0f
                                 ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 60, 255);
      char buf[80];
      // yaw error: ±90° 기준 (0° = 완전 수평 정렬)
      std::snprintf(buf, sizeof(buf), "Yaw error : %+.1f deg  (+-90 range)", yaw_deg);
      cv::putText(disp, buf, {10, 38},
                  cv::FONT_HERSHEY_SIMPLEX, 0.75, yaw_col, 2, cv::LINE_AA);

      std::snprintf(buf, sizeof(buf), "Edge mean_y: %.0f px", mean_y);
      cv::putText(disp, buf, {10, 65},
                  cv::FONT_HERSHEY_SIMPLEX, 0.65, {0, 200, 80}, 1, cv::LINE_AA);

      drawYawBar(disp, yaw_deg);
    } else {
      cv::putText(disp, "Yaw: N/A", {10, 38},
                  cv::FONT_HERSHEY_SIMPLEX, 0.85, {100, 100, 100}, 1, cv::LINE_AA);
    }

    // ROI 좌표 + 라인 수
    char info[120];
    std::snprintf(info, sizeof(info),
                  "ROI x[%d,%d] y[%d,%d] | lines:%zu horiz:%zu | q:종료",
                  rx1_, rx2_, ry1_, ry2_,
                  lines.size(), horiz_idx.size());
    cv::putText(disp, info, {8, disp.rows - 8},
                cv::FONT_HERSHEY_SIMPLEX, 0.42, {160, 160, 160}, 1, cv::LINE_AA);

    cv::imshow(kWinLines, disp);
  }

  void drawYawBar(cv::Mat &img, float yaw_deg) const {
    const int bw = 240, bh = 14;
    const int bx = img.cols / 2 - bw / 2;
    const int by = img.rows - 28;

    cv::rectangle(img, {bx, by}, {bx + bw, by + bh}, {50, 50, 50}, -1);
    cv::rectangle(img, {bx, by}, {bx + bw, by + bh}, {130, 130, 130}, 1);

    const float clamped = std::clamp(yaw_deg, -45.0f, 45.0f);
    const int   mid     = bx + bw / 2;
    const int   fill_w  = static_cast<int>(clamped / 45.0f * (bw / 2));
    const int   fill_x  = std::min(mid, mid + fill_w);
    const int   fill_xe = std::max(mid, mid + fill_w);
    if (fill_x != fill_xe) {
      const cv::Scalar c = std::abs(yaw_deg) < 3.0f
                           ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 60, 255);
      cv::rectangle(img, {fill_x, by + 1}, {fill_xe, by + bh - 1}, c, -1);
    }
    cv::line(img, {mid, by}, {mid, by + bh}, {255, 255, 255}, 1);
    cv::putText(img, "-45", {bx - 28, by + 11},
                cv::FONT_HERSHEY_SIMPLEX, 0.35, {150,150,150}, 1);
    cv::putText(img, "+45", {bx + bw + 3, by + 11},
                cv::FONT_HERSHEY_SIMPLEX, 0.35, {150,150,150}, 1);
  }

  void drawEdges(const cv::Mat &edges) {
    cv::Mat colored;
    cv::cvtColor(edges, colored, cv::COLOR_GRAY2BGR);
    cv::rectangle(colored, {rx1_, ry1_}, {rx2_, ry2_}, {100, 100, 255}, 2);
    cv::imshow(kWinEdges, colored);
  }

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
  std::string topic_;
  bool semantic_enabled_ = true;
  std::string semantic_engine_;
  std::unique_ptr<TrtSemanticSegmenter> segmenter_;
  cv::Mat semantic_class_map_;
  double semantic_inference_ms_ = 0.0;
  std::uint64_t frame_sequence_ = 0;
  std::uint64_t last_semantic_sequence_ = std::numeric_limits<std::uint64_t>::max();
  bool depth_enabled_ = true;
  std::string depth_engine_;
  std::unique_ptr<TrtDepthEstimator> depth_estimator_;
  cv::Mat relative_depth_;
  double depth_inference_ms_ = 0.0;
  std::uint64_t last_depth_sequence_ = std::numeric_limits<std::uint64_t>::max();
  cv::Mat frame_;
  std::mutex mtx_;
  bool has_frame_ = false;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<EdgeYawViewer>();

  while (rclcpp::ok()) {
    rclcpp::spin_some(node);
    node->update();
    int key = cv::waitKey(10);
    if (key == 'q' || key == 27) break;
    if (key == 's') node->saveYaml();
  }

  cv::destroyAllWindows();
  rclcpp::shutdown();
  return 0;
}
